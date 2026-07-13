import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from dataclasses import dataclass
from typing import List
from pyvista import PolyData
from relevant_PULSE_files.jax_kernels import (
    Pose, pulse, pulse_batch, qimg,
    interpolate
)
from relevant_PULSE_files.jax_kernels import _quat_slerp
PULSE_TIME_SPACING_DEFAULT = 0.01  # seconds


@dataclass
class PulseParams:
    sigma: float        
    a: float 
    ref_dist: float 
    resolution: int 
    fov: float 
    volumetric_flow_rate: float


class Pulse:
    def __init__(self, sigma: float,
                a: float, ref_dist: float ,
                resolution: int, fov: float ,
                volumetric_flow_rate: float,
                params: PulseParams = None
                ):
        if params is not None:
            self.params = params
        else:
            self.params = PulseParams(
                sigma=sigma, a=a, ref_dist=ref_dist,
                resolution=resolution, fov=fov,
                volumetric_flow_rate=volumetric_flow_rate
            )
        # set by load_mesh
        self.face_v0 = None
        self.face_v1 = None
        self.face_v2 = None
        self.face_normals = None
        self.areas = None
        self.n_faces = None
        self.points = None
        self.faces = None
        self.mesh_file = None

    def load_mesh(self, mesh):
        if not isinstance(mesh, PolyData):
            self.mesh_file = mesh
            mesh = pv.read(str(mesh)).triangulate()
        else:
            mesh = mesh.triangulate()

        mesh = mesh.compute_normals(cell_normals=True, point_normals=False)
        
        # mesh = mesh.compute_normals(cell_normals=False, point_normals=True)

        self.points = np.array(mesh.points)
        self.faces = np.array(mesh.faces).reshape(-1, 4)[:, 1:]  # (n_faces, 3)
        self.n_faces = len(self.faces)
        self.n_points = len(self.points)

        self.face_v0 = jnp.array(self.points[self.faces[:, 0]])
        self.face_v1 = jnp.array(self.points[self.faces[:, 1]])
        self.face_v2 = jnp.array(self.points[self.faces[:, 2]])
        self.face_normals = jnp.array(np.array(mesh.cell_data['Normals']))
        # self.point_normals = jnp.array(mesh.point_data['Normals'])

        self._calculate_triangle_areas()

        v_area = np.zeros(self.n_points)
        # Get the area of every face
        face_areas = mesh.compute_cell_sizes().cell_data['Area']
        # Get the vertex indices for every face
        faces = mesh.faces.reshape(-1, 4)[:, 1:] 

        # Distribute 1/3 of each face area to its 3 vertices
        for i in range(3):
            np.add.at(v_area, faces[:, i], face_areas / 3.0)

        self.vertex_areas = jnp.array(v_area)

    def _calculate_triangle_areas(self):
        edge1 = self.points[self.faces[:, 1]] - self.points[self.faces[:, 0]]
        edge2 = self.points[self.faces[:, 2]] - self.points[self.faces[:, 0]]
        cross = np.cross(edge1, edge2)
        self.areas = jnp.array(0.5 * np.linalg.norm(cross, axis=1))

    def get_scaled_thicknesses(self, thicknesses: np.ndarray) -> np.ndarray:
        return thicknesses / self.areas

    # def get_scaled_thicknesses(self, thicknesses):
    #     return thicknesses / self.vertex_areas

    def evaluate_pulse(self, pose: Pose, rng_key=None) -> np.ndarray:
        return self.evaluate_pulses([pose], rng_key)

    def evaluate_pulses(self, poses: List[Pose], rng_key=None) -> np.ndarray:
        if not all(isinstance(p, Pose) for p in poses):
            raise ValueError("poses must be a list of Pose objects.")
        
        return np.array(pulse(
            poses,
            self.params.sigma,
            self.params.a,
            self.params.ref_dist,
            self.params.resolution,
            self.face_v0,
            self.face_v1,
            self.face_v2,
            self.face_normals,
            self.n_faces,
            fov=self.params.fov,
        ))

    def evaluate_pulses_masked(self, pose_traj: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:

            def deposit_one(single_pose):
                # Split the 7D vector into position (3,) and orientation quaternion (4,)
                position = single_pose[0:3]
                rotation = single_pose[3:7]
                return deposit(
                    position, 
                    rotation, 
                    self.params.sigma, 
                    self.params.a, 
                    self.params.ref_dist, 
                    self.params.resolution,
                    self.face_v0, 
                    self.face_v1, 
                    self.face_v2, 
                    self.face_normals, 
                    self.n_faces, 
                    fov=self.params.fov
                )

            # Vectorize deposit across the max_substeps dimension
            all_thicknesses = jax.vmap(deposit_one)(pose_traj)

            # Expand mask dimensions from (max_substeps,) to (max_substeps, 1) for broadcasting
            # If valid_mask is False at a index, that entire face thickness row gets zeroed out
            masked_thicknesses = jnp.where(valid_mask[:, jnp.newaxis], all_thicknesses, 0.0)

            # Accumulate contributions across the trajectory steps dimension
            return jnp.sum(masked_thicknesses, axis=0)

    def evaluate_pulses_with_dwells(self, poses: List[List[Pose]], dwell_times: list,
                                     pulse_time_spacing=PULSE_TIME_SPACING_DEFAULT) -> np.ndarray:
        interpolated_poses, _, _ = self.interpolate_poses_by_dwell(poses, dwell_times, pulse_time_spacing)
        return self.evaluate_pulses(interpolated_poses)

    def interp_pose(self, target_time: float, poses: List[Pose], global_times: list,
                    time_start_idx: int) -> Pose:
        time_offset = 0
        while time_offset < len(poses) and global_times[time_start_idx + time_offset] < target_time:
            time_offset += 1
        if time_offset == len(poses):
            return poses[-1]
        if time_offset == 0:
            return poses[0]

        pose_before = poses[time_offset - 1]
        pose_after = poses[time_offset]
        time_before = global_times[time_start_idx + time_offset - 1]
        time_after = global_times[time_start_idx + time_offset]
        dt = time_after - time_before
        ratio = 0.0 if dt <= 0 else (target_time - time_before) / dt

        p_interp = pose_before.position * (1 - ratio) + pose_after.position * ratio
        # slerp rotation
        
        r_interp = _quat_slerp(pose_before.rotation, pose_after.rotation, ratio)
        return Pose(position=p_interp, rotation=r_interp)

    def interpolate_poses_by_dwell(self, paths: List[List[Pose]], dwell_times: List[float],
                                    pulse_time_spacing=PULSE_TIME_SPACING_DEFAULT):
        global_times = []
        cumulative_time = 0.0
        if len(dwell_times) != sum([len(p) for p in paths]):
            idx = 0
            for path_poses in paths:
                global_times.append(cumulative_time)
                for _ in range(len(path_poses) - 1):
                    cumulative_time += dwell_times[idx]
                    global_times.append(cumulative_time)
                    idx += 1
        else:
            for dt in dwell_times:
                cumulative_time += dt
                global_times.append(cumulative_time)

        interpolated_poses = []
        dwell_out = []
        pulse_indices = []
        interp_idx_counter = 0
        start_time_idx = 0

        for path_poses in paths:
            num_poses = len(path_poses)
            for i_pulse in range(num_poses - 1):
                pulse_indices_temp = []
                start_time = global_times[start_time_idx + i_pulse]
                end_time = global_times[start_time_idx + i_pulse + 1]
                next_t = start_time + pulse_time_spacing / 2
                if next_t >= end_time:
                    next_t = end_time

                interpolated_poses.append(self.interp_pose(next_t, path_poses, global_times, start_time_idx))
                dwell_out.append(next_t - start_time)
                pulse_indices_temp.append(interp_idx_counter)
                interp_idx_counter += 1

                while next_t + pulse_time_spacing < end_time:
                    next_t += pulse_time_spacing
                    interpolated_poses.append(self.interp_pose(next_t, path_poses, global_times, start_time_idx))
                    dwell_out.append(pulse_time_spacing)
                    pulse_indices_temp.append(interp_idx_counter)
                    interp_idx_counter += 1

                pulse_indices.append(pulse_indices_temp)
            start_time_idx += num_poses

        return interpolated_poses, dwell_out, pulse_indices

    # def get_qimg(self, pose: Pose, fov: float, resolution: int,
    #              thicknesses, samples_per_pixel: int = 1000, rng_key=None) -> np.ndarray:
    #     if rng_key is None:
    #         rng_key = jax.random.PRNGKey(0)
    #     thicknesses = jnp.array(thicknesses)
    #     return np.array(qimg(
    #         pose, fov, resolution, samples_per_pixel,
    #         self.face_v0, self.face_v1, self.face_v2,
    #         self.face_normals, thicknesses, rng_key
    #     ))

    def get_qimg(self, error_map: jnp.ndarray, ee_pos: jnp.ndarray, ee_xmat: jnp.ndarray, fov, resolution) -> jnp.ndarray:
        return qimg(
            self.face_v0, self.face_v1, self.face_v2,
            error_map,
            ee_pos, ee_xmat,
            fov_deg=fov,
            resolution=resolution,
        )