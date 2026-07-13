"""
POC: JAX-PULSE Spray Cooling on 3D Mesh

Loads refined_plate.obj and simulates spray cooling using JAX-PULSE
projective rasterization. deposit() computes spray intensity per triangle
face which becomes h_local — the convective heat transfer coefficient.
Heat equation evolves temperature per triangle face over time.

Physics:
    dT/dt = alpha * laplacian_approx(T) - (h_local / rho_c) * (T - T_ambient)

    h_local per face = h_ambient + deposit(pose) * h_spray_scale
"""

import sys
sys.path.insert(0, '/mnt/c/Users/Owner/Downloads/jax-pulse-master/jax-pulse-master')
sys.path.insert(0, '/mnt/c/Users/Owner/Downloads/jax-pulse-master/jax-pulse-master/lib')

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import time
from collections import defaultdict

from jax_kernels import Pose, deposit
from jax_pulse import Pulse, PulseParams

# =============================================================================
# parameters
# =============================================================================
T_initial     = 900.0
k             = 50.0
T_ambient     = 25.0
h_spray_scale = 50000.0
h_ambient     = 10.0
c             = 500.0
rho           = 7800.0
rho_c         = rho * c
alpha         = k / rho_c

dt            = 0.1
t_end         = 600.0
steps         = int(t_end / dt)

# JAX-PULSE parameters
sigma         = 0.8
a             = 1.0
ref_dist      = 1.0
resolution    = 64
fov           = 90.0

mesh_path = '/mnt/c/Users/Owner/Downloads/jax-pulse-master/jax-pulse-master/data/meshes/obj/refined_plate.obj'

# =============================================================================
# load mesh
# =============================================================================
print("Loading mesh...")
pulse_model = Pulse(
    sigma=sigma, a=a, ref_dist=ref_dist,
    resolution=resolution, fov=fov,
    volumetric_flow_rate=1e-5
)
pulse_model.load_mesh(mesh_path)

face_v0      = pulse_model.face_v0
face_v1      = pulse_model.face_v1
face_v2      = pulse_model.face_v2
face_normals = pulse_model.face_normals
n_faces      = pulse_model.n_faces
areas        = pulse_model.areas

print(f"Mesh loaded. Faces: {n_faces}")
print(f"Mesh bounds: x [{float(face_v0[:,0].min()):.2f}, {float(face_v0[:,0].max()):.2f}]")
print(f"             y [{float(face_v0[:,1].min()):.2f}, {float(face_v0[:,1].max()):.2f}]")

# =============================================================================
# zigzag nozzle path
# =============================================================================
z_height = 1.5
x_range  = np.linspace(0.1, 2.9, 20)
y_range  = np.linspace(0.1, 2.9, 20)

poses = []
for i, y in enumerate(y_range):
    xs = x_range if i % 2 == 0 else x_range[::-1]
    for x in xs:
        pos = jnp.array([x, y, z_height])
        rot = jnp.array([1.0, 0.0, 0.0, 0.0])
        poses.append(Pose(position=pos, rotation=rot))

n_poses = len(poses)
print(f"Total nozzle poses: {n_poses}")

pose_positions = jnp.stack([p.position for p in poses])
pose_rotations = jnp.stack([p.rotation for p in poses])
steps_per_move = max(1, steps // n_poses)

# =============================================================================
# precompute h_local for each pose
# =============================================================================
print("Precomputing spray distributions...")

def compute_h_for_pose(pos, rot):
    weight = deposit(
        pos, rot, sigma, a, ref_dist, resolution,
        face_v0, face_v1, face_v2, face_normals, n_faces, fov
    )
    return h_ambient + weight * h_spray_scale

h_fields = jax.vmap(compute_h_for_pose)(pose_positions, pose_rotations)
h_fields = jnp.array(h_fields)
print("Done precomputing.")

# =============================================================================
# build face adjacency
# =============================================================================
print("Building face adjacency...")
mesh_pv  = pv.read(mesh_path).triangulate()
faces_np = np.array(mesh_pv.faces).reshape(-1, 4)[:, 1:]

edge_to_faces = defaultdict(list)
for fi, face in enumerate(faces_np):
    for j in range(3):
        edge = tuple(sorted([face[j], face[(j+1)%3]]))
        edge_to_faces[edge].append(fi)

max_neighbors = 3
neighbors = -1 * np.ones((n_faces, max_neighbors), dtype=np.int32)
for fi, face in enumerate(faces_np):
    nb_count = 0
    for j in range(3):
        edge = tuple(sorted([face[j], face[(j+1)%3]]))
        for fj in edge_to_faces[edge]:
            if fj != fi and nb_count < max_neighbors:
                neighbors[fi, nb_count] = fj
                nb_count += 1

neighbors_jax = jnp.array(neighbors)
print("Adjacency built.")

# =============================================================================
# JAX step function
# =============================================================================
T_init = jnp.full((n_faces,), T_initial)

@jax.jit
def step(carry, _):
    T, step_idx = carry

    pose_idx = jnp.minimum(step_idx // steps_per_move, n_poses - 1)
    h_field  = h_fields[pose_idx]

    def get_neighbor_temp(nb_idx):
        safe_idx = jnp.maximum(nb_idx, 0)
        return jnp.where(nb_idx >= 0, T[safe_idx], T)

    nb_temps  = jax.vmap(get_neighbor_temp)(neighbors_jax.T)
    valid     = (neighbors_jax.T >= 0).astype(jnp.float32)
    n_valid   = jnp.maximum(valid.sum(axis=0), 1.0)
    mean_nb   = (nb_temps * valid).sum(axis=0) / n_valid
    laplacian = mean_nb - T

    dTdt  = alpha * laplacian - (h_field / rho_c) * (T - T_ambient)
    T_new = T + dTdt * dt

    return (T_new, step_idx + 1), (T_new, pose_idx)

# =============================================================================
# run simulation
# =============================================================================
print("Running simulation...")
_, (T_history, pose_idx_history) = jax.lax.scan(
    step, (T_init, jnp.int32(0)), None, length=steps
)
T_history        = np.array(T_history)
pose_idx_history = np.array(pose_idx_history)
print("Done.")

print(f"\nFinal results:")
print(f"  Peak temp:    {T_history[-1].max():.1f} C")
print(f"  Min temp:     {T_history[-1].min():.1f} C")
print(f"  Temp spread:  {T_history[-1].max() - T_history[-1].min():.1f} C")
print(f"  Avg temp:     {T_history[-1].mean():.1f} C")
print(f"  deposit() max: {float(h_fields.max()):.2f}")

# =============================================================================
# animated visualization — looping
# =============================================================================
print("Animating...")
mesh_vis = pv.read(mesh_path).triangulate()
mesh_vis.cell_data['temperature'] = T_history[0]

plotter = pv.Plotter(off_screen=False)
plotter.set_background('black')
plotter.add_title('JAX-PULSE Spray Cooling on 3D Mesh')

actor = plotter.add_mesh(
    mesh_vis,
    scalars='temperature',
    cmap='inferno',
    show_edges=False,
    clim=[T_ambient, T_initial],
    scalar_bar_args={'title': 'Temperature (°C)'}
)

path_points = np.array([[p.position[0], p.position[1], p.position[2]]
                         for p in poses])
path_pv = pv.Spline(path_points, 1000)
plotter.add_mesh(path_pv, color='cyan', line_width=2, opacity=0.4)

nozzle_point = pv.PolyData(np.array([[poses[0].position[0],
                                       poses[0].position[1],
                                       poses[0].position[2]]],
                                     dtype=np.float32))
plotter.add_mesh(nozzle_point, color='cyan',
                 point_size=20, render_points_as_spheres=True)

plotter.show(auto_close=False, interactive_update=True)

skip       = max(1, steps // 300)
frame_list = list(range(0, steps, skip))

while True:
    for frame in frame_list:
        mesh_vis.cell_data['temperature'] = T_history[frame]
        actor.mapper.dataset.cell_data['temperature'] = T_history[frame]
        actor.mapper.dataset.Modified()

        pidx = int(pose_idx_history[frame])
        nozzle_point.points = np.array([[poses[pidx].position[0],
                                          poses[pidx].position[1],
                                          poses[pidx].position[2]]],
                                        dtype=np.float32)
        plotter.render()
        time.sleep(0.03)