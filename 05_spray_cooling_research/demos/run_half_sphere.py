"""
Robotic Spray Cooling on a Hemispherical Shell
================================================================

Research question:
    Can a single fixed-base UR5e adequately cool a hemispherical part
    via spray, using the same physics-informed framework as the flat plate?

Finding (spoiler for the poster):
    No. Ring-based Fibonacci-style waypoints on the upper dome are
    partially unreachable from a single robot base. The uncooled zones
    directly motivate turntable indexing, multi-robot coordination,
    or a mobile base as the next research step.

Physics (identical to flat plate):
    dT/dt = alpha * laplacian_approx(T) - (h_local / rho_c) * (T - T_ambient)

    Discretization: face-adjacency graph Laplacian on triangle mesh,
    geometry-weighted via shape function gradients (Josh's direction).
    Time integration: JAX lax.scan, jitted.

Spray coupling:
    JAX-PULSE deposit() computes per-face h-field for each robot pose,
    accounting for incidence angle -- so curved surfaces automatically
    get non-uniform spray intensity, unlike a flat plate.

Robot integration:
    UR5e loaded via URDF, ikpy for inverse kinematics with
    normal-following orientation (tool +Z aligned to local outward normal).
    Waypoints that IK cannot solve are logged as unreachable and skipped.
"""

# ================================================================
# Section 1 - Imports and config
# ================================================================
from collections import defaultdict
from ikpy.chain import Chain
from matplotlib.backends.backend_agg import FigureCanvasAgg
from relevant_PULSE_files.jax_kernels import Pose, deposit
from relevant_PULSE_files.jax_pulse import Pulse
from robot_descriptions.loaders.yourdfpy import load_robot_description
from spray_cooling.config import load_config
from spray_cooling.geometry.queries import nearest_surface_point_and_normal, resolve_mesh_path
from spray_cooling.geometry.surface_mesh import load_surface_mesh
from spray_cooling.physics.thermal_steps import make_flat_plate_step
from spray_cooling.planning.hemisphere import build_hemisphere_path
from spray_cooling.robotics.runtime import (
    world_to_base, set_initial_joint, q_to_cfg,
    get_tool0_transform_world, get_tool0_world,
    get_nozzle_tip_world, get_nozzle_direction_world,
)
from spray_cooling.robotics.trajectory import smooth_joint_path, unwrap_to_reference
from spray_cooling.spray.pulse_model import make_compute_h_for_pose
from spray_cooling.visualization.common import set_actor_matrix, set_spray_head_position

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import pyvista as pv
import time
import trimesh
import vtk
import xml.etree.ElementTree as ET

# Runtime directory for auto-generated files (rescaled meshes, expanded URDF)
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "artifacts", "generated")
os.makedirs(RUNTIME_DIR, exist_ok=True)

# Load config from configs/half_sphere.toml
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "half_sphere.toml")
cfg = load_config(CONFIG_PATH)

# Physics constants (carbon steel, same as flat plate)
T_initial     = cfg.thermal.initial_temperature_c
T_ambient     = cfg.thermal.ambient_temperature_c
k             = cfg.thermal.thermal_conductivity_w_mk
c             = cfg.thermal.specific_heat_j_kgk
rho           = cfg.thermal.density_kg_m3
rho_c         = rho * c
alpha         = k / rho_c
h_ambient     = cfg.thermal.ambient_h_w_m2k
h_spray_scale = cfg.thermal.spray_h_scale_w_m2k
SHELL_THICKNESS = cfg.thermal.thickness_m

# Simulation
dt_sim        = cfg.simulation.dt_s
t_end         = cfg.simulation.end_time_s
steps         = int(t_end / dt_sim)

# PULSE spray parameters
sigma         = cfg.spray.sigma
a             = cfg.spray.a
ref_dist      = cfg.spray.reference_distance_m
resolution    = cfg.spray.resolution
fov           = cfg.spray.field_of_view_deg

# Scene geometry
SPHERE_CENTER = np.array(cfg.geometry.plate_center_m, dtype=float)  # reused field name
SPHERE_RADIUS = cfg.geometry.plate_size_m / 2.0                     # half of "plate_size"
ROBOT_BASE    = np.array(cfg.robot.base_position_m, dtype=float)
NOZZLE_STANDOFF = cfg.spray.reference_distance_m  # standoff along outward normal
NOZZLE_LENGTH = cfg.robot.visual_nozzle_length_m

mesh_path = os.fspath(cfg.geometry.mesh_path)

print("=" * 70)
print("Hemisphere spray-cooling demo -- single UR5e, fixed base, no turntable.")
print("Purpose: expose reachability limits of a single fixed-base robot")
print(f"         on a curved geometry (dome radius = {SPHERE_RADIUS*1000:.1f} mm).")
print("=" * 70)

# ================================================================
# Section 2 - Load hemisphere mesh, enforce outward normals
# ================================================================
print("\nLoading hemisphere mesh...")

tmp_mesh_path = os.path.join(RUNTIME_DIR, "_rescaled_halfsphere.obj")

surface_mesh = load_surface_mesh(
    mesh_path,
    target_x_extent_m=2.0 * SPHERE_RADIUS,
    center_m=SPHERE_CENTER,
    runtime_path=tmp_mesh_path,
)

raw_mesh = surface_mesh.polydata

# Enforce outward normals on the dome.
# A hemisphere has a well-defined center; any face whose normal has
# negative component along (face_center - sphere_center) is pointing
# inward -- flip it. This is safer than trusting OBJ winding.
face_centers_pre = surface_mesh.face_centers
radial_vectors = face_centers_pre - SPHERE_CENTER
radial_dot_normal = np.einsum("ij,ij->i", surface_mesh.face_normals, radial_vectors)

flip_mask = radial_dot_normal < 0.0
face_normals_corrected = surface_mesh.face_normals.copy()
face_normals_corrected[flip_mask] *= -1.0
n_flipped = int(np.sum(flip_mask))
print(f"  Outward-normal enforcement: flipped {n_flipped} of {len(flip_mask)} face normals.")

# Bind PULSE to this mesh
pulse_model = Pulse(
    sigma=sigma, a=a, ref_dist=ref_dist,
    resolution=resolution, fov=fov,
    volumetric_flow_rate=cfg.spray.volumetric_flow_rate_m3_s,
)
pulse_model.load_mesh(mesh_path)

face_v0      = pulse_model.face_v0
face_v1      = pulse_model.face_v1
face_v2      = pulse_model.face_v2
face_normals = jnp.asarray(face_normals_corrected, dtype=jnp.float32)
n_faces      = pulse_model.n_faces

print(f"  Mesh loaded. Faces: {n_faces}")
print(f"  Sphere bounds: x [{raw_mesh.bounds[0]:.3f}, {raw_mesh.bounds[1]:.3f}]  "
      f"y [{raw_mesh.bounds[2]:.3f}, {raw_mesh.bounds[3]:.3f}]  "
      f"z [{raw_mesh.bounds[4]:.3f}, {raw_mesh.bounds[5]:.3f}]")

# Numpy versions for geometric queries
face_v0_np      = np.array(face_v0)
face_v1_np      = np.array(face_v1)
face_v2_np      = np.array(face_v2)
face_centers_np = (face_v0_np + face_v1_np + face_v2_np) / 3.0
face_normals_np = np.array(face_normals)

# ================================================================
# Section 3 - Generate ring-based waypoints on the dome
# ================================================================
print("\nGenerating ring-based dome waypoints...")

# Path config
ring_count      = cfg.path.number_of_rings if hasattr(cfg.path, "number_of_rings") else cfg.path.number_of_rows
points_per_ring = cfg.path.points_per_row
polar_min_deg   = 5.0    # start slightly below apex to avoid singularity
polar_max_deg   = 80.0   # stop before the rim (dome/base transition)

hemisphere_path = build_hemisphere_path(
    sphere_center=SPHERE_CENTER,
    sphere_radius=SPHERE_RADIUS,
    ring_count=ring_count,
    points_per_ring=points_per_ring,
    polar_min_deg=polar_min_deg,
    polar_max_deg=polar_max_deg,
    standoff=NOZZLE_STANDOFF,
)

# Extract for downstream use
surface_points   = hemisphere_path.surface_points    # (N, 3) points on dome surface
surface_normals  = hemisphere_path.surface_normals   # (N, 3) outward normals at each point
tool_positions   = hemisphere_path.tool_positions    # (N, 3) nozzle tip positions (surface + standoff * normal)
pulse_quats      = hemisphere_path.pulse_quaternions_xyzw  # (N, 4) PULSE poses

n_waypoints_total = len(surface_points)
print(f"  Total candidate waypoints on dome: {n_waypoints_total}")
print(f"  Ring layout: {ring_count} rings x up to {points_per_ring} pts/ring")
print(f"  Polar range: {polar_min_deg} deg to {polar_max_deg} deg from apex")
print(f"  Standoff:    {NOZZLE_STANDOFF*1000:.1f} mm along outward normal")

# ================================================================
# Section 4 - Robot setup (URDF, ikpy chain, initial pose)
# ================================================================
print("\nLoading UR5e URDF...")

# Robot base transform in world
robot_base_T = np.eye(4)
robot_base_T[:3, 3] = ROBOT_BASE

# Load UR5e URDF via robot_descriptions; save expanded XML for ikpy
urdf = load_robot_description(cfg.robot.description)
urdf_expanded_path = os.path.join(RUNTIME_DIR, "_ur5e_expanded.urdf")
urdf.write_xml_file(urdf_expanded_path)
print(f"  UR5e URDF written to: {urdf_expanded_path}")

urdf = yourdfpy.URDF.load(urdf_expanded_path) if False else None  # visualization uses raw
import yourdfpy
urdf = yourdfpy.URDF.load(urdf_expanded_path)

# Build ikpy chain
chain = Chain.from_urdf_file(
    urdf_expanded_path,
    base_elements=["base_link"],
    active_links_mask=[
        False, False,             # OriginLink, base_link inertia
        True, True, True, True, True, True,  # 6 UR5e joints
        False,                    # fixed tool0/flange
    ],
)
print(f"  ikpy chain built with {len(chain.links)} links.")

# Joint names (ikpy order)
UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Initial joint state -- arm extended forward, slightly down
_last_joint_state = np.zeros(len(chain.links))
set_initial_joint(_last_joint_state, chain, "shoulder_pan_joint", 0.0)
set_initial_joint(_last_joint_state, chain, "shoulder_lift_joint", -1.5708)
set_initial_joint(_last_joint_state, chain, "elbow_joint", 1.5708)
set_initial_joint(_last_joint_state, chain, "wrist_1_joint", -1.5708)
set_initial_joint(_last_joint_state, chain, "wrist_2_joint", -1.5708)
set_initial_joint(_last_joint_state, chain, "wrist_3_joint", 0.0)


# ================================================================
# Section 5 - IK with graceful failure and reachability logging
# ================================================================
# For each dome waypoint, attempt IK with normal-following orientation.
# Log successes and failures. The reachability shadow is the research finding.
#
# Failure modes we handle:
#   1. IK exception (ikpy raised an error)
#   2. IK converged but end-effector is too far from target (unreachable)
#   3. IK converged but wrist configuration passes through the sphere
#      (not detected here; visual inspection during animation)
#
# Tolerance: 30mm. Waypoints failing beyond this are flagged unreachable.
IK_REACH_TOLERANCE = 0.030  # meters

def solve_ik_with_failure(target_world, target_normal_world, last_q):
    """
    Attempt IK for a single waypoint with normal-following orientation.
    Returns (q, status) where status is one of:
        'ok'           -- IK converged within tolerance
        'unreachable'  -- IK converged but tool0 too far from target
        'exception'    -- ikpy raised
    """
    target_world = np.asarray(target_world, dtype=float)
    target_normal_world = np.asarray(target_normal_world, dtype=float)
    target_base = world_to_base(target_world)
    try:
        q_raw = chain.inverse_kinematics(
            target_position=target_base,
            target_orientation=target_normal_world,
            orientation_mode="Z",
            initial_position=last_q,
        )
    except Exception:
        try:
            q_raw = chain.inverse_kinematics(
                target_position=target_base,
                initial_position=last_q,
            )
        except Exception:
            return last_q, "exception"

    # Verify reach: forward kinematics gives actual tool0 position
    fk_matrix = chain.forward_kinematics(q_raw)
    tool0_base = fk_matrix[:3, 3]
    err = np.linalg.norm(tool0_base - target_base)
    if err > IK_REACH_TOLERANCE:
        return last_q, "unreachable"

    return unwrap_to_reference(q_raw, last_q), "ok"


print("\nSolving IK for all dome waypoints...")
print(f"  Tolerance: {IK_REACH_TOLERANCE*1000:.0f}mm from target -> flagged unreachable")

reachable_indices = []
unreachable_indices = []
q_solutions = []  # only for reachable waypoints
reachable_surface_points = []
reachable_surface_normals = []
reachable_tool_positions = []
reachable_pulse_quats = []

current_q = _last_joint_state.copy()
for i in range(n_waypoints_total):
    tool_pos = tool_positions[i]
    surf_normal = surface_normals[i]
    q, status = solve_ik_with_failure(tool_pos, surf_normal, current_q)

    if status == "ok":
        reachable_indices.append(i)
        q_solutions.append(q.copy())
        reachable_surface_points.append(surface_points[i])
        reachable_surface_normals.append(surface_normals[i])
        reachable_tool_positions.append(tool_positions[i])
        reachable_pulse_quats.append(pulse_quats[i])
        current_q = q  # warm-start next solve
    else:
        unreachable_indices.append(i)

n_reachable = len(reachable_indices)
n_unreachable = len(unreachable_indices)
reach_percent = 100.0 * n_reachable / n_waypoints_total

print(f"  Reachable:   {n_reachable}/{n_waypoints_total} ({reach_percent:.1f}%)")
print(f"  Unreachable: {n_unreachable} waypoints")

# Convert to arrays for downstream sim
reachable_surface_points  = np.array(reachable_surface_points)
reachable_surface_normals = np.array(reachable_surface_normals)
reachable_tool_positions  = np.array(reachable_tool_positions)
reachable_pulse_quats     = np.array(reachable_pulse_quats)
q_solutions               = np.array(q_solutions)

# ================================================================
# Section 6 - Precompute h-fields for reachable waypoints only
# ================================================================
print("\nPrecomputing spray h-fields for reachable waypoints...")

compute_h_for_pose = make_compute_h_for_pose(
    a=a,
    face_normals=face_normals,
    face_v0=face_v0,
    face_v1=face_v1,
    face_v2=face_v2,
    fov=fov,
    h_ambient=h_ambient,
    h_spray_scale=h_spray_scale,
    n_faces=n_faces,
    ref_dist=ref_dist,
    resolution=resolution,
    sigma=sigma,
)

h_fields = []
for i in range(n_reachable):
    pos = jnp.asarray(reachable_tool_positions[i], dtype=jnp.float32)
    rot = jnp.asarray(reachable_pulse_quats[i], dtype=jnp.float32)
    h_field = compute_h_for_pose(pos, rot)
    h_fields.append(np.asarray(h_field))

h_fields = np.array(h_fields)  # (n_reachable, n_faces)
print(f"  Precomputed {len(h_fields)} h-fields, each of shape ({n_faces},)")
print(f"  h_field per-pose peak: {h_fields.max():.1f} W/m^2K")
nonzero_mask = h_fields > h_ambient  # exclude the ambient floor
if nonzero_mask.any():
    print(f"  h_field spray region mean: {h_fields[nonzero_mask].mean():.1f} W/m^2K")

# ================================================================
# Section 7 - Build geometry-weighted thermal operator
# ================================================================
# Same operator as flat plate: face-adjacency graph Laplacian with
# per-edge conductance weights and per-face thermal mass. Josh's
# direction -- geometry-aware, works on any triangle mesh.
print("\nBuilding geometry-weighted thermal operator...")

# Face adjacency via shared edges
edge_to_faces = defaultdict(list)
faces_arr = np.array(surface_mesh.faces)  # (n_faces, 3) vertex indices
verts_arr = np.array(surface_mesh.points)

for face_idx in range(n_faces):
    v_ids = faces_arr[face_idx]
    for a, b in [(v_ids[0], v_ids[1]),
                 (v_ids[1], v_ids[2]),
                 (v_ids[2], v_ids[0])]:
        key = (min(int(a), int(b)), max(int(a), int(b)))
        edge_to_faces[key].append(face_idx)

# Build sparse neighbor list and conductance
max_neighbors = 3
neighbors = -1 * np.ones((n_faces, max_neighbors), dtype=np.int32)
conductance = np.zeros((n_faces, max_neighbors), dtype=np.float32)
nb_count = np.zeros(n_faces, dtype=np.int32)

face_areas_np = np.array(surface_mesh.face_areas)
for (va, vb), flist in edge_to_faces.items():
    if len(flist) != 2:
        continue
    f1, f2 = flist
    edge_len = np.linalg.norm(verts_arr[va] - verts_arr[vb])
    # thermal conductance per edge (W/K per unit temperature difference)
    G = k * SHELL_THICKNESS * edge_len / max(
        np.linalg.norm(
            (face_v0_np[f1] + face_v1_np[f1] + face_v2_np[f1]) / 3.0
            - (face_v0_np[f2] + face_v1_np[f2] + face_v2_np[f2]) / 3.0
        ), 1e-9
    )
    for src, dst in [(f1, f2), (f2, f1)]:
        idx = nb_count[src]
        if idx < max_neighbors:
            neighbors[src, idx] = dst
            conductance[src, idx] = G
            nb_count[src] += 1

neighbors_jax    = jnp.array(neighbors)
conductance_jax  = jnp.array(conductance)
face_area_jax    = jnp.array(face_areas_np)
thermal_mass_jax = rho_c * SHELL_THICKNESS * face_area_jax

positive_G = conductance[conductance > 0]
print("  Geometry-weighted thermal operator built.")
print(f"  face area: min={face_areas_np.min():.3e}, max={face_areas_np.max():.3e}")
print(f"  conductance: min={positive_G.min():.3e}, max={positive_G.max():.3e}")


# ================================================================
# Section 8 - JAX heat step + lax.scan rollout
# ================================================================
# Reachable-only h-fields drive the sim. Steps advance through reachable
# waypoints; unreachable ones simply don't exist in this array.
print("\nRunning simulation...")

T_init = jnp.full((n_faces,), T_initial)
n_reachable_moves = n_reachable
steps_per_move = max(1, steps // n_reachable_moves)

h_fields_jax = jnp.array(h_fields, dtype=jnp.float32)  # (n_reachable, n_faces)

step = jax.jit(
    make_flat_plate_step(
        h_fields=h_fields_jax,
        steps_per_move=steps_per_move,
        n_waypoints=n_reachable_moves,
        neighbors=neighbors_jax,
        conductance=conductance_jax,
        thermal_mass=thermal_mass_jax,
        volumetric_heat_capacity=rho_c,
        thickness_m=SHELL_THICKNESS,
        ambient_temperature_c=T_ambient,
        dt_s=dt_sim,
    )
)

_, (T_history, pose_idx_history) = jax.lax.scan(
    step, (T_init, jnp.int32(0)), None, length=steps
)
T_history        = np.array(T_history)
pose_idx_history = np.array(pose_idx_history)
print("Done.")

# ================================================================
# Section 13 (early) - Results with reachability shadow stats
# ================================================================
# Separate reachable-face stats from unreachable-face stats to show the
# thermal signature of the reachability shadow directly.
# A face is "meaningfully sprayed" only if some pose delivered spray-scale
# convection, not just the ambient floor. Threshold at 2× ambient.
was_sprayed_per_face = (h_fields.max(axis=0) > 2.0 * h_ambient)
n_sprayed_faces      = int(was_sprayed_per_face.sum())
n_shadow_faces       = n_faces - n_sprayed_faces

T_final = T_history[-1]
T_sprayed = T_final[was_sprayed_per_face]
T_shadow  = T_final[~was_sprayed_per_face]

print(f"\n{'='*70}")
print(f"FINAL RESULTS  (hemisphere, single UR5e, no turntable)")
print(f"{'='*70}")
print(f"Waypoint reachability:")
print(f"  Reachable poses:  {n_reachable}/{n_waypoints_total}  ({reach_percent:.1f}%)")
print(f"Face-level coverage:")
print(f"  Sprayed faces:    {n_sprayed_faces}/{n_faces}  ({100.0*n_sprayed_faces/n_faces:.1f}%)")
print(f"  Shadow faces:     {n_shadow_faces}/{n_faces}  ({100.0*n_shadow_faces/n_faces:.1f}%)")
print(f"Final temperature (all faces):")
print(f"  Peak:    {T_final.max():.1f} C")
print(f"  Min:     {T_final.min():.1f} C")
print(f"  Spread:  {T_final.max() - T_final.min():.1f} C")
print(f"  Avg:     {T_final.mean():.1f} C")
if n_sprayed_faces > 0:
    print(f"Sprayed-region final temperature:")
    print(f"  Avg:     {T_sprayed.mean():.1f} C")
    print(f"  Peak:    {T_sprayed.max():.1f} C")
if n_shadow_faces > 0:
    print(f"Shadow-region final temperature (the research finding):")
    print(f"  Avg:     {T_shadow.mean():.1f} C")
    print(f"  Peak:    {T_shadow.max():.1f} C")
    print(f"  Delta vs sprayed:  {T_shadow.mean() - T_sprayed.mean():.1f} C hotter")
print(f"{'='*70}")