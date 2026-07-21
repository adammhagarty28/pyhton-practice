"""
POC: UR5e Robot Spray Cooling with JAX-PULSE on 3D Mesh — PyVista single window.

UR5e arm follows zigzag path above a 0.6m x 0.6m plate. tool0 world position
feeds into deposit() which computes spray distribution. Heat equation evolves
temperature per triangle face. Robot loaded from URDF, IK via ikpy.

Physics:
    dT/dt = alpha * laplacian_approx(T) - (h_local / rho_c) * (T - T_ambient)
    h_local = h_ambient + deposit(tool0_pose) * h_spray_scale
"""

# IMPORTANT PYVISTA SHUTDOWN NOTE:
# This application uses a manually updated PyVista/VTK animation loop.
# On this workstation, closing the native PyVista window with its X button can
# leave the VTK window in a corrupted update state and cause severe flashing.
#
# ALWAYS stop this program from the terminal with Ctrl+C.
# DO NOT close the PyVista window with the X button.
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
#
# CRITICAL: enable JAX float64 BEFORE any jax operation is executed.
# Radiation T^4 at 1173K ~= 6.5e11 needs float64 precision.
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)
import os
import time
from collections import defaultdict
import xml.etree.ElementTree as ET
import trimesh

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import vtk

from robot_descriptions.loaders.yourdfpy import load_robot_description
from ikpy.chain import Chain

from relevant_PULSE_files.jax_kernels import Pose, deposit
from relevant_PULSE_files.jax_pulse import Pulse
from spray_cooling.config import load_config
from spray_cooling.geometry.surface_mesh import load_surface_mesh
from spray_cooling.robotics.trajectory import smooth_joint_path, unwrap_to_reference
from spray_cooling.spray.pulse_model import make_compute_h_for_pose
from spray_cooling.geometry.queries import nearest_surface_point_and_normal, resolve_mesh_path
from spray_cooling.robotics.runtime import world_to_base, set_initial_joint, q_to_cfg, get_tool0_transform_world, get_tool0_world, get_nozzle_tip_world, get_nozzle_direction_world
from spray_cooling.visualization.common import set_actor_matrix, set_spray_head_position
from spray_cooling.physics.thermal_steps import make_flat_plate_step


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "flat_plate.toml")
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "artifacts", "generated")
os.makedirs(RUNTIME_DIR, exist_ok=True)

cfg = load_config(CONFIG_PATH, project_root=PROJECT_ROOT)


# Physics parameters (POC 3 volume-mesh + IMEX + boiling curve)
T_initial   = cfg.thermal.initial_temperature_c        # [C]
T_ambient   = cfg.thermal.ambient_temperature_c        # [C]
rho         = cfg.thermal.density_kg_m3                # [kg/m^3]
h_ambient   = cfg.thermal.ambient_h_w_m2k              # [W/(m^2 K)]
PLATE_THICKNESS   = cfg.thermal.thickness_m            # [m]
LAYER_THICKS_NP   = np.array(cfg.thermal.layer_thicknesses_m)   # [m], top->bottom
N_LAYERS          = cfg.thermal.n_layers               # [-]
# Legacy fallback constants for pre-JAX printouts and any legacy calls
k           = cfg.thermal.thermal_conductivity_w_mk
c           = cfg.thermal.specific_heat_j_kgk
rho_c       = rho * c
alpha       = k / rho_c
# Boiling curve + radiation
STEFAN_BOLTZMANN = cfg.radiation.stefan_boltzmann_w_m2k4   # [W/(m^2 K^4)]

dt_sim        = cfg.simulation.dt_s
t_end         = cfg.simulation.end_time_s
steps         = int(t_end / dt_sim)

#PULSE parameters — sigma stays small since plate is smaller
sigma         = cfg.spray.sigma
a             = cfg.spray.a
ref_dist      = cfg.spray.reference_distance_m
resolution    = cfg.spray.resolution
fov           = cfg.spray.field_of_view_deg

#scene geometry — UR5e reach is 850mm
PLATE_SIZE   = cfg.geometry.plate_size_m
PLATE_CENTER = np.array(cfg.geometry.plate_center_m, dtype=float)
PLATE_Z      = float(PLATE_CENTER[2])
ROBOT_BASE   = np.array(cfg.robot.base_position_m, dtype=float)
NOZZLE_Z     = cfg.path.nozzle_world_z_m
NOZZLE_LENGTH = cfg.robot.visual_nozzle_length_m

mesh_path = os.fspath(cfg.geometry.mesh_path)


#load mesh + JAX-PULSE, and rescale plate to PLATE_SIZE
print("Loading mesh...")

# Load, uniformly scale, recenter, and preprocess the triangular surface.
# The same geometry representation will later be used for half_sphere.obj.
tmp_mesh_path = os.path.join(RUNTIME_DIR, '_rescaled_plate.obj')

surface_mesh = load_surface_mesh(
    mesh_path,
    target_x_extent_m=PLATE_SIZE,
    center_m=PLATE_CENTER,
    runtime_path=tmp_mesh_path,
)

raw_mesh = surface_mesh.polydata

pulse_model = Pulse(
    sigma=sigma, a=a, ref_dist=ref_dist,
    resolution=resolution, fov=fov,
    volumetric_flow_rate=cfg.spray.volumetric_flow_rate_m3_s
)
pulse_model.load_mesh(tmp_mesh_path)

face_v0      = pulse_model.face_v0
face_v1      = pulse_model.face_v1
face_v2      = pulse_model.face_v2
face_normals = pulse_model.face_normals
n_faces      = pulse_model.n_faces
print(f"Mesh loaded. Faces: {n_faces}")
print(f"Plate bounds: x [{raw_mesh.bounds[0]:.3f}, {raw_mesh.bounds[1]:.3f}]  "
      f"y [{raw_mesh.bounds[2]:.3f}, {raw_mesh.bounds[3]:.3f}]")


#waypoints — zigzag over plate at NOZZLE_Z
half = PLATE_SIZE / 2.0
margin = cfg.path.edge_margin_m
x_min = PLATE_CENTER[0] - half + margin
x_max = PLATE_CENTER[0] + half - margin
y_min = PLATE_CENTER[1] - half + margin
y_max = PLATE_CENTER[1] + half - margin

x_range = np.linspace(x_min, x_max, cfg.path.points_per_row)
y_range = np.linspace(y_min, y_max, cfg.path.number_of_rows)

waypoints = []
for i, y in enumerate(y_range):
    xs = x_range if i % 2 == 0 else x_range[::-1]
    for x in xs:
        waypoints.append(np.array([x, y, NOZZLE_Z]))

n_waypoints    = len(waypoints)
steps_per_move = max(1, steps // n_waypoints)
print(f"Total waypoints: {n_waypoints}")


#precompute normalized PULSE spray masks (per-pose, per-surface-triangle)
print("Precomputing PULSE spray masks...")
rot = jnp.array([1.0, 0.0, 0.0, 0.0])   # nozzle points -Z
pose_positions = jnp.array(np.array(waypoints))
pose_rotations = jnp.tile(rot, (n_waypoints, 1))

def _compute_weight_for_pose(pos, rot_):
    return deposit(
        pos, rot_, sigma, a, ref_dist, resolution,
        face_v0, face_v1, face_v2, face_normals, n_faces, fov,
    )
weights_raw   = jax.vmap(_compute_weight_for_pose)(pose_positions, pose_rotations)
peak_weight   = jnp.max(weights_raw)
if float(peak_weight) < 1e-12:
    raise RuntimeError("PULSE returned zero weights - check nozzle Z, sigma, ref_dist, fov")
spray_masks_surf = jnp.array(weights_raw / peak_weight)   # [n_poses, n_faces] in [0,1]
print(f"  PULSE peak weight (raw): {float(peak_weight):.4f} -> normalized to 1.0")
print(f"  active poses (any spray hit): {int(jnp.sum(jnp.max(spray_masks_surf, axis=1) > 0.01))} / {n_waypoints}")



# Volume mesh construction (POC 3): prism extrusion of surface triangles
print("Building 3D volume mesh (prism extrusion)...")

mesh_pv       = surface_mesh.polydata
points_np     = surface_mesh.points
faces_np      = surface_mesh.faces
n_surface     = pulse_model.n_faces
fv_face_centers_np = surface_mesh.face_centers
fv_face_area_np    = surface_mesh.face_areas
areas_surf_np      = np.array(fv_face_area_np)

n_cells = n_surface * N_LAYERS
print(f"  n_surface={n_surface}, N_LAYERS={N_LAYERS}, n_cells={n_cells}")

# Per-cell volumes and layer index
cell_volume_np = np.zeros(n_cells, dtype=np.float64)
cell_layer_np  = np.zeros(n_cells, dtype=np.int32)
for layer in range(N_LAYERS):
    cell_volume_np[layer*n_surface:(layer+1)*n_surface] = areas_surf_np * LAYER_THICKS_NP[layer]
    cell_layer_np[layer*n_surface:(layer+1)*n_surface]  = layer

is_top_np    = (cell_layer_np == 0).astype(np.float64)
is_bottom_np = (cell_layer_np == N_LAYERS - 1).astype(np.float64)
face_area_top_np    = np.where(cell_layer_np == 0,            np.tile(areas_surf_np, N_LAYERS), 0.0)
face_area_bottom_np = np.where(cell_layer_np == N_LAYERS - 1, np.tile(areas_surf_np, N_LAYERS), 0.0)

# In-layer (surface) adjacency + edge lengths (needed for lateral connectivity)
edge_to_faces = defaultdict(list)
edge_length   = {}
for fi, face in enumerate(faces_np):
    for j in range(3):
        v0, v1 = face[j], face[(j + 1) % 3]
        edge = tuple(sorted([v0, v1]))
        edge_to_faces[edge].append(fi)
        if edge not in edge_length:
            edge_length[edge] = float(np.linalg.norm(points_np[v0] - points_np[v1]))

MAX_LATERAL = 3
lateral_nb_surf  = -1 * np.ones((n_surface, MAX_LATERAL), dtype=np.int32)
lateral_edge_len = np.zeros((n_surface, MAX_LATERAL), dtype=np.float64)
for fi, face in enumerate(faces_np):
    slot = 0
    for j in range(3):
        edge = tuple(sorted([face[j], face[(j + 1) % 3]]))
        for fj in edge_to_faces[edge]:
            if fj != fi and slot < MAX_LATERAL:
                lateral_nb_surf[fi, slot]  = fj
                lateral_edge_len[fi, slot] = edge_length[edge]
                slot += 1

# Side-face area per cell: exposed (boundary) edge total length * layer thickness
face_area_side_np = np.zeros(n_cells, dtype=np.float64)
for fi in range(n_surface):
    exposed_len = 0.0
    for j in range(3):
        edge = tuple(sorted([faces_np[fi, j], faces_np[fi, (j + 1) % 3]]))
        if len(edge_to_faces[edge]) == 1:
            exposed_len += edge_length[edge]
    for layer in range(N_LAYERS):
        face_area_side_np[layer*n_surface + fi] = exposed_len * LAYER_THICKS_NP[layer]

# Full 3D adjacency + G_ij = A_ij / (d_ij V_i)  [1/m^2]
MAX_NB_3D = 5   # 3 lateral + 2 vertical
neighbors_3d_np = -1 * np.ones((n_cells, MAX_NB_3D), dtype=np.int32)
G_matrix_np     = np.zeros((n_cells, MAX_NB_3D), dtype=np.float64)

for layer in range(N_LAYERS):
    L_i = LAYER_THICKS_NP[layer]
    for si in range(n_surface):
        ci  = layer * n_surface + si
        V_i = areas_surf_np[si] * L_i
        slot = 0
        # Lateral (in-layer)
        for kk in range(MAX_LATERAL):
            nb_s = lateral_nb_surf[si, kk]
            if nb_s >= 0:
                edge_len = lateral_edge_len[si, kk]
                A_ij = edge_len * L_i
                d_ij = float(np.linalg.norm(fv_face_centers_np[si] - fv_face_centers_np[nb_s]))
                d_ij = max(d_ij, 1e-9)
                neighbors_3d_np[ci, slot] = layer * n_surface + nb_s
                G_matrix_np[ci, slot]     = A_ij / (d_ij * V_i)
                slot += 1
        # Up neighbor
        if layer > 0:
            L_up = LAYER_THICKS_NP[layer - 1]
            A_ij = areas_surf_np[si]
            d_ij = 0.5 * (L_i + L_up)
            neighbors_3d_np[ci, slot] = (layer - 1) * n_surface + si
            G_matrix_np[ci, slot]     = A_ij / (d_ij * V_i)
            slot += 1
        # Down neighbor
        if layer < N_LAYERS - 1:
            L_dn = LAYER_THICKS_NP[layer + 1]
            A_ij = areas_surf_np[si]
            d_ij = 0.5 * (L_i + L_dn)
            neighbors_3d_np[ci, slot] = (layer + 1) * n_surface + si
            G_matrix_np[ci, slot]     = A_ij / (d_ij * V_i)
            slot += 1

positive_G = G_matrix_np[G_matrix_np > 0]
print(f"  volume adjacency built. G stats: min={positive_G.min():.3e}, max={positive_G.max():.3e} [1/m^2]")

# JAX arrays for step function
neighbors_3d_jax  = jnp.array(neighbors_3d_np)
G_matrix_jax      = jnp.array(G_matrix_np)
cell_volume_jax   = jnp.array(cell_volume_np)
face_area_top_jax    = jnp.array(face_area_top_np)
face_area_bottom_jax = jnp.array(face_area_bottom_np)
face_area_side_jax   = jnp.array(face_area_side_np)
is_top_mask_jax      = jnp.array(is_top_np)
surface_areas_jax    = jnp.array(areas_surf_np)


# JAX step + rollout (POC 3 IMEX BE with volume mesh + boiling curve + radiation)
T_init_K = jnp.full((n_cells,), T_initial + 273.15)

steps = int(cfg.simulation.end_time_s / cfg.simulation.dt_s)
steps_per_move = max(1, steps // n_waypoints)

step = jax.jit(make_flat_plate_step(
    spray_masks_surf=spray_masks_surf,
    steps_per_move=steps_per_move,
    n_waypoints=n_waypoints,
    n_surface=n_surface,
    n_cells=n_cells,
    neighbors_3d=neighbors_3d_jax,
    G_matrix=G_matrix_jax,
    cell_volume=cell_volume_jax,
    face_area_top=face_area_top_jax,
    face_area_bottom=face_area_bottom_jax,
    face_area_side=face_area_side_jax,
    is_top_mask=is_top_mask_jax,
    surface_areas=surface_areas_jax,
    temp_table_k=cfg.thermal.temp_table_k,
    k_table_w_mk=cfg.thermal.k_table_w_mk,
    cp_table_j_kgk=cfg.thermal.cp_table_j_kgk,
    emissivity_table=cfg.thermal.emissivity_table,
    density_kg_m3=rho,
    stefan_boltzmann=STEFAN_BOLTZMANN,
    h_ambient_w_m2k=h_ambient,
    boiling_temp_table_c=cfg.spray.boiling_temp_table_c,
    boiling_h_table_w_m2k=cfg.spray.boiling_h_table_w_m2k,
    ambient_temperature_c=T_ambient,
    dt_s=cfg.simulation.dt_s,
    delta_off_fraction=cfg.control.delta_off_fraction,
    delta_on_fraction=cfg.control.delta_on_fraction,
    spray_ramp_tau_s=cfg.control.spray_ramp_tau_s,
))

print("Running IMEX backward-Euler simulation on volume mesh...")
init_carry = (T_init_K, jnp.int32(0), jnp.float64(1.0), jnp.float64(1.0), jnp.float64(1.0))
_, (T_history_C_full, pose_idx_history, spray_history, T_top_history_C, T_bot_history_C) = \
    jax.lax.scan(step, init_carry, None, length=steps)
T_history_C_full  = np.array(T_history_C_full)          # [steps, n_cells] full 3D history
pose_idx_history  = np.array(pose_idx_history)
spray_history_np  = np.array(spray_history)
T_top_history_C   = np.array(T_top_history_C)           # [steps, n_surface]
T_bot_history_C   = np.array(T_bot_history_C)           # [steps, n_surface]
# For click-history compatibility, expose top-layer as the default 'T_history'
T_history = T_top_history_C
print("Done.")

# Volume-weighted plate average per step
vol_np_arr  = cell_volume_np
plate_avg_C = (T_history_C_full * vol_np_arr[None, :]).sum(axis=1) / vol_np_arr.sum()
final_top   = T_top_history_C[-1]
final_bot   = T_bot_history_C[-1]
peak_grad   = np.max(T_bot_history_C.mean(axis=1) - T_top_history_C.mean(axis=1))
peak_grad_t = float(np.argmax(T_bot_history_C.mean(axis=1) - T_top_history_C.mean(axis=1))) * cfg.simulation.dt_s
n_on  = int(np.sum(spray_history_np > 0.5))
n_off = int(np.sum(spray_history_np <= 0.5))
transitions = int(np.sum(np.abs(np.diff((spray_history_np > 0.5).astype(np.int32)))))

print("\n" + "="*70)
print("FINAL RESULTS (POC 3 volume mesh + IMEX + boiling curve)")
print("="*70)

# Build volumetric UnstructuredGrid for 3D visualization (POC 3)
print("Building 3D volume mesh for visualization...")
# Extrude surface points downward through N_LAYERS layers.
z_offsets = np.zeros(N_LAYERS + 1, dtype=np.float64)
for i in range(N_LAYERS):
    z_offsets[i + 1] = z_offsets[i] - LAYER_THICKS_NP[i]

n_verts_surf = len(points_np)
verts_3d = np.zeros((n_verts_surf * (N_LAYERS + 1), 3), dtype=np.float64)
for lvl in range(N_LAYERS + 1):
    verts_3d[lvl * n_verts_surf:(lvl + 1) * n_verts_surf, 0:2] = points_np[:, 0:2]
    verts_3d[lvl * n_verts_surf:(lvl + 1) * n_verts_surf, 2]   = points_np[:, 2] + z_offsets[lvl]

# VTK_WEDGE (=13) cells: 6 vertices per prism.
cells_vtk = []
for layer in range(N_LAYERS):
    for si in range(n_surface):
        top_v = faces_np[si] + layer * n_verts_surf
        bot_v = faces_np[si] + (layer + 1) * n_verts_surf
        cells_vtk.extend([6, top_v[0], top_v[1], top_v[2], bot_v[0], bot_v[1], bot_v[2]])
cells_vtk = np.array(cells_vtk, dtype=np.int64)
cell_types = np.full(n_cells, 13, dtype=np.uint8)
vol_grid_pv = pv.UnstructuredGrid(cells_vtk, cell_types, verts_3d)
vol_grid_pv.cell_data['temperature'] = T_history_C_full[0]

# Compute per-cell centroids for click-anywhere picking (top, sides, bottom)
cell_centroids_np = np.zeros((n_cells, 3), dtype=np.float64)
for layer in range(N_LAYERS):
    z_top    = float(np.sum(LAYER_THICKS_NP[:layer]))
    z_bottom = float(np.sum(LAYER_THICKS_NP[:layer + 1]))
    z_mid    = -0.5 * (z_top + z_bottom)   # negative because we extrude downward
    for si in range(n_surface):
        ci = layer * n_surface + si
        cell_centroids_np[ci, 0:2] = fv_face_centers_np[si, 0:2]
        cell_centroids_np[ci, 2]   = points_np[faces_np[si, 0], 2] + z_mid
print(f"  volumetric grid built: {n_cells} wedge cells")

print(f"  Initial avg temp:      {T_initial:.1f} C")
print(f"  Final avg temp:        {plate_avg_C[-1]:.1f} C   (delta {T_initial - plate_avg_C[-1]:.1f} C)")
print(f"  Final top peak:        {final_top.max():.1f} C")
print(f"  Final top min:         {final_top.min():.1f} C")
print(f"  Final top spread:      {final_top.max() - final_top.min():.1f} C")
print(f"  Top layer  final:  avg {final_top.mean():.1f} C   min {final_top.min():.1f} C   max {final_top.max():.1f} C")
print(f"  Bottom layer final: avg {final_bot.mean():.1f} C  min {final_bot.min():.1f} C  max {final_bot.max():.1f} C")
print(f"  Peak through-thickness gradient during sim: {peak_grad:.1f} C  (at t = {peak_grad_t:.1f} s)")
print(f"  Spray ON steps:  {n_on}  ({100*n_on/steps:.1f}%)")
print(f"  Spray OFF steps: {n_off} ({100*n_off/steps:.1f}%)")
print(f"  Spray on/off transitions: {transitions}")
print("="*70)


#UR5e URDF load + ikpy chain
print("Loading UR5e URDF...")
urdf = load_robot_description(cfg.robot.description)

#find URDF file path so ikpy can parse it directly
#yourdfpy already expanded the xacro when we called load_robot_description above.
#Save the expanded URDF to disk so ikpy can read it as a plain XML file.
urdf_file = os.path.join(RUNTIME_DIR, '_ur5e_expanded.urdf')
urdf.write_xml_file(urdf_file)
print(f"UR5e URDF written to: {urdf_file}")

# ikpy chain: only include the movable joints — mark fixed ones as inactive
# IMPORTANT:
# ikpy inserts its own OriginLink at index 0. That means the active_links_mask
# must include one extra False at the beginning, otherwise every joint index is shifted.
chain = Chain.from_urdf_file(
    urdf_file,
    base_elements=["base_link"],
    active_links_mask=[
        False,   # index 0: ikpy OriginLink
        False,   # index 1: base_link -> base_link_inertia fixed joint
        True,    # shoulder_pan_joint
        True,    # shoulder_lift_joint
        True,    # elbow_joint
        True,    # wrist_1_joint
        True,    # wrist_2_joint
        True,    # wrist_3_joint
        False,   # fixed flange/tool0 link if present
    ],
)

print(f"ikpy chain built with {len(chain.links)} links.")

print("\nIKPY CHAIN LINKS:")
for i, link in enumerate(chain.links):
    print(i, link.name, "active =", chain.active_links_mask[i])
print()

# joint names the URDF uses, in the same order the ikpy chain reports them
UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# tool0 orientation we want: nozzle pointing straight down (-Z world)
# ikpy target_orientation is a rotation matrix; we solve for orientation too
NOZZLE_DOWN_R = np.array([
    [1.0, 0.0,  0.0],
    [0.0, 1.0,  0.0],
    [0.0, 0.0, -1.0],
])

# ikpy target position is in robot base frame; convert world -> base

_last_joint_state = np.zeros(len(chain.links))


# arm extended forward and slightly down — closer to typical spray poses
set_initial_joint(_last_joint_state, chain, "shoulder_pan_joint", 0.0)
set_initial_joint(_last_joint_state, chain, "shoulder_lift_joint", -1.5708)
set_initial_joint(_last_joint_state, chain, "elbow_joint", 1.5708)
set_initial_joint(_last_joint_state, chain, "wrist_1_joint", -1.5708)
set_initial_joint(_last_joint_state, chain, "wrist_2_joint", -1.5708)
set_initial_joint(_last_joint_state, chain, "wrist_3_joint", 0.0)








# Convert JAX mesh arrays to NumPy once for geometric surface-normal lookup.
# This generalizes beyond a flat plate: nearest triangle gives local surface normal.
face_v0_np = np.array(face_v0)
face_v1_np = np.array(face_v1)
face_v2_np = np.array(face_v2)
face_centers_np = (face_v0_np + face_v1_np + face_v2_np) / 3.0
face_normals_np = np.array(face_normals)





def solve_ik_direct(target_world):
    """
    Direct orientation-aware IK initialization.

    Tool local +Z is aligned with the local surface normal.
    Since the spray is drawn from tool0 toward the surface, this makes the
    end effector normal to the part instead of sideways/upside-down.
    """
    target_world = np.array(target_world, dtype=float)
    target_base = world_to_base(target_world)

    _, n_world = nearest_surface_point_and_normal(face_centers_np, face_normals_np, target_world)

    try:
        q_raw = chain.inverse_kinematics(
            target_position=target_base,
            target_orientation=n_world,
            orientation_mode="Z",
            initial_position=_last_joint_state,
        )
    except Exception:
        q_raw = chain.inverse_kinematics(
            target_position=target_base,
            initial_position=_last_joint_state,
        )

    return unwrap_to_reference(q_raw, _last_joint_state)


def solve_ik(target_world):
    """
    Stable orientation-aware IK.

    Requirement:
      end-effector local +Z aligns with local surface normal.
      spray direction is therefore local -Z into the surface.

    This is the non-temporary version needed before moving to sphere/3D shapes.
    """
    global _last_joint_state

    target_world = np.array(target_world, dtype=float)
    target_base = world_to_base(target_world)

    _, n_world = nearest_surface_point_and_normal(face_centers_np, face_normals_np, target_world)

    try:
        q_raw = chain.inverse_kinematics(
            target_position=target_base,
            target_orientation=n_world,
            orientation_mode="Z",
            initial_position=_last_joint_state,
        )
    except Exception:
        q_raw = chain.inverse_kinematics(
            target_position=target_base,
            initial_position=_last_joint_state,
        )

    q_desired = unwrap_to_reference(q_raw, _last_joint_state)

    # Very mild wrist bias only. Strong wrist forcing fights true normal alignment.
    wrist_targets = {
        "wrist_1_joint": -1.5708,
        "wrist_2_joint": -1.5708,
        "wrist_3_joint":  0.0,
    }

    wrist_stiffness = 0.015
    for idx, link in enumerate(chain.links):
        if link.name in wrist_targets:
            preferred = _last_joint_state[idx] + (
                (wrist_targets[link.name] - _last_joint_state[idx] + np.pi)
                % (2.0 * np.pi)
                - np.pi
            )
            q_desired[idx] = (
                (1.0 - wrist_stiffness) * q_desired[idx]
                + wrist_stiffness * preferred
            )

    # Low-pass IK result. Small enough to suppress vibration, high enough to track path.
    alpha = 0.32
    q_next = (1.0 - alpha) * _last_joint_state + alpha * q_desired

    _last_joint_state = q_next
    return q_next



# UR5e mesh actors — parse URDF XML directly for link->mesh mapping
print("Building UR5e mesh actors...")


tree = ET.parse(urdf_file)
root = tree.getroot()

# URDF <mesh filename="..."/> paths use package:// prefixes; yourdfpy resolves
# these when it loads the scene, so we use its resolved geometry names.
# But we need a direct link->file mapping, so we parse XML and resolve paths
# ourselves using the URDF file directory as the resolution root.
urdf_dir = os.path.dirname(urdf_file)


# for each URDF link, collect its visual mesh files (in link-local frame)
# and the visual origin transform (link-local offset).
scene_geom_by_link = {}                       # link_name -> list of pv.PolyData
for link_elem in root.findall("link"):
    link_name = link_elem.get("name")
    for visual in link_elem.findall("visual"):
        origin = visual.find("origin")
        T_visual = np.eye(4)
        if origin is not None:
            xyz = origin.get("xyz", "0 0 0").split()
            rpy = origin.get("rpy", "0 0 0").split()
            T_visual[:3, 3] = [float(v) for v in xyz]
            # rpy -> rotation matrix
            r, p, y = [float(v) for v in rpy]
            cr, sr = np.cos(r), np.sin(r)
            cp, sp = np.cos(p), np.sin(p)
            cy, sy = np.cos(y), np.sin(y)
            Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
            Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
            Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
            T_visual[:3, :3] = Rz @ Ry @ Rx

        geometry = visual.find("geometry")
        if geometry is None:
            continue
        mesh_elem = geometry.find("mesh")
        if mesh_elem is None:
            continue
        mesh_file = resolve_mesh_path(urdf_dir, mesh_elem.get("filename"))
        if mesh_file is None or not os.path.exists(mesh_file):
            print(f"  skip {link_name}: mesh not found ({mesh_elem.get('filename')})")
            continue

        try:
            tm = trimesh.load(mesh_file, force='mesh')
        except Exception as e:
            print(f"  skip {link_name}: load failed ({e})")
            continue

        # apply optional scale from URDF
        scale_str = mesh_elem.get("scale")
        if scale_str:
            s = [float(v) for v in scale_str.split()]
            tm.vertices = tm.vertices * np.array(s)

        # apply visual origin so mesh is in link-local frame
        verts = np.array(tm.vertices)
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        verts_link = (T_visual @ verts_h.T).T[:, :3]

        faces_flat = np.hstack(
            [np.full((len(tm.faces), 1), 3, dtype=np.int64), tm.faces]
        ).flatten()
        pv_mesh = pv.PolyData(verts_link, faces_flat)

        scene_geom_by_link.setdefault(link_name, []).append(pv_mesh)
        print(f"  added link {link_name}: {len(verts)} verts")

print(f"Total links with meshes: {len(scene_geom_by_link)}")


# scene setup
print("Setting up visualization...")
print("=" * 72)
print("PYVISTA SHUTDOWN: return to the terminal and press Ctrl+C.")
print("DO NOT close the PyVista window with its X button.")
print("=" * 72)

# Volumetric grid replaces flat surface for POC 3 rendering
mesh_vis = vol_grid_pv
mesh_vis.cell_data['temperature'] = T_history_C_full[0]

plotter = pv.Plotter(shape=(1, 2), window_size=(1800, 900))
# INTERACTIVE TEMPERATURE HISTORY VIEW:
# LEFT SUBPLOT:
#   Existing robot + spray + heatmapped plate simulation.
# RIGHT SUBPLOT:
#   A normal Matplotlib plot rendered as an image on a PyVista plane.
#
# WHY THIS METHOD:
#   PyVista Chart2D / ChartMPL rendered poorly in this environment.
#   This approach uses real Matplotlib for clean labels/title/legend/grid, then
#   displays the plot as an image texture inside the PyVista window.

plotter.subplot(0, 1)
plotter.set_background("white")

history_curves = []
history_labels = []

def make_history_plot_image():
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=130)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_title("Clicked plate temperature history", color="black", fontsize=13)
    ax.set_xlabel("Time (s)", color="black", fontsize=11)
    ax.set_ylabel("Temperature (°C)", color="black", fontsize=11)

    ax.set_xlim(0.0, (T_history.shape[0] - 1) * dt_sim)
    ax.set_ylim(T_ambient, T_initial)
    ax.grid(True, alpha=0.30)

    ax.tick_params(axis="both", colors="black", labelsize=9)

    if len(history_curves) == 0:
        ax.text(
            0.5,
            0.5,
            "Click the plate on the left\nto add temperature histories",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="black",
            alpha=0.65,
            fontsize=12,
        )
    else:
        for temps, label in zip(history_curves, history_labels):
            ax.plot(time_axis_history, temps, linewidth=2.0, label=label)

        ax.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.tight_layout()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    w, h = canvas.get_width_height()
    rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4).copy()

    plt.close(fig)
    return rgba

# Create the first clean empty plot image.
time_axis_history = np.arange(T_history.shape[0]) * dt_sim
history_img = make_history_plot_image()

# Display the Matplotlib image as a PyVista texture on a plane.
history_texture = pv.Texture(history_img)

history_plane = pv.Plane(
    center=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
    i_size=1.42,
    j_size=1.02,
    i_resolution=1,
    j_resolution=1,
)

history_plane_actor = plotter.add_mesh(history_plane, texture=history_texture)
plotter.camera_position = "xy"
plotter.camera.zoom(1.15)
plotter.disable_parallel_projection()
plotter.enable_parallel_projection()

# The history graph is a Matplotlib image displayed on a PyVista plane.
# It should remain fixed rather than accepting 3D rotate, pan, or zoom input.
# Because subplot (0, 1) is active here, disable() affects the graph renderer.
plotter.disable()

plotter.subplot(0, 0)
plotter.set_background("black")

plotter.set_background("black")
plotter.set_background("black")

plotter.set_background('#0a0a0a')
plotter.enable_lightkit()
# Physical origin triad at robot base (0, 0, 0) — visible in scene, not a corner overlay
_axis_len = 0.15
_axis_r   = 0.008

def _force_on_top(actor):
    # bypass depth test so triad shows through robot mesh
    actor.SetPickable(False)
    actor.GetProperty().SetLighting(False)
    actor.GetProperty().SetAmbient(1.0)
    actor.GetProperty().SetDiffuse(0.0)
    actor.GetMapper().SetResolveCoincidentTopologyToPolygonOffset()
    actor.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(-10, -10)
    # move to overlay pass — always drawn last, no depth test against scene
    actor.SetForceOpaque(True)
    prop = actor.GetProperty()
    prop.SetRenderPointsAsSpheres(True)
    prop.SetRepresentationToSurface()

# X axis (red)
_x_arrow = pv.Arrow(start=(0,0,0), direction=(1,0,0), scale=_axis_len,
                    tip_length=0.20, tip_radius=0.03, shaft_radius=0.012)
_x_actor = plotter.add_mesh(_x_arrow, color='red', ambient=0.5)
_force_on_top(_x_actor)

# Y axis (green)
_y_arrow = pv.Arrow(start=(0,0,0), direction=(0,1,0), scale=_axis_len,
                    tip_length=0.20, tip_radius=0.03, shaft_radius=0.012)
_y_actor = plotter.add_mesh(_y_arrow, color='#00ff00', ambient=0.5)
_force_on_top(_y_actor)

# Z axis (cyan)
_z_arrow = pv.Arrow(start=(0,0,0), direction=(0,0,1), scale=_axis_len,
                    tip_length=0.20, tip_radius=0.03, shaft_radius=0.012)
_z_actor = plotter.add_mesh(_z_arrow, color='cyan', ambient=0.5)
_force_on_top(_z_actor)

# text labels at each arrow tip
plotter.add_point_labels(
    np.array([[_axis_len*1.15, 0, 0], [0, _axis_len*1.15, 0], [0, 0, _axis_len*1.15]]),
    ['+X', '+Y', '+Z'],
    text_color='white',
    font_size=14,
    point_size=1,
    shape=None,
    always_visible=True,
)

# ground plane
ground = pv.Plane(center=(0.4, 0, -0.155), direction=(0, 0, 1), i_size=2.0, j_size=2.0)
ground_actor = plotter.add_mesh(ground, color='#151515', show_edges=False)
ground_actor.SetPickable(False)

# plate heatmap - volumetric with mesh edges visible
heatmap_actor = plotter.add_mesh(
    mesh_vis, scalars='temperature', cmap='inferno',
    show_edges=True, edge_color='#101010', line_width=0.3,
    clim=[T_ambient, T_initial],
    scalar_bar_args={
        'title': 'Temperature (°C)',
        'color': 'white',
        'title_font_size': 16,
        'label_font_size': 12,
    }
)

# add each UR5e link mesh as a PyVista actor with ROBOT_BASE translation baked in
robot_base_T = np.eye(4)
robot_base_T[:3, 3] = ROBOT_BASE

link_actors = {} # link_name -> list of actors
for link_name, meshes in scene_geom_by_link.items():
    link_actors[link_name] = []
    for m in meshes:
        actor = plotter.add_mesh(
            m,
            color='#4a90d9',
            smooth_shading=True,
            specular=0.4, specular_power=15, ambient=0.25,
        )
        actor.SetPickable(False)   # exclude robot arm from click picking
        link_actors[link_name].append(actor)


def update_ur5e_pose(q):
    """Set URDF config, then push each link's world transform to its actors."""
    cfg = q_to_cfg(UR5E_JOINT_NAMES, chain, q)
    urdf.update_cfg(cfg)

    for link_name, actors in link_actors.items():
        T_link = np.array(urdf.get_transform(link_name))
        T_world = robot_base_T @ T_link
        for actor in actors:
            set_actor_matrix(actor, T_world)














# organized pressure-jet spray visualization
# Visual only. This starts at actual robot tool0 and lands on commanded target.
# This version is tighter and more coherent than the previous one.
N_CORE_LINES = 48
N_OUTER_LINES = 32
N_JET_LINES = N_CORE_LINES + N_OUTER_LINES

JET_CORE_RADIUS = 0.0025      # very tight stream at end effector
JET_EXIT_RADIUS = 0.026       # tighter impact footprint
JET_MID_RADIUS  = 0.012       # tighter middle body

_core_theta = np.linspace(0.0, 2.0 * np.pi, N_CORE_LINES, endpoint=False)
_outer_theta = np.linspace(0.0, 2.0 * np.pi, N_OUTER_LINES, endpoint=False)

_core_r = np.full(N_CORE_LINES, 0.18)
_outer_r = np.linspace(0.45, 1.0, N_OUTER_LINES)

_theta = np.concatenate([_core_theta, _outer_theta])
_r_unit = np.concatenate([_core_r, _outer_r])
_phase = np.linspace(0.0, 2.0 * np.pi, N_JET_LINES, endpoint=False)

def make_pressure_jet(jet_origin, impact_center, t):
    """
    Build a coherent jet from robot tool0 to target impact point.
    Much less shimmer / randomness than before.
    """
    jet_origin = np.array(jet_origin, dtype=float)
    impact_center = np.array(impact_center, dtype=float)

    p0 = jet_origin.copy()
    p2_center = np.array([impact_center[0], impact_center[1], PLATE_Z + 0.002])

    axis = p2_center - p0
    axis_norm = np.linalg.norm(axis) + 1e-8
    d = axis / axis_norm

    # Stable perpendicular basis around jet axis.
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(d, tmp)) > 0.95:
        tmp = np.array([1.0, 0.0, 0.0])

    u = np.cross(d, tmp)
    u = u / (np.linalg.norm(u) + 1e-8)
    v = np.cross(d, u)
    v = v / (np.linalg.norm(v) + 1e-8)

    # Top bundle: nearly straight and very tight.
    top_radius = JET_CORE_RADIUS * _r_unit
    top_offsets = (
        top_radius[:, None] * np.cos(_theta)[:, None] * u[None, :]
        + top_radius[:, None] * np.sin(_theta)[:, None] * v[None, :]
    )
    p_top = p0[None, :] + top_offsets

    # Mid bundle: slight widening, almost no wobble.
    p1_center = p0 * 0.48 + p2_center * 0.52
    mid_radius = JET_MID_RADIUS * _r_unit
    mid_angle = _theta + 0.010 * np.sin(3.0 * t + _phase)
    mid_offsets = (
        mid_radius[:, None] * np.cos(mid_angle)[:, None] * u[None, :]
        + mid_radius[:, None] * np.sin(mid_angle)[:, None] * v[None, :]
    )
    p_mid = p1_center[None, :] + mid_offsets

    # Bottom bundle: organized impact cone, very mild shimmer only.
    bot_radius = JET_EXIT_RADIUS * _r_unit + 0.0008 * np.sin(6.0 * t + _phase)
    bot_angle = _theta + 0.015 * np.sin(2.5 * t)
    bot_offsets = (
        bot_radius[:, None] * np.cos(bot_angle)[:, None] * u[None, :]
        + bot_radius[:, None] * np.sin(bot_angle)[:, None] * v[None, :]
    )
    p_bot = p2_center[None, :] + bot_offsets
    p_bot[:, 2] = np.maximum(p_bot[:, 2], PLATE_Z + 0.002)

    points = np.vstack([p_top, p_mid, p_bot]).astype(np.float32)

    lines = []
    for i in range(N_JET_LINES):
        lines.extend([2, i, i + N_JET_LINES])
        lines.extend([2, i + N_JET_LINES, i + 2 * N_JET_LINES])

    return pv.PolyData(points, lines=np.array(lines))

# tiny visual emitter at the robot end-effector.
# This prevents the jet from looking disconnected from the wrist.
spray_head_mesh = pv.Sphere(
    radius=0.012,
    center=(0, 0, 0),
    theta_resolution=20,
    phi_resolution=20,
)

spray_head_actor = plotter.add_mesh(
    spray_head_mesh,
    color='cyan',
    smooth_shading=True,
    specular=0.4,
    specular_power=20,
    opacity=0.85,
)
spray_head_actor.SetPickable(False)


# initial pose + pressure jet actor
q_init = solve_ik_direct(np.array(waypoints[0]))
_last_joint_state = q_init.copy()
update_ur5e_pose(q_init)

tool0_init = get_tool0_world(UR5E_JOINT_NAMES, chain, robot_base_T, urdf, q_init)
impact_init = np.array(waypoints[0], dtype=float)

set_spray_head_position(spray_head_actor, tool0_init)

spray_poly = make_pressure_jet(tool0_init, impact_init, 0.0)
spray_actor = plotter.add_mesh(
    spray_poly,
    color='cyan',
    line_width=4,
    opacity=0.95,
    render_lines_as_tubes=True,
)
spray_actor.SetPickable(False)

# Keep spray hidden while IK path is being precomputed.
spray_actor.SetVisibility(False)
spray_head_actor.SetVisibility(False)

plotter.camera_position = [(1.22, -1.10, 0.77), (0.36, 0.0, 0.05), (0, 0, 1)]
plotter.camera.SetClippingRange(0.01, 10.0)
# clickable plate temperature-history callback
print("Enabling clickable plate temperature-history picking...")

# Use per-cell centroids so clicks can land on top, side, or bottom cells.
plate_pick_centers = cell_centroids_np

# Plate-only click filter.
plate_x_min, plate_x_max = mesh_pv.bounds[0], mesh_pv.bounds[1]
plate_y_min, plate_y_max = mesh_pv.bounds[2], mesh_pv.bounds[3]
plate_z_ref = PLATE_Z
plate_z_tol = 0.100
plate_xy_pad = 0.025

picked_face_ids = []
picked_marker_actors = []

def refresh_history_panel():
    global history_texture

    new_img = make_history_plot_image()

    # Update the existing texture's image data in place instead of
    # removing and re-adding the actor. This forces the GPU to reupload
    # the new pixels immediately on next render.
    new_texture = pv.Texture(new_img)
    history_plane_actor.SetTexture(new_texture)
    history_texture = new_texture   # hold reference so it doesn't get GC'd

    plotter.render()


def add_temperature_curve_from_point(p_click):
    p_click = np.array(p_click, dtype=float)

    # Broadened plate bounding box that includes the full volumetric slab
    # (top face at plate_z_ref, bottom face PLATE_THICKNESS below).
    z_lo = plate_z_ref - PLATE_THICKNESS - 0.020   # 20 mm slack
    z_hi = plate_z_ref + 0.020
    inside_x = (plate_x_min - plate_xy_pad) <= p_click[0] <= (plate_x_max + plate_xy_pad)
    inside_y = (plate_y_min - plate_xy_pad) <= p_click[1] <= (plate_y_max + plate_xy_pad)
    inside_z = z_lo <= p_click[2] <= z_hi

    if not (inside_x and inside_y and inside_z):
        print(
            "Ignored click not on plate volume: "
            f"x={p_click[0]:.3f}, y={p_click[1]:.3f}, z={p_click[2]:.3f}"
        )
        return

    # Nearest CELL centroid (not just surface triangle center).
    dists = np.linalg.norm(plate_pick_centers - p_click[None, :], axis=1)
    cell_idx = int(np.argmin(dists))
    center = plate_pick_centers[cell_idx]

    if dists[cell_idx] > 0.055:
        print(f"Ignored click too far from plate cell: distance={dists[cell_idx]:.3f} m")
        return

    if cell_idx in picked_face_ids:
        print(f"Cell {cell_idx} already plotted, skipping.")
        return

    # Pull temperature history from the FULL 3D volume history, not top-only.
    temps = T_history_C_full[:, cell_idx]

    # Which layer / face-type did we click on? Helpful for the label.
    layer_of_cell = cell_idx // n_surface
    if layer_of_cell == 0:
        face_label = "top"
    elif layer_of_cell == N_LAYERS - 1:
        face_label = "bottom"
    else:
        face_label = f"layer {layer_of_cell}"

    center_base = center - ROBOT_BASE
    label = (f"({center_base[0]*1000:.0f}, {center_base[1]*1000:.0f}, "
             f"{center_base[2]*1000:.0f}) mm from base [{face_label}]")

    history_curves.append(temps)
    history_labels.append(label)
    picked_face_ids.append(cell_idx)

    refresh_history_panel()

    # Offset marker outward from the plate so it's visible from the click side.
    # Determine which face of the volume the cell centroid is closest to.
    plate_top_z    = plate_z_ref
    plate_bottom_z = plate_z_ref - PLATE_THICKNESS
    marker_offset  = 0.010   # [m] 10 mm outside the plate

    marker_center = center.copy()
    if layer_of_cell == 0:
        # Top-face cell: pop marker up into free space above the plate
        marker_center[2] = plate_top_z + marker_offset
    elif layer_of_cell == N_LAYERS - 1:
        # Bottom-face cell: pop marker down below the plate
        marker_center[2] = plate_bottom_z - marker_offset
    else:
        # Interior/side cell: push marker sideways along whichever XY axis
        # points outward from plate center.
        plate_cx = 0.5 * (plate_x_min + plate_x_max)
        plate_cy = 0.5 * (plate_y_min + plate_y_max)
        dx = center[0] - plate_cx
        dy = center[1] - plate_cy
        if abs(dx) >= abs(dy):
            marker_center[0] = center[0] + marker_offset * np.sign(dx if dx != 0 else 1.0)
        else:
            marker_center[1] = center[1] + marker_offset * np.sign(dy if dy != 0 else 1.0)

    plotter.subplot(0, 0)
    marker_sphere = pv.Sphere(radius=0.008, center=marker_center)
    marker_actor = plotter.add_mesh(marker_sphere, name=f"picked_cell_{cell_idx}", color='cyan')
    picked_marker_actors.append(marker_actor)

    print(f"Added temperature history: {label}")
    plotter.render()


def on_click_position_for_history(pick_world, picker=None):
    # click_xy is screen position. Convert it to a 3D pick on the left renderer.
    print(f"[click received] xy = {pick_world}") 
    add_temperature_curve_from_point(pick_world)


    picker = pv._vtk.vtkCellPicker()
    picker.SetTolerance(0.001)

    left_renderer = plotter.renderers[0]
    picker.Pick(x, y, 0, left_renderer)

    cell_id = picker.GetCellId()
    if cell_id < 0:
        print("Ignored click: no mesh cell picked.")
        return

    p_world = picker.GetPickPosition()
    add_temperature_curve_from_point(p_world)


plotter.subplot(0, 0)

# Use PyVista's click-position tracker instead of Chart2D picking.
# This records the mouse click position and then we manually pick only against
# the left renderer.
plotter.enable_point_picking(
    callback=on_click_position_for_history,
    show_message=False,
    tolerance=0.005,
    left_clicking=True,
    picker='point',
    use_picker=True,
    show_point=False,
    pickable_window=False,
)

print("Clickable plate temperature-history picking enabled.")

plotter.show(auto_close=False, interactive_update=True)

# animation loop


print("Animating...")

# More frames = smoother visual path.
skip       = max(1, steps // 1800)
frame_list = list(range(0, steps, skip))
frame_dt   = skip * dt_sim

FRAME_TIME = 0.055   # [s] increased from 0.022 for more visible spray on/off pulsing

# ------------------------------------------------------------------
# Precompute smooth visual target path.
# ------------------------------------------------------------------
print("Precomputing smooth visual target path...")

visual_targets = []
visual_target = np.array(waypoints[0], dtype=np.float32)

for frame in frame_list:
    path_pos = frame / float(steps_per_move)
    i0 = int(np.floor(path_pos))
    i1 = min(i0 + 1, n_waypoints - 1)

    tau = np.clip(path_pos - i0, 0.0, 1.0)

    # smoother than ordinary smoothstep: reduces start/stop jerk
    s = tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)

    w0 = np.array(waypoints[i0], dtype=np.float32)
    w1 = np.array(waypoints[i1], dtype=np.float32)

    raw_target = (1.0 - s) * w0 + s * w1

    # Low-pass the target to soften row/column transitions.
    visual_alpha = 0.09
    visual_target = (1.0 - visual_alpha) * visual_target + visual_alpha * raw_target
    visual_targets.append(visual_target.copy())

visual_targets = np.array(visual_targets)

# ------------------------------------------------------------------
# Precompute orientation-aware IK path.
# ------------------------------------------------------------------
print("Precomputing normal-aligned IK joint path...")

_last_joint_state = solve_ik_direct(visual_targets[0])
q_anim_path = []

for target in visual_targets:
    q = solve_ik(target)
    q_anim_path.append(q.copy())

q_anim_path = np.array(q_anim_path)
q_anim_path = smooth_joint_path(chain, q_anim_path, passes=4)

print(f"Animation frames: {len(frame_list)}")
print("Done precomputing normal-aligned animation path.")

# Physics-robot coupling diagnostic: how well does IK track the commanded path?
# If tracking is tight, waypoint-based PULSE spray masks are physically valid.
# If tracking has significant error, real coupling would require recomputing
# PULSE with tool0-world positions and re-running the physics scan.
_tool0_positions = []
for q in q_anim_path:
    _tool0_positions.append(get_tool0_world(UR5E_JOINT_NAMES, chain, robot_base_T, urdf, q))
_tool0_positions = np.array(_tool0_positions)
_ik_errors = np.linalg.norm(_tool0_positions - visual_targets, axis=1)
print("")
print("Physics-robot coupling report:")
print(f"  IK tracking error: mean={_ik_errors.mean()*1000:.3f} mm, "
      f"max={_ik_errors.max()*1000:.3f} mm")
if _ik_errors.max() < 0.001:
    print(f"  --> waypoint-based PULSE spray masks are numerically valid ")
    print(f"      (IK error << spray footprint scale)")
else:
    print(f"  --> IK tracking exceeds 1 mm; consider re-running physics with ")
    print(f"      tool0-based PULSE for tighter physics-robot coupling")
print("")

# Turn spray on only after trajectory planning is complete.
# Actual visibility is now controlled per-frame by the physics valve state.
spray_actor.SetVisibility(True)
spray_head_actor.SetVisibility(True)
SPRAY_VISIBLE_THRESHOLD = 0.05   # [-] fade in once physics says >5% valve intensity

# set first pose before display loop starts
q_first = q_anim_path[0]
update_ur5e_pose(q_first)

tool0_first = get_tool0_world(UR5E_JOINT_NAMES, chain, robot_base_T, urdf, q_first)
set_spray_head_position(spray_head_actor, tool0_first)

# ------------------------------------------------------------------
# Replay cached trajectory.
# ------------------------------------------------------------------
while True:
    for j, frame in enumerate(frame_list):
        t_start = time.time()
        sim_time = frame * dt_sim

        target = visual_targets[j]
        q = q_anim_path[j]

        # plate temperature: full 3D volume, all cells updated
        full_T_frame = T_history_C_full[frame]
        mesh_vis.cell_data['temperature'] = full_T_frame
        heatmap_actor.mapper.dataset.cell_data['temperature'] = full_T_frame
        heatmap_actor.mapper.dataset.Modified()

        # UR5e pose update from cached joint path
        update_ur5e_pose(q)

        # Tool0 should now be approximately normal to local mesh surface.
        tool0_world = get_tool0_world(UR5E_JOINT_NAMES, chain, robot_base_T, urdf, q)
        jet_origin = tool0_world.copy()

        # Impact point comes from nearest mesh surface, not hardcoded plate z.
        # This is what will matter on sphere / irregular 3D shapes.
        impact_center, surf_normal = nearest_surface_point_and_normal(face_centers_np, face_normals_np, target)

        set_spray_head_position(spray_head_actor, jet_origin)

        if j == 0:
            err_world = np.linalg.norm(tool0_world - target)
            alignment = np.dot(get_nozzle_direction_world(UR5E_JOINT_NAMES, chain, robot_base_T, urdf, q), -surf_normal)
            print(f"Start-of-pass tool0 tracking error: {err_world:.3f} m")
            print(f"Start-of-pass spray-normal alignment: {alignment:.3f}")

        # Pure snake pattern: spray always on at full intensity
        spray_actor.SetVisibility(True)
        spray_actor.GetProperty().SetOpacity(0.95)

        new_spray = make_pressure_jet(jet_origin, impact_center, sim_time)
        spray_poly.points = new_spray.points
        spray_poly.lines = new_spray.lines
        spray_poly.Modified()
        spray_actor.mapper.dataset.Modified()

        plotter.camera.SetClippingRange(0.01, 10.0)
        plotter.update(stime=1, force_redraw=True)

        elapsed = time.time() - t_start
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)