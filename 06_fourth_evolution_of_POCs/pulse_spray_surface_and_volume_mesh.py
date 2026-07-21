"""
POC 2 - Pulsed Spray Cooling on a 3D Triangular Prism Volume Mesh with Moving Nozzle

WHAT THIS FILE IS
    A 3D finite-volume simulation of pulsed water-spray cooling on a flat
    steel plate, meshed as triangular prisms extruded downward from the
    surface .obj mesh. Nozzle moves along a snake pattern above the top
    face. Full POC 1 physics module applied per prism cell, with face-type-
    dependent boundary conditions (top: spray+ambient+radiation, bottom
    and sides: ambient+radiation, interior: conduction only).

GOVERNING EQUATION (per prism cell)
    rho c_p V_i dT_i/dt = sum_j (k_face A_ij / d_ij)(T_j - T_i)
                          + Q_boundary_i

    where A_ij, d_ij are face area and center-to-center distance between
    cells i and j. Divided by (rho c_p V_i) to get dT/dt in K/s.

PHYSICAL MECHANISMS
    Same as POC 1 - conduction, spray convection (via PULSE per-face
    weights), ambient convection, radiation, T-dependent k/cp/eps,
    boiling curve h(T_surface), adaptive hysteretic control, S-curve
    valve dynamics, smoothstep h response.

    NEW: through-thickness resolution via 6-layer prism extrusion.
    Boiling curve h(T) evaluated at TRUE top-layer T (not depth-averaged),
    fixing Bi > 0.1 limitation of POC 1. Face-type-dependent boundary
    conditions: spray on top only, ambient+radiation on bottom+sides.

NUMERICAL METHOD
    IMEX backward Euler. Proper finite-volume geometry-weighted Laplacian
    (precomputed G_ij = A_ij/(d_ij V_i) coefficients per neighbor).
    Conduction and convection implicit via matrix-free CG. Radiation
    and T-dependent coefficients lagged at T_old.

LIMITATIONS
    - Cell-centered k in conduction (asymmetric flux at interfaces).
      For POC; harmonic-mean face k is more rigorous, upgrade later.
    - Prism extrusion assumes vertical stacking from a flat surface.
      Curved geometry needs normal-direction extrusion (later evolution).
    - Boiling curve is Option B generic piecewise, not validated
      correlation for a specific nozzle.
    - PULSE deposit() weights normalized to peak = 1.0; absolute spray
      mass flux not calibrated against measured profiles.
    - No latent heat of phase transformation.
    - No spray hydrodynamics (vapor blanket, pooling).
    - Radiation view factor = 1.0 (flat plate).
"""

from relevant_PULSE_files.jax_kernels import Pose, deposit
from relevant_PULSE_files.jax_pulse import Pulse
import jax
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pyvista as pv
import time
from collections import defaultdict
import os

# ============================================================================
# Material: AISI 1045 (T-dependent from Incropera AISI 1010 table)
# ============================================================================
RHO       = 7850.0              # [kg/m^3]  density of AISI 1045 steel (nominal)
SIGMA_SB  = 5.670374419e-8      # [W/(m^2 K^4)]  Stefan-Boltzmann constant
T_TABLE_K = jnp.array([300.0, 400.0, 600.0, 800.0, 1000.0, 1200.0])   # [K]           tabulated T nodes (Incropera A.1)
K_TABLE   = jnp.array([ 63.9,  58.7,  48.8,  39.2,   31.3,   28.0])   # [W/(m K)]     thermal conductivity vs T
CP_TABLE  = jnp.array([434.0, 487.0, 559.0, 685.0, 1168.0,  600.0])   # [J/(kg K)]    specific heat vs T
EPS_TABLE = jnp.array([ 0.70,  0.73,  0.78,  0.82,   0.85,   0.85])   # [-]           surface emissivity vs T (oxidized steel)
def k_of_T(T):   return jnp.interp(T, T_TABLE_K, K_TABLE)
def cp_of_T(T):  return jnp.interp(T, T_TABLE_K, CP_TABLE)
def eps_of_T(T): return jnp.interp(T, T_TABLE_K, EPS_TABLE)
ALPHA_REF = float(k_of_T(jnp.array(500.0)) / (RHO * cp_of_T(jnp.array(500.0))))   # [m^2/s]  reference alpha at T=500K, for diagnostics only

# ============================================================================
# Geometry (prism extrusion, graded finer at top)
# ============================================================================
PLATE_THICKNESS = 0.020   # [m]  20 mm total plate thickness
LAYER_THICKS    = np.array([0.0010, 0.0020, 0.0030, 0.0040, 0.0050, 0.0050])   # [m]  per-layer thickness top->bottom (graded finer at spray surface)
N_LAYERS        = len(LAYER_THICKS)      # [-]  number of prism layers through thickness
assert abs(LAYER_THICKS.sum() - PLATE_THICKNESS) < 1e-9

# ============================================================================
# HTC
# ============================================================================
H_AMBIENT  = 15.0                                                     # [W/(m^2 K)]  natural convection to still air
H_BOIL_T_C = jnp.array([  25.0,  100.0,  200.0,  300.0,  400.0,  500.0,  900.0])   # [C]           surface T nodes for boiling curve
H_BOIL_VAL = jnp.array([3000.0, 3000.0,30000.0,40000.0,15000.0, 3000.0, 2000.0])   # [W/(m^2 K)]   h_spray vs surface T (Option B generic curve)
def h_spray_of_T(T_K):
    return jnp.interp(T_K - 273.15, H_BOIL_T_C, H_BOIL_VAL)

# ============================================================================
# Temperatures
# ============================================================================
T_INIT_C     = 900.0                     # [C]  initial plate temperature
T_AMBIENT_C  = 25.0                      # [C]  ambient air / spray water temperature
T_INIT_K     = T_INIT_C    + 273.15      # [K]
T_AMBIENT_K  = T_AMBIENT_C + 273.15      # [K]

# ============================================================================
# Time and control
# ============================================================================
DT                = 0.1                  # [s]   time step (backward Euler, chosen for accuracy not stability)
T_MAX             = 1500.0               # [s]   simulation end time
MAX_STEPS         = int(T_MAX / DT)      # [-]   number of integration steps
T_COOL_ENOUGH_C   = 300.0                # [C]   quench target: cool below this to declare success
# Hysteresis thresholds relative to (plate_avg - T_ambient). Fixes bug where
# absolute K thresholds became impossible to satisfy once plate cooled near ambient.
DELTA_OFF_FRAC    = 0.30                 # [-]   spray OFF when zone drops this fraction of (T_plate_avg - T_ambient) below plate avg
DELTA_ON_FRAC     = 0.10                 # [-]   spray ON  when zone recovers to within this fraction of plate avg
SPRAY_RAMP_TAU_S  = 1.0                  # [s]   per-stage 1st-order lag; 2 stages give S-curve valve response

# ============================================================================
# PULSE
# ============================================================================
PULSE_SIGMA        = 0.15   # [-] PULSE cone width parameter (see relevant_PULSE_files/jax_kernels.py)
PULSE_A            = 1.0     # [-] PULSE amplitude parameter
PULSE_REF_DIST     = 0.20   # [m] reference distance for PULSE deposition falloff
PULSE_RESOLUTION   = 64      # [-] raycasting grid resolution per axis
PULSE_FOV          = 90.0    # [deg] full field-of-view of spray cone (verify degree/radian in kernels)
PULSE_VFLOW        = 1e-5    # [m^3/s] volumetric flow rate through nozzle
Z_HEIGHT           = 0.20    # [m] 20cm nozzle standoff (realistic UR5e end-effector clearance)
PATH_N_PER_SIDE    = 20      # [-]  path grid resolution per side (total poses = PATH_N_PER_SIDE^2)
STEPS_PER_POSE     = 30      # [-]  integration steps per nozzle pose; dwell_time [s] = STEPS_PER_POSE * DT = 3.0 s

# ============================================================================
# Load surface mesh
# ============================================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
mesh_path = os.path.join(BASE_DIR, '..', "04_third evoluiton of poc's",
                        'relevant_PULSE_files', 'refined_plate.obj')

print("Generating 0.2m x 0.2m plate surface mesh...")
PLATE_XY_SIZE = 0.20     # [m]  200 mm plate side length (X and Y extents)
NX_MESH = 32             # [-]  surface mesh grid cells per side; total triangles = 2 * NX_MESH^2
verts_gen  = []
faces_gen  = []
xs = np.linspace(0.0, PLATE_XY_SIZE, NX_MESH + 1)
ys = np.linspace(0.0, PLATE_XY_SIZE, NX_MESH + 1)
for iy in range(NX_MESH + 1):
    for ix in range(NX_MESH + 1):
        verts_gen.append([xs[ix], ys[iy], 0.0])
verts_gen = np.array(verts_gen, dtype=np.float64)
def vid(ix, iy): return iy * (NX_MESH + 1) + ix
for iy in range(NX_MESH):
    for ix in range(NX_MESH):
        v00 = vid(ix,   iy  )
        v10 = vid(ix+1, iy  )
        v01 = vid(ix,   iy+1)
        v11 = vid(ix+1, iy+1)
        faces_gen.append([v00, v10, v11])
        faces_gen.append([v00, v11, v01])
faces_gen = np.array(faces_gen, dtype=np.int64)

# Save to .obj for PULSE to load
gen_mesh_path = os.path.join(BASE_DIR, "generated_plate_02m.obj")
with open(gen_mesh_path, "w") as f:
    for v in verts_gen:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for face in faces_gen:
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
mesh_path = gen_mesh_path

print("Loading generated mesh into PULSE...")
pulse_model = Pulse(
    sigma=PULSE_SIGMA, a=PULSE_A, ref_dist=PULSE_REF_DIST,
    resolution=PULSE_RESOLUTION, fov=PULSE_FOV,
    volumetric_flow_rate=PULSE_VFLOW,
)
pulse_model.load_mesh(mesh_path)

face_v0      = pulse_model.face_v0
face_v1      = pulse_model.face_v1
face_v2      = pulse_model.face_v2
face_normals = pulse_model.face_normals
n_surface    = pulse_model.n_faces
areas_surf   = np.array(pulse_model.areas)
print(f"  surface triangles: {n_surface}")
x_extent = float(face_v0[:,0].max() - face_v0[:,0].min())
y_extent = float(face_v0[:,1].max() - face_v0[:,1].min())
print(f"  extents: {x_extent:.2f} x {y_extent:.2f} (mesh units)")

# ============================================================================
# Volume mesh cells
# ============================================================================
n_cells = n_surface * N_LAYERS
print(f"Building volume mesh: {n_cells} cells ({N_LAYERS} layers x {n_surface} triangles)")

cell_volume_np = np.zeros(n_cells, dtype=np.float64)
for layer in range(N_LAYERS):
    cell_volume_np[layer*n_surface:(layer+1)*n_surface] = areas_surf * LAYER_THICKS[layer]
cell_volume = jnp.array(cell_volume_np)

cell_layer_np = np.zeros(n_cells, dtype=np.int32)
for layer in range(N_LAYERS):
    cell_layer_np[layer*n_surface:(layer+1)*n_surface] = layer
is_top_np    = (cell_layer_np == 0).astype(np.float64)
is_bottom_np = (cell_layer_np == N_LAYERS - 1).astype(np.float64)
is_top_jax    = jnp.array(is_top_np)
is_bottom_jax = jnp.array(is_bottom_np)

face_area_top_np    = np.where(cell_layer_np == 0,            np.tile(areas_surf, N_LAYERS), 0.0)
face_area_bottom_np = np.where(cell_layer_np == N_LAYERS - 1, np.tile(areas_surf, N_LAYERS), 0.0)
face_area_top_jax    = jnp.array(face_area_top_np)
face_area_bottom_jax = jnp.array(face_area_bottom_np)

# ============================================================================
# Surface adjacency (in-layer)
# ============================================================================
print("Building surface adjacency...")
mesh_pv  = pv.read(mesh_path).triangulate()
faces_np = np.array(mesh_pv.faces).reshape(-1, 4)[:, 1:]
verts_np = np.array(mesh_pv.points)

# Map edge -> [faces sharing it]
edge_to_faces = defaultdict(list)
edge_length   = {}
for fi, face in enumerate(faces_np):
    for j in range(3):
        v0, v1 = face[j], face[(j+1) % 3]
        edge = tuple(sorted([v0, v1]))
        edge_to_faces[edge].append(fi)
        if edge not in edge_length:
            edge_length[edge] = float(np.linalg.norm(verts_np[v0] - verts_np[v1]))

MAX_LATERAL = 3                # [-]  max in-layer neighbors per triangle (edge-sharing)
lateral_nb_surf   = -1 * np.ones((n_surface, MAX_LATERAL), dtype=np.int32)
lateral_edge_len  = np.zeros((n_surface, MAX_LATERAL), dtype=np.float64)  # shared edge length
for fi, face in enumerate(faces_np):
    slot = 0
    for j in range(3):
        edge = tuple(sorted([face[j], face[(j+1) % 3]]))
        for fj in edge_to_faces[edge]:
            if fj != fi and slot < MAX_LATERAL:
                lateral_nb_surf[fi, slot]  = fj
                lateral_edge_len[fi, slot] = edge_length[edge]
                slot += 1

n_lateral_nb_surf = np.sum(lateral_nb_surf >= 0, axis=1)

# Side-face area per cell: sum of missing-lateral edges * layer thickness
face_area_side_np = np.zeros(n_cells, dtype=np.float64)
for fi in range(n_surface):
    # Total edge length of this triangle
    total_edge_len = 0.0
    exposed_len    = 0.0
    for j in range(3):
        edge = tuple(sorted([faces_np[fi, j], faces_np[fi, (j+1) % 3]]))
        total_edge_len += edge_length[edge]
        if len(edge_to_faces[edge]) == 1:  # only this face uses it => boundary edge
            exposed_len += edge_length[edge]
    for layer in range(N_LAYERS):
        ci = layer * n_surface + fi
        face_area_side_np[ci] = exposed_len * LAYER_THICKS[layer]
face_area_side_jax = jnp.array(face_area_side_np)

# ============================================================================
# Full 3D adjacency + geometric coefficients G_ij = A_ij / (d_ij * V_i)
# ============================================================================
MAX_NB_3D = 5                  # [-]  max 3D cell neighbors: 3 lateral (in-layer) + 2 vertical (adjacent layers)
neighbors_3d = -1 * np.ones((n_cells, MAX_NB_3D), dtype=np.int32)
G_matrix_np  = np.zeros((n_cells, MAX_NB_3D), dtype=np.float64)  # coefficient A_ij / (d_ij * V_i)

# Triangle centroids for lateral distances
tri_centroids = np.zeros((n_surface, 3), dtype=np.float64)
for fi in range(n_surface):
    tri_centroids[fi] = (verts_np[faces_np[fi, 0]] + verts_np[faces_np[fi, 1]] + verts_np[faces_np[fi, 2]]) / 3.0

for layer in range(N_LAYERS):
    L_i = LAYER_THICKS[layer]
    for si in range(n_surface):
        ci = layer * n_surface + si
        V_i = areas_surf[si] * L_i
        slot = 0
        # Lateral neighbors
        for k in range(MAX_LATERAL):
            nb_s = lateral_nb_surf[si, k]
            if nb_s >= 0:
                edge_len = lateral_edge_len[si, k]
                A_ij = edge_len * L_i                                          # shared face area
                d_ij = float(np.linalg.norm(tri_centroids[si] - tri_centroids[nb_s]))
                d_ij = max(d_ij, 1e-9)
                neighbors_3d[ci, slot] = layer * n_surface + nb_s
                G_matrix_np[ci, slot]  = A_ij / (d_ij * V_i)
                slot += 1
        # Up (layer - 1)
        if layer > 0:
            L_up = LAYER_THICKS[layer - 1]
            A_ij = areas_surf[si]                # top face of cell i, bottom face of cell above
            d_ij = 0.5 * (L_i + L_up)
            neighbors_3d[ci, slot] = (layer - 1) * n_surface + si
            G_matrix_np[ci, slot]  = A_ij / (d_ij * V_i)
            slot += 1
        # Down (layer + 1)
        if layer < N_LAYERS - 1:
            L_dn = LAYER_THICKS[layer + 1]
            A_ij = areas_surf[si]
            d_ij = 0.5 * (L_i + L_dn)
            neighbors_3d[ci, slot] = (layer + 1) * n_surface + si
            G_matrix_np[ci, slot]  = A_ij / (d_ij * V_i)
            slot += 1

neighbors_3d_jax = jnp.array(neighbors_3d)
G_matrix_jax     = jnp.array(G_matrix_np)
print(f"  3D adjacency built. G_matrix stats: min {G_matrix_np[G_matrix_np>0].min():.3e}  "
      f"max {G_matrix_np.max():.3e}  mean {G_matrix_np[G_matrix_np>0].mean():.3e} (units 1/m^2)")

# ============================================================================
# Snake nozzle path
# ============================================================================
margin = 0.01
x_min = float(face_v0[:,0].min()) + margin
x_max = float(face_v0[:,0].max()) - margin
y_min = float(face_v0[:,1].min()) + margin
y_max = float(face_v0[:,1].max()) - margin
x_range = np.linspace(x_min, x_max, PATH_N_PER_SIDE)
y_range = np.linspace(y_min, y_max, PATH_N_PER_SIDE)
poses = []
for i, y in enumerate(y_range):
    xs = x_range if i % 2 == 0 else x_range[::-1]
    for x in xs:
        pos = jnp.array([x, y, Z_HEIGHT])
        rot = jnp.array([1.0, 0.0, 0.0, 0.0])
        poses.append(Pose(position=pos, rotation=rot))
n_poses = len(poses)
pose_positions = jnp.stack([p.position for p in poses])
pose_rotations = jnp.stack([p.rotation for p in poses])
print(f"  poses: {n_poses}, dwell: {STEPS_PER_POSE * DT:.1f} s, full-path time: {n_poses * STEPS_PER_POSE * DT:.0f} s")

# ============================================================================
# PULSE spray weights (top surface only)
# ============================================================================
print("PULSE diagnostic - single centered pose test...")
test_pos = jnp.array([PLATE_XY_SIZE/2, PLATE_XY_SIZE/2, Z_HEIGHT])
test_rot = jnp.array([1.0, 0.0, 0.0, 0.0])
def compute_weight_for_pose(pos, rot):
    return deposit(
        pos, rot, PULSE_SIGMA, PULSE_A, PULSE_REF_DIST, PULSE_RESOLUTION,
        face_v0, face_v1, face_v2, face_normals, n_surface, PULSE_FOV,
    )
test_weights = compute_weight_for_pose(test_pos, test_rot)
print(f"  center-pose weights: min={float(jnp.min(test_weights)):.3e}  "
      f"max={float(jnp.max(test_weights)):.3e}  "
      f"mean={float(jnp.mean(test_weights)):.3e}  "
      f"n_nonzero={int(jnp.sum(test_weights > 0))}/{n_surface}")

print("Precomputing PULSE spray weights per pose...")
weights_raw = jax.vmap(compute_weight_for_pose)(pose_positions, pose_rotations)
peak_weight = jnp.max(weights_raw)
# PULSE returns non-zero physically-meaningful weights (verified via diagnostic above).
# Model: screen-space Gaussian x Lambert incidence x inverse-square distance falloff.
if float(peak_weight) < 1e-12:
    raise RuntimeError("PULSE returned zero weights - check Z_HEIGHT, PULSE_SIGMA, PULSE_FOV, PULSE_REF_DIST")
spray_masks_surf = jnp.array(weights_raw / peak_weight)
print(f"  peak weight: {float(peak_weight):.4f} -> normalized to 1.0")
print(f"  active poses (any spray hit): {int(jnp.sum(jnp.max(spray_masks_surf, axis=1) > 0.01))} / {n_poses}")

# ============================================================================
# Physics
# ============================================================================
def volume_laplacian(T, neighbors_arr, G_mat):
    """Finite-volume Laplacian on 3D graph.
    L(T)_i = sum_j G_ij * (T_j - T_i), where G_ij = A_ij/(d_ij*V_i).
    Invalid neighbor slots contribute 0 (G stored as 0 there)."""
    def gather_col(nb_col, G_col):
        safe = jnp.maximum(nb_col, 0)
        T_nb = jnp.where(nb_col >= 0, T[safe], T)   # invalid -> T_i so (T_j - T_i) = 0
        return G_col * (T_nb - T)
    contribs = jax.vmap(gather_col, in_axes=1, out_axes=0)(neighbors_arr, G_mat)
    return jnp.sum(contribs, axis=0)

def linear_operator(T, dt, alpha_field, beta_field, neighbors_arr, G_mat):
    return T - dt * alpha_field * volume_laplacian(T, neighbors_arr, G_mat) + dt * beta_field * T

def radiation_flux(T, gamma_field, T_amb_rad):
    return gamma_field * (T**4 - T_amb_rad**4)

def imex_backward_euler_step(T_old, dt, alpha_field,
                             beta_field, T_amb_conv,
                             gamma_field, T_amb_rad,
                             neighbors_arr, G_mat):
    rad = radiation_flux(T_old, gamma_field, T_amb_rad)
    rhs = T_old + dt * beta_field * T_amb_conv - dt * rad
    A   = lambda T: linear_operator(T, dt, alpha_field, beta_field, neighbors_arr, G_mat)
    T_new, _ = jax.scipy.sparse.linalg.cg(A, rhs, x0=T_old, tol=1e-8, maxiter=400)   # tol [-] relative residual, maxiter [-] hard cap
    return T_new

# ============================================================================
# Initial state
# ============================================================================
T_init = jnp.full((n_cells,), T_INIT_K)
T_INIT_MEAN_C = float(T_init.mean()) - 273.15   # [C]  initial mean plate T, for summary printout
areas_surf_jax = jnp.array(areas_surf)

# ============================================================================
# Time step
# ============================================================================
@jax.jit
def step(carry, _):
    T_old, step_idx, valve_target, spray_stage1, spray_actual = carry

    pose_idx        = jnp.minimum(step_idx // STEPS_PER_POSE, n_poses - 1)
    spray_mask_surf = spray_masks_surf[pose_idx]

    # Sprayed-zone average = area-weighted, only on top-layer cells
    T_top = T_old[:n_surface]
    mask_weight = spray_mask_surf * areas_surf_jax
    mask_wsum   = jnp.sum(mask_weight) + 1e-12    # [m^2] area-like; +eps guards against 0/0 when no spray
    T_zone      = jnp.sum(T_top * mask_weight) / mask_wsum

    # Plate volume-average
    T_plate_avg = jnp.sum(T_old * cell_volume) / jnp.sum(cell_volume)

    # Adaptive hysteresis with RELATIVE thresholds (fixes "stuck-on" bug near ambient)
    # Headroom = how much the plate is above ambient; both thresholds scale with it
    headroom = jnp.maximum(T_plate_avg - T_AMBIENT_K, 1.0)   # [K] driving T-difference; floor at 1 K to avoid zero thresholds
    delta_off = DELTA_OFF_FRAC * headroom
    delta_on  = DELTA_ON_FRAC  * headroom
    has_zone = (mask_wsum > 1e-6).astype(jnp.float64)   # [-] flag: is there any active sprayed area this step?
    turn_off = valve_target         * (T_zone < T_plate_avg - delta_off).astype(jnp.float64) * has_zone
    turn_on  = (1.0 - valve_target) * (T_zone > T_plate_avg - delta_on ).astype(jnp.float64) * has_zone
    valve_target_new = valve_target - turn_off + turn_on

    # S-curve valve
    alpha_ramp = DT / (SPRAY_RAMP_TAU_S + DT)
    spray_stage1_new = spray_stage1 + alpha_ramp * (valve_target_new - spray_stage1)
    spray_actual_new = spray_actual + alpha_ramp * (spray_stage1_new - spray_actual)
    s = jnp.clip(spray_actual_new, 0.0, 1.0)
    spray_smooth = s * s * (3.0 - 2.0 * s)

    # Material properties per cell (lagged at T_old)
    k_field   = k_of_T(T_old)
    cp_field  = cp_of_T(T_old)
    eps_field = eps_of_T(T_old)
    rhocp     = RHO * cp_field
    alpha_fld = k_field / rhocp

    # Boiling curve h at each top-cell's TRUE T (not depth-avg -- fixes Bi>0.1)
    h_top_spray_field = h_spray_of_T(T_old)

    # Extend surface spray mask across all cells (nonzero only on top layer)
    spray_mask_full = jnp.concatenate([spray_mask_surf, jnp.zeros(n_cells - n_surface)])

    # Per-face h on top: spray-weighted + ambient background
    h_top_face = (spray_mask_full * spray_smooth * (h_top_spray_field - H_AMBIENT) + H_AMBIENT) \
                 * is_top_jax
    beta_top    = h_top_face  * face_area_top_jax    / (rhocp * cell_volume)
    beta_bottom = H_AMBIENT   * face_area_bottom_jax / (rhocp * cell_volume)
    beta_side   = H_AMBIENT   * face_area_side_jax   / (rhocp * cell_volume)
    beta_field  = beta_top + beta_bottom + beta_side

    A_rad = face_area_top_jax + face_area_bottom_jax + face_area_side_jax
    gamma_fld = eps_field * SIGMA_SB * A_rad / (rhocp * cell_volume)

    T_new = imex_backward_euler_step(
        T_old, DT, alpha_fld,
        beta_field, T_AMBIENT_K,
        gamma_fld, T_AMBIENT_K,
        neighbors_3d_jax, G_matrix_jax,
    )

    # Return only top-layer temps + summary scalars per step (memory efficient)
    T_top_layer = T_new[:n_surface]
    T_bottom_layer = T_new[(N_LAYERS-1)*n_surface:]
    plate_avg = jnp.sum(T_new * cell_volume) / jnp.sum(cell_volume)
    grad_through = jnp.mean(T_bottom_layer) - jnp.mean(T_top_layer)
    return (T_new, step_idx + 1, valve_target_new, spray_stage1_new, spray_actual_new), \
           (T_new, T_top_layer, T_bottom_layer, plate_avg, pose_idx, spray_actual_new, grad_through)

# ============================================================================
# Run
# ============================================================================
print("Running simulation...")
print(f"  N_LAYERS={N_LAYERS}, layer thicknesses (mm): {(LAYER_THICKS*1000).round(2).tolist()}")
print(f"  alpha (ref) = {ALPHA_REF:.3e} m^2/s")
print(f"  h_spray: 900C={float(h_spray_of_T(T_INIT_K)):.0f}  "
      f"300C={float(h_spray_of_T(jnp.array(573.15))):.0f}  "
      f"150C={float(h_spray_of_T(jnp.array(423.15))):.0f}")

init_carry = (T_init, jnp.int32(0), jnp.float64(1.0), jnp.float64(1.0), jnp.float64(1.0))
_, (T_full_history_K, T_top_history_K, T_bot_history_K, plate_avg_history_K, pose_idx_history, spray_history, grad_history_K) = \
    jax.lax.scan(step, init_carry, None, length=MAX_STEPS)
grad_history_C = np.array(grad_history_K)
# Subsample full-cell history for animation (memory: keep every Nth frame)
SUBSAMPLE = 20      # [-]  animation frame subsampling stride (store every Nth step for memory)
T_full_sub_C = np.array(T_full_history_K[::SUBSAMPLE]) - 273.15
pose_idx_sub = np.array(pose_idx_history[::SUBSAMPLE])
print(f"  Full-history subsampled: {T_full_sub_C.shape[0]} frames (every {SUBSAMPLE} steps = {SUBSAMPLE*DT}s)")
T_top_history_C   = np.array(T_top_history_K) - 273.15
T_bot_history_C   = np.array(T_bot_history_K) - 273.15
plate_avg_C_all   = np.array(plate_avg_history_K) - 273.15
pose_idx_history  = np.array(pose_idx_history)

# ============================================================================
# Summary
# ============================================================================
cool_idx = np.argmax(plate_avg_C_all < T_COOL_ENOUGH_C)
top_layer_final    = T_top_history_C[-1]
bottom_layer_final = T_bot_history_C[-1]

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"  Initial avg temp:      {T_INIT_MEAN_C:.1f} C")
print(f"  Final avg temp:        {plate_avg_C_all[-1]:.1f} C   (delta {T_INIT_MEAN_C - plate_avg_C_all[-1]:.1f} C)")
print(f"  Final top peak (max):  {top_layer_final.max():.1f} C")
print(f"  Final top min:         {top_layer_final.min():.1f} C")
print(f"  Final top spread:      {top_layer_final.max() - top_layer_final.min():.1f} C")
print(f"  ---")
print(f"  Top layer  final:  avg {top_layer_final.mean():.1f} C   min {top_layer_final.min():.1f} C   max {top_layer_final.max():.1f} C")
print(f"  Bottom layer final: avg {bottom_layer_final.mean():.1f} C  min {bottom_layer_final.min():.1f} C  max {bottom_layer_final.max():.1f} C")
print(f"  Final through-thickness gradient: {bottom_layer_final.mean() - top_layer_final.mean():.1f} C")
peak_grad_idx = np.argmax(grad_history_C)
print(f"  Peak through-thickness gradient during sim: {grad_history_C[peak_grad_idx]:.1f} C  (at t = {peak_grad_idx * DT:.1f} s)")
spray_arr = np.array(spray_history)
n_on  = int(np.sum(spray_arr > 0.5))
n_off = int(np.sum(spray_arr <= 0.5))
print(f"  Spray ON steps:  {n_on}  ({100*n_on/MAX_STEPS:.1f}% of simulation)")
print(f"  Spray OFF steps: {n_off} ({100*n_off/MAX_STEPS:.1f}% of simulation)")
# Count transitions to prove pulsing happens throughout
transitions = int(np.sum(np.abs(np.diff((spray_arr > 0.5).astype(np.int32)))))
print(f"  Spray on/off transitions: {transitions}  (higher = more pulsing across cool-down)")
print(f"  ---")
if cool_idx == 0 and plate_avg_C_all[-1] >= T_COOL_ENOUGH_C:
    print(f"  Plate did not reach {T_COOL_ENOUGH_C} C avg within {T_MAX} s")
else:
    print(f"  Time to reach {T_COOL_ENOUGH_C} C avg: {cool_idx * DT:.1f} s")
print("="*70 + "\n")

# ============================================================================
# Visualization
# ============================================================================
print("Building 3D prism volume mesh for visualization...")
# Extrude surface vertices downward through N_LAYERS to build 3D prism grid
z_offsets = np.zeros(N_LAYERS + 1, dtype=np.float64)
for i in range(N_LAYERS):
    z_offsets[i+1] = z_offsets[i] - LAYER_THICKS[i]

n_verts_surf = len(verts_np)
verts_3d = np.zeros((n_verts_surf * (N_LAYERS + 1), 3), dtype=np.float64)
for lvl in range(N_LAYERS + 1):
    verts_3d[lvl*n_verts_surf:(lvl+1)*n_verts_surf, 0:2] = verts_np[:, 0:2]
    verts_3d[lvl*n_verts_surf:(lvl+1)*n_verts_surf, 2]   = verts_np[:, 2] + z_offsets[lvl]

# VTK_WEDGE (=13) cells
cells_vtk = []
for layer in range(N_LAYERS):
    for si in range(n_surface):
        top_v = faces_np[si] + layer * n_verts_surf
        bot_v = faces_np[si] + (layer + 1) * n_verts_surf
        cells_vtk.extend([6, top_v[0], top_v[1], top_v[2], bot_v[0], bot_v[1], bot_v[2]])
cells_vtk = np.array(cells_vtk, dtype=np.int64)
cell_types = np.full(n_cells, 13, dtype=np.uint8)
vol_grid = pv.UnstructuredGrid(cells_vtk, cell_types, verts_3d)
vol_grid.cell_data['temperature'] = T_full_sub_C[0]

print("Animating full 3D volume (ctrl+C to exit)...")
plotter = pv.Plotter(off_screen=False)
plotter.set_background('white')

actor = plotter.add_mesh(
    vol_grid, scalars='temperature', cmap='inferno',
    show_edges=True, edge_color='black', line_width=0.3, clim=[T_AMBIENT_C, T_INIT_C],
    scalar_bar_args={
        'title': 'Temperature (C)',
        'color': 'black',
        'title_font_size': 14,
        'label_font_size': 12,
    },
)

nozzle_point = pv.PolyData(np.array(
    [[poses[0].position[0], poses[0].position[1], poses[0].position[2]]], dtype=np.float32))
plotter.add_mesh(nozzle_point, color='blue', point_size=20, render_points_as_spheres=True)

plotter.camera_position = 'iso'
plotter.show(auto_close=False, interactive_update=True)

n_frames = T_full_sub_C.shape[0]
while True:
    for frame in range(n_frames):
        vol_grid.cell_data['temperature'] = T_full_sub_C[frame]
        actor.mapper.dataset.cell_data['temperature'] = T_full_sub_C[frame]
        actor.mapper.dataset.Modified()
        pidx = int(pose_idx_sub[frame])
        nozzle_point.points = np.array(
            [[poses[pidx].position[0], poses[pidx].position[1], poses[pidx].position[2]]], dtype=np.float32)
        plotter.render()
        time.sleep(0.08)
