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
#

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


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "flat_plate.toml")
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "artifacts", "generated")
os.makedirs(RUNTIME_DIR, exist_ok=True)

cfg = load_config(CONFIG_PATH, project_root=PROJECT_ROOT)


#parameters
T_initial     = cfg.thermal.initial_temperature_c
k             = cfg.thermal.thermal_conductivity_w_mk
T_ambient     = cfg.thermal.ambient_temperature_c

# PHYSICS CHANGE:
# OLD:
#   h_spray_scale = 200000.0 was used with the old graph-Laplacian sink:
#       dTdt = alpha*laplacian - (h/rho_c)*(T - T_ambient)
#
# NEW:
#   h_spray_scale is now treated as a surface convection coefficient [W/m^2-K].
#   The new heat equation divides cooling by rho*c*PLATE_THICKNESS:
#       dTdt_spray = -h*(T - T_ambient)/(rho*c*PLATE_THICKNESS)
#
# Starting with 2000 W/m^2-K keeps the cooling aggressive but more defensible
# than the previous dimensionally inconsistent 200000 value.
h_spray_scale = cfg.thermal.spray_h_scale_w_m2k

h_ambient     = cfg.thermal.ambient_h_w_m2k
c             = cfg.thermal.specific_heat_j_kgk
rho           = cfg.thermal.density_kg_m3
rho_c         = rho * c
alpha         = k / rho_c

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
PLATE_THICKNESS = cfg.thermal.thickness_m
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


#precompute h_fields — nozzle points straight down
print("Precomputing spray distributions...")
rot = jnp.array([1.0, 0.0, 0.0, 0.0])                # 180° about x -> point -Z

pose_positions = jnp.array(np.array(waypoints))
pose_rotations = jnp.tile(rot, (n_waypoints, 1))

def compute_h_for_pose(pos, rot):
    weight = deposit(
        pos, rot, sigma, a, ref_dist, resolution,
        face_v0, face_v1, face_v2, face_normals, n_faces, fov
    )
    return h_ambient + weight * h_spray_scale

h_fields = jax.vmap(compute_h_for_pose)(pose_positions, pose_rotations)
h_fields = jnp.array(h_fields)
print("Done precomputing.")



#geometry-weighted thermal operator
print("Building geometry-weighted thermal operator...")

# PHYSICS CHANGE:
# OLD:
#   We previously used a graph Laplacian:
#       laplacian = mean_neighbor_temperature - T
#       dTdt = alpha*laplacian - (h/rho_c)*(T - T_ambient)
#
#   That was useful for a POC, but it ignored triangle area, shared-edge length,
#   centroid spacing, plate thickness, and face thermal mass.
#
# NEW:
#   We now use a finite-volume / FEM-style thermal surface operator:
#       thermal_mass_i = rho*c*thickness*area_i
#       G_ij = k*thickness*shared_edge_length / center_distance
#       dTdt_cond_i = sum_j G_ij*(T_j - T_i) / thermal_mass_i
#
#   Spray cooling is now treated as a surface Neumann convection sink:
#       dTdt_spray_i = -h_i*(T_i - T_ambient)/(rho*c*thickness)
#
#   This is the thermal-only piece we actually need from Josh's FEM direction:
#   transient heat storage + geometry-aware conduction + surface convection.


# RESULT COMPARISON:
# OLD GRAPH-LAPLACIAN PHYSICS TERMINAL RESULT:
#   Building face adjacency...
#   Adjacency built.
#
#   Final results:
#     Peak temp:     308.2 C
#     Min temp:       37.4 C
#     Temp spread:   270.8 C
#     Avg temp:       68.5 C
#
# NEW GEOMETRY-WEIGHTED THERMAL OPERATOR RESULT:
#   Building geometry-weighted thermal operator...
#   Geometry-weighted thermal operator built.
#     face area: min=9.766e-06, max=9.766e-06
#     conductance: min=7.500e-01, max=1.500e+00
#
#   Final results:
#     Peak temp:      94.1 C
#     Min temp:       50.5 C
#     Temp spread:    43.5 C
#     Avg temp:       71.3 C
#
# INTERPRETATION:
#   The average temperature stayed in the same general range, but the peak
#   temperature and spread dropped dramatically. This means the new operator is
#   not simply "overcooling" the whole plate. Instead, it is redistributing heat
#   through geometry-aware conduction so corners and boundary regions no longer
#   remain unrealistically hot.
#
#   Old model: useful POC, but graph-based and dimensionally weak.
#   New model: thermal-only FEM-style bridge using thermal mass, conductance,
#   plate thickness, and surface convection from the robotic spray field.

mesh_pv = surface_mesh.polydata
points_np = surface_mesh.points
faces_np = surface_mesh.faces

fv_face_centers_np = surface_mesh.face_centers
fv_face_area_np = surface_mesh.face_areas

edge_to_faces = defaultdict(list)
edge_to_length = {}

for fi, face in enumerate(faces_np):
    for j in range(3):
        a_idx = face[j]
        b_idx = face[(j + 1) % 3]
        edge = tuple(sorted([a_idx, b_idx]))
        edge_to_faces[edge].append(fi)

        pa = points_np[a_idx]
        pb = points_np[b_idx]
        edge_to_length[edge] = np.linalg.norm(pb - pa)

max_neighbors = 3
neighbors = -1 * np.ones((n_faces, max_neighbors), dtype=np.int32)
conductance = np.zeros((n_faces, max_neighbors), dtype=np.float32)

for edge, fs in edge_to_faces.items():
    if len(fs) != 2:
        # Boundary edge: no neighbor across this edge.
        # Boundary/surface cooling is handled through h_field below.
        continue

    f0, f1 = fs
    edge_len = edge_to_length[edge]

    c0 = fv_face_centers_np[f0]
    c1 = fv_face_centers_np[f1]
    center_dist = max(np.linalg.norm(c1 - c0), 1e-12)

    # Conductance between neighboring triangular control volumes.
    G = k * PLATE_THICKNESS * edge_len / center_dist

    for a_face, b_face in [(f0, f1), (f1, f0)]:
        open_slots = np.where(neighbors[a_face] < 0)[0]
        if len(open_slots) == 0:
            continue

        slot = open_slots[0]
        neighbors[a_face, slot] = b_face
        conductance[a_face, slot] = G

neighbors_jax = jnp.array(neighbors)
conductance_jax = jnp.array(conductance)
face_area_jax = jnp.array(fv_face_area_np)
thermal_mass_jax = rho_c * PLATE_THICKNESS * face_area_jax

positive_G = conductance[conductance > 0]
print("Geometry-weighted thermal operator built.")
print(f"  face area: min={fv_face_area_np.min():.3e}, max={fv_face_area_np.max():.3e}")
print(f"  conductance: min={positive_G.min():.3e}, max={positive_G.max():.3e}")


#JAX heat step + rollout
T_init = jnp.full((n_faces,), T_initial)

@jax.jit
def step(carry, _):
    T, step_idx = carry
    pose_idx = jnp.minimum(step_idx // steps_per_move, n_waypoints - 1)
    h_field  = h_fields[pose_idx]

    safe_neighbors = jnp.where(neighbors_jax >= 0, neighbors_jax, 0)
    T_neighbors = T[safe_neighbors]
    valid = (neighbors_jax >= 0).astype(jnp.float32)

    conductive_power = jnp.sum(
        conductance_jax * valid * (T_neighbors - T[:, None]),
        axis=1
    )

    dTdt_conduction = conductive_power / thermal_mass_jax
    dTdt_spray = -(h_field / (rho_c * PLATE_THICKNESS)) * (T - T_ambient)

    T_new = T + (dTdt_conduction + dTdt_spray) * dt_sim
    T_new = jnp.maximum(T_new, T_ambient)

    return (T_new, step_idx + 1), (T_new, pose_idx)

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
def world_to_base(pos_world):
    # Robot base is now the origin, so world frame == robot base frame
    return np.array(pos_world)

_last_joint_state = np.zeros(len(chain.links))

def set_initial_joint(name, value):
    """Set initial joint value by joint name, not by fragile hardcoded index."""
    global _last_joint_state
    for idx, link in enumerate(chain.links):
        if link.name == name:
            _last_joint_state[idx] = value
            return
    print(f"WARNING: joint {name} not found in ikpy chain")

# arm extended forward and slightly down — closer to typical spray poses
set_initial_joint("shoulder_pan_joint", 0.0)
set_initial_joint("shoulder_lift_joint", -1.5708)
set_initial_joint("elbow_joint", 1.5708)
set_initial_joint("wrist_1_joint", -1.5708)
set_initial_joint("wrist_2_joint", -1.5708)
set_initial_joint("wrist_3_joint", 0.0)








# Convert JAX mesh arrays to NumPy once for geometric surface-normal lookup.
# This generalizes beyond a flat plate: nearest triangle gives local surface normal.
face_v0_np = np.array(face_v0)
face_v1_np = np.array(face_v1)
face_v2_np = np.array(face_v2)
face_centers_np = (face_v0_np + face_v1_np + face_v2_np) / 3.0
face_normals_np = np.array(face_normals)

def nearest_surface_point_and_normal(nozzle_world):
    """
    Return nearest mesh face center and outward normal relative to nozzle_world.

    For the flat plate:
      surface point ~= [x, y, 0]
      normal ~= [0, 0, 1]

    For future curved geometry:
      normal becomes the local triangle normal, flipped so it points toward the nozzle.
    """
    nozzle_world = np.array(nozzle_world, dtype=float)

    # Nearest face center to current nozzle/target position.
    diff = face_centers_np - nozzle_world[None, :]
    idx = int(np.argmin(np.sum(diff * diff, axis=1)))

    p_surf = face_centers_np[idx].copy()
    n = face_normals_np[idx].copy()
    n = n / (np.linalg.norm(n) + 1e-8)

    # Flip normal so it points from surface toward nozzle.
    # This is crucial for spheres / irregular surfaces later.
    to_nozzle = nozzle_world - p_surf
    if np.dot(n, to_nozzle) < 0.0:
        n = -n

    return p_surf, n


def unwrap_to_reference(q_candidate, q_reference):
    """
    Keep IK continuous by choosing equivalent joint angles closest to previous pose.
    This prevents +/-2pi branch jumps that look like vibration.
    """
    q_candidate = np.array(q_candidate, dtype=float)
    q_reference = np.array(q_reference, dtype=float)
    delta = q_candidate - q_reference
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    return q_reference + delta


def solve_ik_direct(target_world):
    """
    Direct orientation-aware IK initialization.

    Tool local +Z is aligned with the local surface normal.
    Since the spray is drawn from tool0 toward the surface, this makes the
    end effector normal to the part instead of sideways/upside-down.
    """
    target_world = np.array(target_world, dtype=float)
    target_base = world_to_base(target_world)

    _, n_world = nearest_surface_point_and_normal(target_world)

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

    _, n_world = nearest_surface_point_and_normal(target_world)

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


def smooth_joint_path(q_path, passes=3):
    """
    Post-process cached IK trajectory to suppress tiny rapid vibrations.

    This is a visual trajectory smoother, similar in spirit to robot motion planning:
    don't execute raw noisy inverse-kinematics frame outputs directly.
    """
    q_path = np.array(q_path, dtype=float)

    # Joint-specific max step per rendered frame.
    max_steps = np.full(q_path.shape[1], 0.040)
    for idx, link in enumerate(chain.links):
        if link.name in ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"):
            max_steps[idx] = 0.022
        elif link.name in ("wrist_1_joint", "wrist_2_joint", "wrist_3_joint"):
            max_steps[idx] = 0.026

    q = q_path.copy()

    for _ in range(passes):
        # Forward velocity limit
        for i in range(1, len(q)):
            dq = q[i] - q[i - 1]
            dq = np.clip(dq, -max_steps, max_steps)
            q[i] = q[i - 1] + dq

        # Backward velocity limit
        for i in range(len(q) - 2, -1, -1):
            dq = q[i] - q[i + 1]
            dq = np.clip(dq, -max_steps, max_steps)
            q[i] = q[i + 1] + dq

        # Small moving-average pass for acceleration smoothing.
        q2 = q.copy()
        for i in range(1, len(q) - 1):
            q2[i] = 0.25 * q[i - 1] + 0.50 * q[i] + 0.25 * q[i + 1]
        q = q2

    return q

# UR5e mesh actors — parse URDF XML directly for link->mesh mapping
print("Building UR5e mesh actors...")


tree = ET.parse(urdf_file)
root = tree.getroot()

# URDF <mesh filename="..."/> paths use package:// prefixes; yourdfpy resolves
# these when it loads the scene, so we use its resolved geometry names.
# But we need a direct link->file mapping, so we parse XML and resolve paths
# ourselves using the URDF file directory as the resolution root.
urdf_dir = os.path.dirname(urdf_file)

def resolve_mesh_path(filename_attr):
    """Convert URDF mesh filename to an absolute path on disk."""
    if filename_attr.startswith("package://"):
        # strip package://<pkg>/ and look under the ur_description repo
        stripped = filename_attr[len("package://"):]
        # first path segment is the package name
        parts = stripped.split("/", 1)
        if len(parts) == 2:
            # ur5e_description module has REPOSITORY_PATH — try that root
            from robot_descriptions import ur5e_description as ur_mod
            candidate = os.path.join(ur_mod.REPOSITORY_PATH, parts[1])
            if os.path.exists(candidate):
                return candidate
            # fallback: search under REPOSITORY_PATH recursively
            target_name = os.path.basename(parts[1])
            for r, _, files in os.walk(ur_mod.REPOSITORY_PATH):
                if target_name in files:
                    return os.path.join(r, target_name)
    elif filename_attr.startswith("file://"):
        return filename_attr[len("file://"):]
    else:
        # relative path — try relative to URDF file dir
        candidate = os.path.join(urdf_dir, filename_attr)
        if os.path.exists(candidate):
            return candidate
    return None

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
        mesh_file = resolve_mesh_path(mesh_elem.get("filename"))
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

def set_actor_matrix(actor, M):
    """Apply a 4x4 numpy matrix to a vtk actor."""
    vtk_m = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_m.SetElement(i, j, M[i, j])
    actor.SetUserMatrix(vtk_m)

# scene setup
print("Setting up visualization...")
print("=" * 72)
print("PYVISTA SHUTDOWN: return to the terminal and press Ctrl+C.")
print("DO NOT close the PyVista window with its X button.")
print("=" * 72)

mesh_vis = pv.read(tmp_mesh_path).triangulate()
mesh_vis.cell_data['temperature'] = T_history[0]

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

# plate heatmap
heatmap_actor = plotter.add_mesh(
    mesh_vis, scalars='temperature', cmap='inferno',
    show_edges=False, clim=[T_ambient, T_initial],
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

def q_to_cfg(q):
    """
    Convert ikpy q vector into a yourdfpy config dictionary.

    Do this by matching joint names, not by assuming q[i + 1].
    This prevents ikpy's OriginLink or fixed links from shifting the UR5e joints.
    """
    cfg = {}
    for idx, link in enumerate(chain.links):
        if link.name in UR5E_JOINT_NAMES:
            cfg[link.name] = float(q[idx])
    return cfg

def update_ur5e_pose(q):
    """Set URDF config, then push each link's world transform to its actors."""
    cfg = q_to_cfg(q)
    urdf.update_cfg(cfg)

    for link_name, actors in link_actors.items():
        T_link = np.array(urdf.get_transform(link_name))
        T_world = robot_base_T @ T_link
        for actor in actors:
            set_actor_matrix(actor, T_world)

def get_tool0_transform_world(q):
    """Return full tool0 transform in world coordinates."""
    cfg = q_to_cfg(q)
    urdf.update_cfg(cfg)

    T_tool0 = np.array(urdf.get_transform("tool0"))
    return robot_base_T @ T_tool0


def get_tool0_world(q):
    """Return tool0 position in world coordinates."""
    return get_tool0_transform_world(q)[:3, 3]


def get_nozzle_tip_world(q):
    """
    Return physical nozzle exit position.

    The visual nozzle points along local -Z from tool0, so the tip is
    NOZZLE_LENGTH below tool0 in the tool0 local frame.
    """
    T_tool0_world = get_tool0_transform_world(q)
    tip_local = np.array([0.0, 0.0, -NOZZLE_LENGTH, 1.0])
    return (T_tool0_world @ tip_local)[:3]


def get_nozzle_direction_world(q):
    """
    Return spray direction in world coordinates.

    The nozzle sprays along local -Z of tool0.
    """
    T_tool0_world = get_tool0_transform_world(q)
    local_minus_z = np.array([0.0, 0.0, -1.0])
    d = T_tool0_world[:3, :3] @ local_minus_z
    return d / (np.linalg.norm(d) + 1e-8)







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

def set_spray_head_position(pos_world):
    M = np.eye(4)
    M[:3, 3] = np.array(pos_world)
    set_actor_matrix(spray_head_actor, M)

# initial pose + pressure jet actor
q_init = solve_ik_direct(np.array(waypoints[0]))
_last_joint_state = q_init.copy()
update_ur5e_pose(q_init)

tool0_init = get_tool0_world(q_init)
impact_init = np.array(waypoints[0], dtype=float)

set_spray_head_position(tool0_init)

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

# Use face centers from the geometry-weighted thermal operator.
plate_pick_centers = fv_face_centers_np

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

    on_plate_z = abs(p_click[2] - plate_z_ref) <= plate_z_tol
    inside_x = (plate_x_min - plate_xy_pad) <= p_click[0] <= (plate_x_max + plate_xy_pad)
    inside_y = (plate_y_min - plate_xy_pad) <= p_click[1] <= (plate_y_max + plate_xy_pad)

    if not (on_plate_z and inside_x and inside_y):
        print(
            "Ignored click not on plate: "
            f"x={p_click[0]:.3f}, y={p_click[1]:.3f}, z={p_click[2]:.3f}"
        )
        return

    dists = np.linalg.norm(plate_pick_centers - p_click[None, :], axis=1)
    face_idx = int(np.argmin(dists))
    center = plate_pick_centers[face_idx]

    if dists[face_idx] > 0.055:
        print(f"Ignored click too far from plate face: distance={dists[face_idx]:.3f} m")
        return

    # skip if we already picked this exact face
    if face_idx in picked_face_ids:
        print(f"Face {face_idx} already plotted, skipping.")
        return
    
    # skip if we already picked this exact face
    if face_idx in picked_face_ids:
        print(f"Face {face_idx} already plotted, skipping.")
        return

    temps = T_history[:, face_idx]
    # Convert world-frame point to robot base frame for meaningful robotics coordinates
    center_base = center - ROBOT_BASE
    label = f"({center_base[0]*1000:.0f}, {center_base[1]*1000:.0f}, {center_base[2]*1000:.0f}) mm from base"

    history_curves.append(temps)
    history_labels.append(label)
    picked_face_ids.append(face_idx)

    refresh_history_panel()

    # Add marker on plate.
    plotter.subplot(0, 0)
    marker_center = center.copy()
    marker_center[2] = plate_z_ref + 0.006

    marker_sphere = pv.Sphere(radius=0.006, center=marker_center)
    marker_actor = plotter.add_mesh(marker_sphere, name=f"picked_face_{face_idx}")
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

FRAME_TIME = 0.022

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
q_anim_path = smooth_joint_path(q_anim_path, passes=4)

print(f"Animation frames: {len(frame_list)}")
print("Done precomputing normal-aligned animation path.")

# Turn spray on only after trajectory planning is complete.
spray_actor.SetVisibility(True)
spray_head_actor.SetVisibility(True)

# set first pose before display loop starts
q_first = q_anim_path[0]
update_ur5e_pose(q_first)

tool0_first = get_tool0_world(q_first)
set_spray_head_position(tool0_first)

# ------------------------------------------------------------------
# Replay cached trajectory.
# ------------------------------------------------------------------
while True:
    for j, frame in enumerate(frame_list):
        t_start = time.time()
        sim_time = frame * dt_sim

        target = visual_targets[j]
        q = q_anim_path[j]

        # plate temperature still comes from precomputed thermal simulation
        mesh_vis.cell_data['temperature'] = T_history[frame]
        heatmap_actor.mapper.dataset.cell_data['temperature'] = T_history[frame]
        heatmap_actor.mapper.dataset.Modified()

        # UR5e pose update from cached joint path
        update_ur5e_pose(q)

        # Tool0 should now be approximately normal to local mesh surface.
        tool0_world = get_tool0_world(q)
        jet_origin = tool0_world.copy()

        # Impact point comes from nearest mesh surface, not hardcoded plate z.
        # This is what will matter on sphere / irregular 3D shapes.
        impact_center, surf_normal = nearest_surface_point_and_normal(target)

        set_spray_head_position(jet_origin)

        if j == 0:
            err_world = np.linalg.norm(tool0_world - target)
            alignment = np.dot(get_nozzle_direction_world(q), -surf_normal)
            print(f"Start-of-pass tool0 tracking error: {err_world:.3f} m")
            print(f"Start-of-pass spray-normal alignment: {alignment:.3f}")

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