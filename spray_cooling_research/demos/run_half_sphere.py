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
from spray_cooling.planning.hemisphere import (
    build_hemisphere_path,
)

from spray_cooling.geometry.surface_mesh import load_surface_mesh
import hashlib
from scipy.spatial.transform import Rotation as SciRotation


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "half_sphere.toml")
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

# Build the JAX-PULSE triangle arrays directly from the already-loaded
# PyVista mesh. Pulse.load_mesh() misread the resaved OBJ as one face.
vtk_faces = np.asarray(raw_mesh.faces, dtype=np.int64)

if vtk_faces.size % 4 != 0:
    raise RuntimeError(
        "Unexpected PyVista face-array length: "
        f"{vtk_faces.size}"
    )

vtk_faces = vtk_faces.reshape(-1, 4)

if not np.all(vtk_faces[:, 0] == 3):
    raise RuntimeError(
        "The half-sphere mesh contains non-triangular cells."
    )

triangle_indices = vtk_faces[:, 1:4]
mesh_points_np = np.asarray(raw_mesh.points, dtype=np.float64)

p0_np = mesh_points_np[triangle_indices[:, 0]]
p1_np = mesh_points_np[triangle_indices[:, 1]]
p2_np = mesh_points_np[triangle_indices[:, 2]]

direct_face_centers = (p0_np + p1_np + p2_np) / 3.0

direct_face_normals = np.cross(
    p1_np - p0_np,
    p2_np - p0_np,
)

normal_magnitudes = np.linalg.norm(
    direct_face_normals,
    axis=1,
    keepdims=True,
)

if np.any(normal_magnitudes[:, 0] < 1.0e-14):
    raise RuntimeError(
        "The half-sphere contains a degenerate triangle."
    )

direct_face_normals /= normal_magnitudes

# Correct orientation:
# dome normals point radially outward;
# flat-base normals point upward.
mesh_bounds_now = raw_mesh.bounds

orientation_center = np.array(
    [
        0.5 * (mesh_bounds_now[0] + mesh_bounds_now[1]),
        0.5 * (mesh_bounds_now[2] + mesh_bounds_now[3]),
        mesh_bounds_now[4],
    ],
    dtype=float,
)

base_tolerance = max(
    1.0e-8,
    1.0e-5 * (mesh_bounds_now[5] - mesh_bounds_now[4]),
)

dome_faces = (
    direct_face_centers[:, 2]
    > mesh_bounds_now[4] + base_tolerance
)

radial_vectors = (
    direct_face_centers
    - orientation_center
)

wrong_dome_normals = (
    dome_faces
    & (
        np.sum(
            direct_face_normals * radial_vectors,
            axis=1,
        )
        < 0.0
    )
)

direct_face_normals[wrong_dome_normals] *= -1.0

wrong_base_normals = (
    (~dome_faces)
    & (direct_face_normals[:, 2] < 0.0)
)

direct_face_normals[wrong_base_normals] *= -1.0

face_v0 = jnp.asarray(p0_np, dtype=jnp.float32)
face_v1 = jnp.asarray(p1_np, dtype=jnp.float32)
face_v2 = jnp.asarray(p2_np, dtype=jnp.float32)

face_normals = jnp.asarray(
    direct_face_normals,
    dtype=jnp.float32,
)

n_faces = int(triangle_indices.shape[0])

if n_faces != raw_mesh.n_cells:
    raise RuntimeError(
        f"Triangle mismatch: arrays={n_faces}, "
        f"PyVista cells={raw_mesh.n_cells}"
    )

if n_faces != 2780:
    raise RuntimeError(
        f"Expected 2780 half-sphere faces, found {n_faces}"
    )

print(f"Mesh loaded. Faces: {n_faces}")
print(f"Part bounds: x [{raw_mesh.bounds[0]:.3f}, {raw_mesh.bounds[1]:.3f}]  "
      f"y [{raw_mesh.bounds[2]:.3f}, {raw_mesh.bounds[3]:.3f}]")


# Curved spray path over the hemispherical dome.
#
# The mesh remains a surface mesh. PLATE_THICKNESS supplies the
# effective shell thickness used by the thermal operator.
mesh_bounds = raw_mesh.bounds

mesh_x_min = float(mesh_bounds[0])
mesh_x_max = float(mesh_bounds[1])
mesh_y_min = float(mesh_bounds[2])
mesh_y_max = float(mesh_bounds[3])
mesh_z_min = float(mesh_bounds[4])
mesh_z_max = float(mesh_bounds[5])

SPHERE_CENTER = np.array(
    [
        0.5 * (mesh_x_min + mesh_x_max),
        0.5 * (mesh_y_min + mesh_y_max),
        mesh_z_min,
    ],
    dtype=float,
)

SPHERE_RADIUS = 0.5 * max(
    mesh_x_max - mesh_x_min,
    mesh_y_max - mesh_y_min,
)

NOZZLE_SURFACE_STANDOFF = float(ref_dist)

# PHYSICAL NOZZLE TIP INTEGRATION
#
# JAX-PULSE and the visible water begin at the physical nozzle tip.
# IK controls tool0, which must remain one nozzle length farther outward.
HEMISPHERE_STANDOFF = (
    NOZZLE_SURFACE_STANDOFF
    + NOZZLE_LENGTH
)

hemisphere_path = build_hemisphere_path(
    sphere_center=SPHERE_CENTER,
    sphere_radius=SPHERE_RADIUS,
    ring_count=12,
    points_per_ring=32,
    polar_min_deg=8.0,
    polar_max_deg=50.0,
    standoff=HEMISPHERE_STANDOFF,
)

# FIXED-BASE COLLISION-REDUCTION SECTOR
#
# A stationary UR5e cannot safely reach around the entire back side of the
# dome without link-part collisions. Keep only the upper sector facing the
# robot. Full-surface coverage requires robot repositioning or a turntable.
robot_facing_xy = (
    ROBOT_BASE[:2]
    - SPHERE_CENTER[:2]
)

robot_facing_xy /= (
    np.linalg.norm(robot_facing_xy)
    + 1.0e-12
)

path_normal_xy = (
    hemisphere_path.surface_normals[:, :2]
)

path_normal_xy /= (
    np.linalg.norm(
        path_normal_xy,
        axis=1,
        keepdims=True,
    )
    + 1.0e-12
)

ACCESSIBLE_HALF_ANGLE_DEG = 65.0

facing_alignment = (
    path_normal_xy
    @ robot_facing_xy
)

accessible_mask = (
    facing_alignment
    >= np.cos(
        np.deg2rad(
            ACCESSIBLE_HALF_ANGLE_DEG
        )
    )
)

hemisphere_path = type(hemisphere_path)(
    surface_points=(
        hemisphere_path.surface_points[
            accessible_mask
        ]
    ),
    surface_normals=(
        hemisphere_path.surface_normals[
            accessible_mask
        ]
    ),
    tool_positions=(
        hemisphere_path.tool_positions[
            accessible_mask
        ]
    ),
    pulse_quaternions_xyzw=(
        hemisphere_path.pulse_quaternions_xyzw[
            accessible_mask
        ]
    ),
)

print(
    "Fixed-base accessible spray sector:"
    f" +/-{ACCESSIBLE_HALF_ANGLE_DEG:.1f} deg,"
    f" retained {len(hemisphere_path.tool_positions)} waypoints"
)



# Tool0 targets used by the UR5e inverse-kinematics solver.
# RING-INTERLEAVED INDEXED TURNTABLE
#
# The UR5e repeats one physically accessible world-space sector.
# The part indexes after each small group of rings while spray is disabled.
#
# The thermal mesh remains in material coordinates. For each turntable
# angle, the world-space nozzle pose is transformed back into the rotating
# part's material frame before JAX-PULSE deposition is calculated.
TURNTABLE_PASS_ANGLES_DEG = np.array(
    [0.0, 120.0, 240.0],
    dtype=float,
)

# CONTROLLED RING-INTERLEAVING EXPERIMENT
#
# Preserve the baseline schedule size:
#     396 spray-on poses
#      40 spray-off poses
#     436 total poses
#
# Two rings are cooled at all three indexed orientations before moving to
# the next two-ring group. This changes cooling order without changing the
# total simulated process time or total spray exposure.
RINGS_PER_INTERLEAVED_GROUP = 2
INTER_BLOCK_TRANSITION_WAYPOINTS = 2
FINAL_RETURN_WAYPOINTS = 6

base_tool_positions = np.asarray(
    hemisphere_path.tool_positions,
    dtype=float,
)

base_impact_positions = np.asarray(
    hemisphere_path.surface_points,
    dtype=float,
)

base_surface_normals = np.asarray(
    hemisphere_path.surface_normals,
    dtype=float,
)

base_pulse_quaternions = np.asarray(
    hemisphere_path.pulse_quaternions_xyzw,
    dtype=float,
)

base_nozzle_positions = (
    base_impact_positions
    + NOZZLE_SURFACE_STANDOFF
    * base_surface_normals
)

BASE_PASS_WAYPOINT_COUNT = len(
    base_tool_positions
)

if BASE_PASS_WAYPOINT_COUNT < 2:
    raise RuntimeError(
        "The accessible spray sector contains too few waypoints."
    )


def _rotation_z(angle_rad):
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)

    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _world_point_to_material(point_world, table_angle_rad):
    rotation_inverse = _rotation_z(
        -table_angle_rad
    )

    return (
        SPHERE_CENTER
        + rotation_inverse
        @ (
            np.asarray(point_world, dtype=float)
            - SPHERE_CENTER
        )
    )


def _world_quaternion_to_material(
    quaternion_world,
    table_angle_rad,
):
    rotation_inverse = _rotation_z(
        -table_angle_rad
    )

    rotation_world = (
        SciRotation
        .from_quat(
            np.asarray(
                quaternion_world,
                dtype=float,
            )
        )
        .as_matrix()
    )

    rotation_material = (
        rotation_inverse
        @ rotation_world
    )

    return (
        SciRotation
        .from_matrix(rotation_material)
        .as_quat()
    )


scheduled_tool_positions = []
scheduled_impact_positions = []
scheduled_surface_normals = []

scheduled_material_nozzle_positions = []
scheduled_material_quaternions = []

scheduled_turntable_angles = []
scheduled_spray_enabled = []


def _append_schedule_pose(
    tool_position,
    impact_position,
    surface_normal,
    nozzle_position,
    pulse_quaternion,
    table_angle_rad,
    spray_enabled,
):
    scheduled_tool_positions.append(
        np.asarray(tool_position, dtype=float)
    )

    scheduled_impact_positions.append(
        np.asarray(impact_position, dtype=float)
    )

    scheduled_surface_normals.append(
        np.asarray(surface_normal, dtype=float)
    )

    scheduled_material_nozzle_positions.append(
        _world_point_to_material(
            nozzle_position,
            table_angle_rad,
        )
    )

    scheduled_material_quaternions.append(
        _world_quaternion_to_material(
            pulse_quaternion,
            table_angle_rad,
        )
    )

    scheduled_turntable_angles.append(
        float(table_angle_rad)
    )

    scheduled_spray_enabled.append(
        bool(spray_enabled)
    )


# The accessible mask preserves the original flattened ring ordering.
# Recover contiguous ring boundaries from the constant normal-z value on
# each polar ring.
ring_break_indices = (
    np.flatnonzero(
        np.abs(
            np.diff(
                base_surface_normals[:, 2]
            )
        )
        > 1.0e-10
    )
    + 1
)

base_ring_waypoint_indices = [
    ring_indices.astype(int)
    for ring_indices in np.split(
        np.arange(
            BASE_PASS_WAYPOINT_COUNT,
            dtype=int,
        ),
        ring_break_indices,
    )
]

if len(base_ring_waypoint_indices) < 2:
    raise RuntimeError(
        "Failed to recover multiple hemisphere rings from the "
        "accessible waypoint path."
    )

if any(
    len(ring_indices) == 0
    for ring_indices in base_ring_waypoint_indices
):
    raise RuntimeError(
        "Recovered an empty hemisphere ring."
    )

if sum(
    len(ring_indices)
    for ring_indices in base_ring_waypoint_indices
) != BASE_PASS_WAYPOINT_COUNT:
    raise RuntimeError(
        "Recovered ring waypoint counts do not match the "
        "accessible-sector waypoint count."
    )

ring_waypoint_counts = np.asarray(
    [
        len(ring_indices)
        for ring_indices in base_ring_waypoint_indices
    ],
    dtype=int,
)

ring_waypoint_groups = []

for ring_start in range(
    0,
    len(base_ring_waypoint_indices),
    RINGS_PER_INTERLEAVED_GROUP,
):
    ring_waypoint_groups.append(
        np.concatenate(
            base_ring_waypoint_indices[
                ring_start:
                ring_start
                + RINGS_PER_INTERLEAVED_GROUP
            ]
        )
    )


# Build the spray blocks before adding spray-off transitions.
#
# Even groups:
#     0 -> 120 -> 240 degrees
#
# Odd groups:
#     240 -> 120 -> 0 degrees
#
# This avoids accumulating continuous turntable rotation while still
# exposing every ring group to all three material orientations.
schedule_blocks = []

for ring_group_index, ring_group_indices in enumerate(
    ring_waypoint_groups
):
    if ring_group_index % 2 == 0:
        group_angles_deg = (
            TURNTABLE_PASS_ANGLES_DEG
        )
    else:
        group_angles_deg = (
            TURNTABLE_PASS_ANGLES_DEG[::-1]
        )

    for local_pass_index, pass_angle_deg in enumerate(
        group_angles_deg
    ):
        # Reverse the middle pass so consecutive orientations share the
        # same robot endpoint whenever possible.
        if local_pass_index % 2 == 0:
            order = ring_group_indices.copy()
        else:
            order = ring_group_indices[::-1].copy()

        schedule_blocks.append(
            (
                int(ring_group_index),
                float(pass_angle_deg),
                order,
            )
        )


# Confirm that every accessible waypoint is sprayed exactly once at each
# material orientation.
expected_base_indices = np.arange(
    BASE_PASS_WAYPOINT_COUNT,
    dtype=int,
)

for required_angle_deg in TURNTABLE_PASS_ANGLES_DEG:
    indices_at_angle = np.concatenate(
        [
            order
            for _, block_angle_deg, order
            in schedule_blocks
            if np.isclose(
                block_angle_deg,
                required_angle_deg,
            )
        ]
    )

    if not np.array_equal(
        np.sort(indices_at_angle),
        expected_base_indices,
    ):
        raise RuntimeError(
            "Ring-interleaved schedule does not cover every "
            f"accessible waypoint exactly once at "
            f"{required_angle_deg:.1f} degrees."
        )


def _normalized_linear_interpolation(
    start_vector,
    end_vector,
    fraction,
):
    interpolated = (
        (1.0 - fraction)
        * np.asarray(start_vector, dtype=float)
        + fraction
        * np.asarray(end_vector, dtype=float)
    )

    magnitude = np.linalg.norm(interpolated)

    if magnitude <= 1.0e-12:
        raise RuntimeError(
            "Cannot normalize a near-zero transition vector."
        )

    return interpolated / magnitude


def _interpolate_quaternion(
    start_quaternion,
    end_quaternion,
    fraction,
):
    quaternion_start = np.asarray(
        start_quaternion,
        dtype=float,
    )

    quaternion_end = np.asarray(
        end_quaternion,
        dtype=float,
    )

    # q and -q represent the same rotation. Choose the shorter
    # interpolation direction.
    if np.dot(
        quaternion_start,
        quaternion_end,
    ) < 0.0:
        quaternion_end = -quaternion_end

    interpolated = (
        (1.0 - fraction)
        * quaternion_start
        + fraction
        * quaternion_end
    )

    magnitude = np.linalg.norm(interpolated)

    if magnitude <= 1.0e-12:
        raise RuntimeError(
            "Cannot normalize a near-zero transition quaternion."
        )

    return interpolated / magnitude


for block_index, (
    ring_group_index,
    pass_angle_deg,
    order,
) in enumerate(schedule_blocks):
    pass_angle_rad = np.deg2rad(
        pass_angle_deg
    )

    for waypoint_index in order:
        _append_schedule_pose(
            base_tool_positions[waypoint_index],
            base_impact_positions[waypoint_index],
            base_surface_normals[waypoint_index],
            base_nozzle_positions[waypoint_index],
            base_pulse_quaternions[waypoint_index],
            pass_angle_rad,
            True,
        )

    if block_index >= len(schedule_blocks) - 1:
        continue

    _, next_angle_deg, next_order = (
        schedule_blocks[
            block_index + 1
        ]
    )

    next_angle_rad = np.deg2rad(
        next_angle_deg
    )

    current_endpoint_index = int(
        order[-1]
    )

    next_startpoint_index = int(
        next_order[0]
    )

    for transition_index in range(
        INTER_BLOCK_TRANSITION_WAYPOINTS
    ):
        fraction = (
            transition_index + 1
        ) / (
            INTER_BLOCK_TRANSITION_WAYPOINTS
            + 1
        )

        transition_tool_position = (
            (1.0 - fraction)
            * base_tool_positions[
                current_endpoint_index
            ]
            + fraction
            * base_tool_positions[
                next_startpoint_index
            ]
        )

        transition_impact_position = (
            (1.0 - fraction)
            * base_impact_positions[
                current_endpoint_index
            ]
            + fraction
            * base_impact_positions[
                next_startpoint_index
            ]
        )

        transition_surface_normal = (
            _normalized_linear_interpolation(
                base_surface_normals[
                    current_endpoint_index
                ],
                base_surface_normals[
                    next_startpoint_index
                ],
                fraction,
            )
        )

        transition_nozzle_position = (
            (1.0 - fraction)
            * base_nozzle_positions[
                current_endpoint_index
            ]
            + fraction
            * base_nozzle_positions[
                next_startpoint_index
            ]
        )

        transition_quaternion = (
            _interpolate_quaternion(
                base_pulse_quaternions[
                    current_endpoint_index
                ],
                base_pulse_quaternions[
                    next_startpoint_index
                ],
                fraction,
            )
        )

        transition_angle_rad = (
            (1.0 - fraction)
            * pass_angle_rad
            + fraction
            * next_angle_rad
        )

        _append_schedule_pose(
            transition_tool_position,
            transition_impact_position,
            transition_surface_normal,
            transition_nozzle_position,
            transition_quaternion,
            transition_angle_rad,
            False,
        )


# Return the robot to the first accessible waypoint with spray disabled.
# With an even number of ring groups the table already ends at zero
# degrees; the angle interpolation also handles future odd group counts.
final_block_angle_rad = np.deg2rad(
    schedule_blocks[-1][1]
)

final_endpoint_index = int(
    schedule_blocks[-1][2][-1]
)

return_indices = np.linspace(
    final_endpoint_index,
    0,
    FINAL_RETURN_WAYPOINTS,
).astype(int)

return_angles = np.linspace(
    final_block_angle_rad,
    0.0,
    FINAL_RETURN_WAYPOINTS,
)

for waypoint_index, return_angle in zip(
    return_indices,
    return_angles,
):
    _append_schedule_pose(
        base_tool_positions[waypoint_index],
        base_impact_positions[waypoint_index],
        base_surface_normals[waypoint_index],
        base_nozzle_positions[waypoint_index],
        base_pulse_quaternions[waypoint_index],
        return_angle,
        False,
    )


expected_spray_pose_count = (
    len(TURNTABLE_PASS_ANGLES_DEG)
    * BASE_PASS_WAYPOINT_COUNT
)

expected_spray_off_pose_count = (
    (
        len(schedule_blocks) - 1
    )
    * INTER_BLOCK_TRANSITION_WAYPOINTS
    + FINAL_RETURN_WAYPOINTS
)

if len(scheduled_tool_positions) != (
    expected_spray_pose_count
    + expected_spray_off_pose_count
):
    raise RuntimeError(
        "Unexpected ring-interleaved schedule length."
    )

if np.count_nonzero(
    scheduled_spray_enabled
) != expected_spray_pose_count:
    raise RuntimeError(
        "Unexpected ring-interleaved spray-on pose count."
    )


waypoints = [
    point.copy()
    for point in scheduled_tool_positions
]

impact_waypoints = np.asarray(
    scheduled_impact_positions,
    dtype=float,
)

waypoint_surface_normals = np.asarray(
    scheduled_surface_normals,
    dtype=float,
)

turntable_angles = np.asarray(
    scheduled_turntable_angles,
    dtype=float,
)

spray_enabled_schedule = np.asarray(
    scheduled_spray_enabled,
    dtype=bool,
)

pose_positions = jnp.asarray(
    np.asarray(
        scheduled_material_nozzle_positions,
        dtype=float,
    ),
    dtype=jnp.float32,
)

pose_rotations = jnp.asarray(
    np.asarray(
        scheduled_material_quaternions,
        dtype=float,
    ),
    dtype=jnp.float32,
)

n_waypoints = len(waypoints)
steps_per_move = max(
    1,
    steps // n_waypoints,
)

print("Ring-interleaved indexed turntable schedule:")
print(
    f"  recovered rings:         "
    f"{len(base_ring_waypoint_indices)}"
)
print(
    f"  waypoints per ring:      "
    f"{ring_waypoint_counts.tolist()}"
)
print(
    f"  rings per group:         "
    f"{RINGS_PER_INTERLEAVED_GROUP}"
)
print(
    f"  ring groups:             "
    f"{len(ring_waypoint_groups)}"
)
print(
    f"  spray blocks:            "
    f"{len(schedule_blocks)}"
)
print(
    f"  indexed orientations:    "
    f"{len(TURNTABLE_PASS_ANGLES_DEG)}"
)
print(
    f"  accessible waypoints:    "
    f"{BASE_PASS_WAYPOINT_COUNT}"
)
print(
    f"  total scheduled poses:   "
    f"{n_waypoints}"
)
print(
    f"  spray-on poses:          "
    f"{np.count_nonzero(spray_enabled_schedule)}"
)
print(
    f"  transition/return poses: "
    f"{np.count_nonzero(~spray_enabled_schedule)}"
)

if n_waypoints != 436:
    raise RuntimeError(
        "Controlled comparison requires exactly 436 scheduled poses; "
        f"received {n_waypoints}."
    )

if np.count_nonzero(
    spray_enabled_schedule
) != 396:
    raise RuntimeError(
        "Controlled comparison requires exactly 396 spray-on poses."
    )

if np.count_nonzero(
    ~spray_enabled_schedule
) != 40:
    raise RuntimeError(
        "Controlled comparison requires exactly 40 spray-off poses."
    )

print("Precomputing indexed-turntable spray distributions...")



def compute_h_for_pose(pos, rot):
    weight = deposit(
        pos, rot, sigma, a, ref_dist, resolution,
        face_v0, face_v1, face_v2, face_normals, n_faces, fov
    )
    return h_ambient + weight * h_spray_scale

_spray_cache_hash = hashlib.sha256()

_spray_cache_hash.update(
    np.ascontiguousarray(
        np.asarray(pose_positions)
    ).tobytes()
)

_spray_cache_hash.update(
    np.ascontiguousarray(
        np.asarray(pose_rotations)
    ).tobytes()
)

_spray_cache_hash.update(
    np.ascontiguousarray(
        spray_enabled_schedule
    ).tobytes()
)

_spray_cache_hash.update(
    np.asarray(
        [
            sigma,
            a,
            ref_dist,
            resolution,
            fov,
            h_ambient,
            h_spray_scale,
        ],
        dtype=np.float64,
    ).tobytes()
)

_spray_cache_path = os.path.join(
    RUNTIME_DIR,
    "half_sphere_turntable_h_"
    + _spray_cache_hash.hexdigest()[:16]
    + ".npy",
)

h_fields = None

if os.path.isfile(_spray_cache_path):
    cached_h_fields = np.load(
        _spray_cache_path,
        allow_pickle=False,
    )

    if cached_h_fields.shape == (
        n_waypoints,
        n_faces,
    ):
        h_fields = jnp.asarray(
            cached_h_fields,
            dtype=jnp.float32,
        )

        print(
            "Loaded cached indexed-turntable "
            "spray distributions."
        )

if h_fields is None:
    h_fields = jax.vmap(
        compute_h_for_pose
    )(
        pose_positions,
        pose_rotations,
    )

    spray_enabled_jax = jnp.asarray(
        spray_enabled_schedule,
        dtype=bool,
    )

    h_fields = jnp.where(
        spray_enabled_jax[:, None],
        h_fields,
        jnp.full_like(
            h_fields,
            h_ambient,
        ),
    )

    h_fields.block_until_ready()

    np.save(
        _spray_cache_path,
        np.asarray(
            h_fields,
            dtype=np.float32,
        ),
        allow_pickle=False,
    )

    print(
        "Saved indexed-turntable spray cache:"
    )
    print(
        f"  {_spray_cache_path}"
    )

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

# ==================================================================
# CONSERVATIVE THIN-SHELL THERMAL PHYSICS
# ==================================================================
#
# Each triangular face is treated as a thermal control volume:
#
#   C_i dT_i/dt
#       = sum_j G_ij (T_j - T_i)
#         - h_i A_i (T_i - T_ambient)
#
# where:
#
#   C_i  = rho*c*thickness*A_i               [J/K]
#   G_ij = k*thickness*shared_edge_length
#          / dual_center_distance             [W/K]
#
# The OBJ itself has zero geometric thickness. PLATE_THICKNESS supplies
# the uniform physical shell thickness. This is a surface-shell model:
# it resolves conduction along the part but assumes temperature is uniform
# through the thickness.
#
# Shared-edge fluxes are assembled pairwise. Every conductive watt removed
# from one face is added to its neighboring face, so internal conduction
# conserves total thermal energy.
# ==================================================================

print("Building conservative thin-shell thermal operator...")

mesh_pv = surface_mesh.polydata

points_np = np.asarray(
    surface_mesh.points,
    dtype=np.float64,
)

faces_np = np.asarray(
    surface_mesh.faces,
    dtype=np.int64,
)

fv_face_centers_np = np.asarray(
    surface_mesh.face_centers,
    dtype=np.float64,
)

fv_face_area_np = np.asarray(
    surface_mesh.face_areas,
    dtype=np.float64,
)

if faces_np.shape != (n_faces, 3):
    raise RuntimeError(
        "Thermal mesh mismatch: expected "
        f"({n_faces}, 3), received {faces_np.shape}."
    )

if np.any(fv_face_area_np <= 0.0):
    raise RuntimeError(
        "The thermal mesh contains a nonpositive face area."
    )

if PLATE_THICKNESS <= 0.0:
    raise RuntimeError(
        "Shell thickness must be greater than zero."
    )

# Lumped thermal capacity of every triangular surface control volume.
thermal_capacity_np = (
    rho_c
    * PLATE_THICKNESS
    * fv_face_area_np
)

if np.any(thermal_capacity_np <= 0.0):
    raise RuntimeError(
        "The thermal mesh contains nonpositive thermal capacity."
    )

# Map each welded topological edge to the faces sharing that edge.
edge_to_faces = {}

for face_index, face in enumerate(faces_np):
    for local_edge in range(3):
        vertex_a = int(face[local_edge])
        vertex_b = int(face[(local_edge + 1) % 3])

        edge = (
            min(vertex_a, vertex_b),
            max(vertex_a, vertex_b),
        )

        edge_to_faces.setdefault(
            edge,
            [],
        ).append(face_index)

edge_i = []
edge_j = []
edge_conductance = []

boundary_edge_count = 0
nonmanifold_edges = []

for edge, adjacent_faces in edge_to_faces.items():
    if len(adjacent_faces) == 1:
        boundary_edge_count += 1
        continue

    if len(adjacent_faces) != 2:
        nonmanifold_edges.append(
            (edge, tuple(adjacent_faces))
        )
        continue

    face_i, face_j = adjacent_faces
    vertex_a, vertex_b = edge

    point_a = points_np[vertex_a]
    point_b = points_np[vertex_b]

    edge_length = float(
        np.linalg.norm(
            point_b - point_a
        )
    )

    edge_midpoint = 0.5 * (
        point_a + point_b
    )

    # Two half-cell conduction paths in series:
    # face center i -> shared edge -> face center j.
    distance_i = float(
        np.linalg.norm(
            fv_face_centers_np[face_i]
            - edge_midpoint
        )
    )

    distance_j = float(
        np.linalg.norm(
            fv_face_centers_np[face_j]
            - edge_midpoint
        )
    )

    dual_distance = max(
        distance_i + distance_j,
        1.0e-12,
    )

    conductance_ij = (
        k
        * PLATE_THICKNESS
        * edge_length
        / dual_distance
    )

    if not np.isfinite(conductance_ij) \
            or conductance_ij <= 0.0:
        raise RuntimeError(
            "Invalid thermal conductance on edge "
            f"{edge}: {conductance_ij}"
        )

    edge_i.append(face_i)
    edge_j.append(face_j)
    edge_conductance.append(conductance_ij)

if nonmanifold_edges:
    examples = nonmanifold_edges[:5]

    raise RuntimeError(
        "The thermal mesh contains nonmanifold edges. "
        f"Examples: {examples}"
    )

edge_i_np = np.asarray(
    edge_i,
    dtype=np.int32,
)

edge_j_np = np.asarray(
    edge_j,
    dtype=np.int32,
)

edge_conductance_np = np.asarray(
    edge_conductance,
    dtype=np.float64,
)

if edge_conductance_np.size == 0:
    raise RuntimeError(
        "No shared thermal edges were assembled."
    )

# Sum outgoing conductance at each face for the explicit stability bound.
conductance_sum_np = np.zeros(
    n_faces,
    dtype=np.float64,
)

np.add.at(
    conductance_sum_np,
    edge_i_np,
    edge_conductance_np,
)

np.add.at(
    conductance_sum_np,
    edge_j_np,
    edge_conductance_np,
)

# Confirm that pairwise conduction is internally energy-conservative.
conservation_test_temperature = np.linspace(
    T_ambient,
    T_initial,
    n_faces,
    dtype=np.float64,
)

conservation_test_flux = (
    edge_conductance_np
    * (
        conservation_test_temperature[edge_j_np]
        - conservation_test_temperature[edge_i_np]
    )
)

conservation_test_power = np.zeros(
    n_faces,
    dtype=np.float64,
)

np.add.at(
    conservation_test_power,
    edge_i_np,
    conservation_test_flux,
)

np.add.at(
    conservation_test_power,
    edge_j_np,
    -conservation_test_flux,
)

net_internal_power = float(
    np.sum(conservation_test_power)
)

absolute_internal_power = float(
    np.sum(
        np.abs(conservation_test_power)
    )
)

conservation_tolerance = max(
    1.0e-9,
    1.0e-11 * absolute_internal_power,
)

if abs(net_internal_power) > conservation_tolerance:
    raise RuntimeError(
        "Internal conduction failed conservation check: "
        f"net power={net_internal_power:.6e} W."
    )

# Convert immutable geometry and material arrays to JAX.
edge_i_jax = jnp.asarray(
    edge_i_np,
    dtype=jnp.int32,
)

edge_j_jax = jnp.asarray(
    edge_j_np,
    dtype=jnp.int32,
)

edge_conductance_jax = jnp.asarray(
    edge_conductance_np,
    dtype=jnp.float32,
)

face_area_jax = jnp.asarray(
    fv_face_area_np,
    dtype=jnp.float32,
)

thermal_capacity_jax = jnp.asarray(
    thermal_capacity_np,
    dtype=jnp.float32,
)

# Explicit-Euler stability:
#
#   lambda_i =
#       [sum_j G_ij + h_i,max A_i] / C_i
#
# Use a conservative fraction of 1/lambda_max.
h_fields_np = np.asarray(
    h_fields,
    dtype=np.float64,
)

if h_fields_np.shape != (
    n_waypoints,
    n_faces,
):
    raise RuntimeError(
        "Spray-field shape mismatch: expected "
        f"({n_waypoints}, {n_faces}), "
        f"received {h_fields_np.shape}."
    )

if not np.isfinite(h_fields_np).all():
    raise FloatingPointError(
        "Spray fields contain non-finite values."
    )

if np.any(h_fields_np < 0.0):
    raise RuntimeError(
        "Spray heat-transfer coefficients cannot be negative."
    )

maximum_h_per_face_np = np.max(
    h_fields_np,
    axis=0,
)

maximum_decay_rate_per_face = (
    conductance_sum_np
    + maximum_h_per_face_np
    * fv_face_area_np
) / thermal_capacity_np

maximum_decay_rate = float(
    np.max(
        maximum_decay_rate_per_face
    )
)

if not np.isfinite(maximum_decay_rate) \
        or maximum_decay_rate <= 0.0:
    raise RuntimeError(
        "Unable to determine a valid thermal stability rate."
    )

stable_dt_limit = (
    0.45 / maximum_decay_rate
)

THERMAL_SUBSTEPS = max(
    1,
    int(
        np.ceil(
            dt_sim / stable_dt_limit
        )
    ),
)

THERMAL_SUBSTEP_DT = (
    dt_sim / THERMAL_SUBSTEPS
)

print("Conservative thin-shell operator built.")
print(
    f"  faces:                 {n_faces}"
)
print(
    f"  shared thermal edges:  "
    f"{len(edge_conductance_np)}"
)
print(
    f"  boundary edges:        "
    f"{boundary_edge_count}"
)
print(
    f"  shell thickness:       "
    f"{PLATE_THICKNESS:.6f} m"
)
print(
    f"  face area:             "
    f"{fv_face_area_np.min():.3e} to "
    f"{fv_face_area_np.max():.3e} m^2"
)
print(
    f"  face capacity:         "
    f"{thermal_capacity_np.min():.3e} to "
    f"{thermal_capacity_np.max():.3e} J/K"
)
print(
    f"  edge conductance:      "
    f"{edge_conductance_np.min():.3e} to "
    f"{edge_conductance_np.max():.3e} W/K"
)
print(
    f"  conduction balance:    "
    f"{net_internal_power:.3e} W"
)
print(
    f"  maximum decay rate:    "
    f"{maximum_decay_rate:.3e} 1/s"
)
print(
    f"  macro dt:              "
    f"{dt_sim:.6e} s"
)
print(
    f"  internal substeps:     "
    f"{THERMAL_SUBSTEPS}"
)
print(
    f"  internal dt:           "
    f"{THERMAL_SUBSTEP_DT:.6e} s"
)


# ==================================================================
# JAX TRANSIENT SOLVER
# ==================================================================

T_init = jnp.full(
    (n_faces,),
    T_initial,
    dtype=jnp.float32,
)


@jax.jit
def step(carry, _):
    """
    Advance one recorded simulation interval.

    The spray pose remains fixed during the internal thermal substeps.
    """
    temperature, step_index = carry

    pose_index = jnp.minimum(
        step_index // steps_per_move,
        n_waypoints - 1,
    )

    h_field = h_fields[pose_index]

    def thermal_substep(_, temperature_sub):
        # Positive edge power flows from face j into face i.
        edge_temperature_difference = (
            temperature_sub[edge_j_jax]
            - temperature_sub[edge_i_jax]
        )

        edge_power = (
            edge_conductance_jax
            * edge_temperature_difference
        )

        conductive_power = jnp.zeros_like(
            temperature_sub
        )

        conductive_power = conductive_power.at[
            edge_i_jax
        ].add(
            edge_power
        )

        conductive_power = conductive_power.at[
            edge_j_jax
        ].add(
            -edge_power
        )

        convective_power = (
            h_field
            * face_area_jax
            * (
                temperature_sub
                - T_ambient
            )
        )

        net_power = (
            conductive_power
            - convective_power
        )

        temperature_next = (
            temperature_sub
            + THERMAL_SUBSTEP_DT
            * net_power
            / thermal_capacity_jax
        )

        return temperature_next

    temperature_new = jax.lax.fori_loop(
        0,
        THERMAL_SUBSTEPS,
        thermal_substep,
        temperature,
    )

    return (
        temperature_new,
        step_index + 1,
    ), (
        temperature_new,
        pose_index,
    )


print("Running conservative thin-shell simulation...")

_, (
    T_history,
    pose_idx_history,
) = jax.lax.scan(
    step,
    (
        T_init,
        jnp.int32(0),
    ),
    None,
    length=steps,
)

T_history = np.asarray(
    T_history,
    dtype=np.float64,
)

pose_idx_history = np.asarray(
    pose_idx_history,
    dtype=np.int32,
)

if T_history.shape != (
    steps,
    n_faces,
):
    raise RuntimeError(
        "Temperature-history shape mismatch: "
        f"{T_history.shape}"
    )

if not np.isfinite(T_history).all():
    bad_count = int(
        T_history.size
        - np.count_nonzero(
            np.isfinite(T_history)
        )
    )

    raise FloatingPointError(
        "Thermal simulation produced "
        f"{bad_count} non-finite values."
    )

temperature_tolerance = 1.0e-2

minimum_temperature = float(
    np.min(T_history)
)

maximum_temperature = float(
    np.max(T_history)
)

if minimum_temperature < (
    T_ambient - temperature_tolerance
):
    raise RuntimeError(
        "Thermal solution fell below ambient temperature: "
        f"{minimum_temperature:.6f} C."
    )

if maximum_temperature > (
    T_initial + temperature_tolerance
):
    raise RuntimeError(
        "Thermal solution exceeded the initial maximum: "
        f"{maximum_temperature:.6f} C."
    )

initial_excess_energy = float(
    np.sum(
        thermal_capacity_np
        * (
            T_initial
            - T_ambient
        )
    )
)

final_excess_energy = float(
    np.sum(
        thermal_capacity_np
        * (
            T_history[-1]
            - T_ambient
        )
    )
)

energy_removed = (
    initial_excess_energy
    - final_excess_energy
)

if energy_removed < -1.0e-6:
    raise RuntimeError(
        "Thermal energy increased despite having no heat source."
    )

print("PASS: all temperatures are finite.")
print("PASS: temperatures remain between ambient and initial.")
print("PASS: internal conduction is energy-conservative.")
print("PASS: total thermal energy decreases under convection.")
print("Done.")

print("\nFinal results:")
print(
    f"  Peak temp:       "
    f"{T_history[-1].max():.1f} C"
)
print(
    f"  Min temp:        "
    f"{T_history[-1].min():.1f} C"
)
print(
    f"  Temp spread:     "
    f"{T_history[-1].max() - T_history[-1].min():.1f} C"
)
print(
    f"  Avg temp:        "
    f"{T_history[-1].mean():.1f} C"
)
print(
    f"  Initial energy:  "
    f"{initial_excess_energy / 1000.0:.2f} kJ above ambient"
)
print(
    f"  Final energy:    "
    f"{final_excess_energy / 1000.0:.2f} kJ above ambient"
)
print(
    f"  Energy removed:  "
    f"{energy_removed / 1000.0:.2f} kJ"
)




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

# BOUNDED UR5E IK SAFETY
#
# IKPy rejects an initial joint state when any value lies outside the
# corresponding URDF bounds. The curved path's angle-unwrapping logic can
# produce an equivalent angle displaced by ±2*pi. Canonicalize every input
# and output joint vector before it enters or leaves IKPy.
_original_chain_inverse_kinematics = chain.inverse_kinematics
_ik_sanitization_events = 0
_ik_call_count = 0


def _link_bounds(index):
    """Return finite/infinite numerical bounds for one chain link."""
    bounds = getattr(chain.links[index], "bounds", None)

    if bounds is None or len(bounds) != 2:
        return -np.inf, np.inf

    lower, upper = bounds

    lower = (
        -np.inf
        if lower is None
        else float(lower)
    )

    upper = (
        np.inf
        if upper is None
        else float(upper)
    )

    return lower, upper


def _equivalent_angle_in_bounds(
    value,
    lower,
    upper,
    reference=None,
):
    """
    Move an angle by integer multiples of 2*pi into its legal interval.

    This preserves the physical orientation rather than merely clipping an
    out-of-range angle to a joint limit.
    """
    value = float(value)

    if not np.isfinite(value):
        if np.isfinite(lower) and np.isfinite(upper):
            return 0.5 * (lower + upper)

        if np.isfinite(lower):
            return lower

        if np.isfinite(upper):
            return upper

        return 0.0

    if not np.isfinite(lower) and not np.isfinite(upper):
        return value

    two_pi = 2.0 * np.pi

    candidates = []

    for shift in range(-12, 13):
        candidate = value + shift * two_pi

        if candidate >= lower - 1.0e-10 \
                and candidate <= upper + 1.0e-10:
            candidates.append(candidate)

    if candidates:
        if reference is not None and np.isfinite(reference):
            selected = min(
                candidates,
                key=lambda candidate: abs(
                    candidate - float(reference)
                ),
            )
        else:
            interval_target = np.clip(
                value,
                lower,
                upper,
            )

            selected = min(
                candidates,
                key=lambda candidate: abs(
                    candidate - interval_target
                ),
            )

        return float(
            np.clip(selected, lower, upper)
        )

    return float(
        np.clip(value, lower, upper)
    )


def sanitize_ik_joint_vector(
    joint_vector,
    *,
    context,
    reference=None,
):
    """
    Return a finite joint vector whose entries satisfy every IKPy bound.
    """
    global _ik_sanitization_events

    q = np.asarray(
        joint_vector,
        dtype=float,
    ).copy()

    if q.shape != (len(chain.links),):
        raise ValueError(
            f"{context}: expected {len(chain.links)} joint values, "
            f"received shape {q.shape}."
        )

    if reference is not None:
        reference_array = np.asarray(
            reference,
            dtype=float,
        )

        if reference_array.shape != q.shape:
            reference_array = None
    else:
        reference_array = None

    corrections = []

    for index, link in enumerate(chain.links):
        lower, upper = _link_bounds(index)

        reference_value = (
            None
            if reference_array is None
            else reference_array[index]
        )

        corrected = _equivalent_angle_in_bounds(
            q[index],
            lower,
            upper,
            reference=reference_value,
        )

        if not np.isclose(
            corrected,
            q[index],
            rtol=0.0,
            atol=1.0e-10,
        ):
            corrections.append(
                (
                    index,
                    link.name,
                    float(q[index]),
                    corrected,
                    lower,
                    upper,
                )
            )

        q[index] = corrected

    if not np.isfinite(q).all():
        raise FloatingPointError(
            f"{context}: non-finite IK joint state remained "
            "after sanitization."
        )

    for index, link in enumerate(chain.links):
        lower, upper = _link_bounds(index)

        if q[index] < lower - 1.0e-9 \
                or q[index] > upper + 1.0e-9:
            raise ValueError(
                f"{context}: joint {index} ({link.name}) = "
                f"{q[index]:.6f} remains outside "
                f"[{lower:.6f}, {upper:.6f}]."
            )

    if corrections:
        _ik_sanitization_events += 1

        if _ik_sanitization_events <= 12:
            print(
                f"IK bound correction during {context}:"
            )

            for (
                index,
                name,
                original,
                corrected,
                lower,
                upper,
            ) in corrections:
                print(
                    f"  joint {index} {name}: "
                    f"{original:.6f} -> {corrected:.6f} "
                    f"within [{lower:.6f}, {upper:.6f}]"
                )

        elif _ik_sanitization_events == 13:
            print(
                "Additional repeated IK bound corrections "
                "will be counted but not printed."
            )

    return q


def _safe_chain_inverse_kinematics(*args, **kwargs):
    """
    Bound-check all IKPy initial guesses and returned joint solutions.
    """
    global _ik_call_count

    _ik_call_count += 1

    target_position = kwargs.get(
        "target_position",
        args[0] if args else None,
    )

    initial_position = kwargs.get(
        "initial_position",
    )

    sanitized_initial = None

    if initial_position is not None:
        sanitized_initial = sanitize_ik_joint_vector(
            initial_position,
            context=f"IK call {_ik_call_count} initial guess",
        )

        kwargs["initial_position"] = sanitized_initial

    try:
        solution = _original_chain_inverse_kinematics(
            *args,
            **kwargs,
        )
    except Exception:
        print()
        print("IK SOLVER FAILURE")
        print(f"  IK call: { _ik_call_count }")
        print(f"  target position: {target_position}")

        if sanitized_initial is not None:
            print(
                "  sanitized initial state:",
                np.array2string(
                    sanitized_initial,
                    precision=5,
                ),
            )

        print("  joint bounds:")

        for index, link in enumerate(chain.links):
            lower, upper = _link_bounds(index)

            print(
                f"    {index}: {link.name} "
                f"[{lower:.6f}, {upper:.6f}]"
            )

        raise

    return sanitize_ik_joint_vector(
        solution,
        context=f"IK call {_ik_call_count} solution",
        reference=sanitized_initial,
    )


chain.inverse_kinematics = _safe_chain_inverse_kinematics


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


# Exact curved-surface IK.
#
# The earlier implementation correctly requested a surface-normal pose,
# but then modified the valid IK result with wrist bias, joint low-pass
# filtering, and later whole-path smoothing. Those modifications destroyed
# both tool position and nozzle orientation.
_ik_orientation_sign = None
_ik_home_seed = None
_ik_retry_count = 0


def desired_outward_normal(target_world):
    """Return the analytical outward normal of the hemisphere."""
    target_world = np.asarray(
        target_world,
        dtype=float,
    )

    radial = target_world - SPHERE_CENTER
    magnitude = np.linalg.norm(radial)

    if magnitude < 1.0e-12:
        return np.array(
            [0.0, 0.0, 1.0],
            dtype=float,
        )

    return radial / magnitude


def ik_solution_errors(target_world, q):
    """
    Measure the actual URDF tool position and spray-direction errors.

    Desired spray direction is inward, opposite the outward dome normal.
    """
    target_world = np.asarray(
        target_world,
        dtype=float,
    )

    achieved_position = np.asarray(
        get_tool0_world(q),
        dtype=float,
    )

    achieved_spray_direction = np.asarray(
        get_nozzle_direction_world(q),
        dtype=float,
    )

    achieved_spray_direction /= (
        np.linalg.norm(achieved_spray_direction)
        + 1.0e-12
    )

    desired_spray_direction = -desired_outward_normal(
        target_world
    )

    position_error = float(
        np.linalg.norm(
            achieved_position - target_world
        )
    )

    alignment = float(
        np.clip(
            np.dot(
                achieved_spray_direction,
                desired_spray_direction,
            ),
            -1.0,
            1.0,
        )
    )

    normal_error_deg = float(
        np.degrees(
            np.arccos(alignment)
        )
    )

    return position_error, normal_error_deg


def ik_candidate_score(target_world, q):
    """
    Combine position and orientation error for candidate comparison.

    One degree of normal error receives a 2 mm equivalent penalty.
    """
    position_error, normal_error_deg = (
        ik_solution_errors(target_world, q)
    )

    score = (
        position_error
        + 0.002 * normal_error_deg
    )

    return score, position_error, normal_error_deg


def solve_oriented_candidate(
    target_world,
    initial_state,
    orientation_sign,
):
    """Calculate one exact orientation-constrained IK solution."""
    target_world = np.asarray(
        target_world,
        dtype=float,
    )

    target_base = world_to_base(
        target_world
    )

    outward_normal = desired_outward_normal(
        target_world
    )

    # IKPy aligns the terminal frame's local +Z with this vector.
    orientation_target = (
        float(orientation_sign)
        * outward_normal
    )

    initial_state = sanitize_ik_joint_vector(
        initial_state,
        context="exact IK initial state",
    )

    q_raw = chain.inverse_kinematics(
        target_position=target_base,
        target_orientation=orientation_target,
        orientation_mode="Z",
        initial_position=initial_state,
    )

    # Select the equivalent in-bounds representation closest to the
    # preceding pose. Do not average or smooth the IK result.
    return sanitize_ik_joint_vector(
        q_raw,
        context="exact IK solution",
        reference=initial_state,
    )


def calibrate_ik_orientation(target_world):
    """
    Determine which IKPy terminal-axis sign matches the URDF tool0 -Z spray.

    This is measured from the actual URDF transforms rather than assumed.
    """
    global _ik_orientation_sign
    global _ik_home_seed
    global _last_joint_state

    seed = sanitize_ik_joint_vector(
        _last_joint_state,
        context="IK orientation calibration seed",
    )

    _ik_home_seed = seed.copy()

    candidates = []

    for sign in (1.0, -1.0):
        try:
            q = solve_oriented_candidate(
                target_world,
                seed,
                sign,
            )
        except Exception as error:
            print(
                f"IK orientation sign {sign:+.0f} failed: "
                f"{type(error).__name__}: {error}"
            )
            continue

        score, position_error, normal_error = (
            ik_candidate_score(
                target_world,
                q,
            )
        )

        candidates.append(
            (
                score,
                sign,
                q,
                position_error,
                normal_error,
            )
        )

        print(
            f"IK orientation sign {sign:+.0f}: "
            f"position={position_error * 1000.0:.2f} mm, "
            f"normal={normal_error:.2f} deg"
        )

    if not candidates:
        raise RuntimeError(
            "Neither tool-axis orientation produced an IK solution."
        )

    (
        _,
        selected_sign,
        selected_q,
        selected_position_error,
        selected_normal_error,
    ) = min(
        candidates,
        key=lambda item: item[0],
    )

    _ik_orientation_sign = selected_sign
    _last_joint_state = selected_q.copy()

    print("IK orientation calibration selected:")
    print(
        f"  terminal-axis sign: {selected_sign:+.0f}"
    )
    print(
        f"  position error:     "
        f"{selected_position_error * 1000.0:.2f} mm"
    )
    print(
        f"  normal error:       "
        f"{selected_normal_error:.2f} deg"
    )

    return selected_q.copy()


def solve_ik_direct(target_world):
    """Return an exact, orientation-constrained IK solution."""
    global _ik_orientation_sign
    global _last_joint_state

    if _ik_orientation_sign is None:
        return calibrate_ik_orientation(
            target_world
        )

    q = solve_oriented_candidate(
        target_world,
        _last_joint_state,
        _ik_orientation_sign,
    )

    _last_joint_state = q.copy()
    return q


def solve_ik(target_world):
    """
    Solve one exact curved-surface pose using the previous pose as the seed.

    No wrist bias, low-pass filtering, or joint-path smoothing is applied.
    A secondary seed is tried only when the first exact solution has poor
    measured tracking.
    """
    global _ik_orientation_sign
    global _ik_home_seed
    global _ik_retry_count
    global _last_joint_state

    if _ik_orientation_sign is None:
        return calibrate_ik_orientation(
            target_world
        )

    previous_state = sanitize_ik_joint_vector(
        _last_joint_state,
        context="previous exact IK state",
    )

    candidate_solutions = []

    seeds = [
        (
            "previous pose",
            previous_state,
        ),
    ]

    for seed_name, seed in seeds:
        try:
            q = solve_oriented_candidate(
                target_world,
                seed,
                _ik_orientation_sign,
            )

            score, position_error, normal_error = (
                ik_candidate_score(
                    target_world,
                    q,
                )
            )

            candidate_solutions.append(
                (
                    score,
                    q,
                    position_error,
                    normal_error,
                    seed_name,
                )
            )
        except Exception:
            continue

    primary_is_good = (
        candidate_solutions
        and candidate_solutions[0][2] <= 0.015
        and candidate_solutions[0][3] <= 8.0
    )

    if not primary_is_good:
        retry_seeds = []

        if _ik_home_seed is not None:
            retry_seeds.append(
                (
                    "home seed",
                    _ik_home_seed,
                )
            )

        retry_seeds.append(
            (
                "neutral seed",
                np.zeros(
                    len(chain.links),
                    dtype=float,
                ),
            )
        )

        for seed_name, seed in retry_seeds:
            try:
                q = solve_oriented_candidate(
                    target_world,
                    seed,
                    _ik_orientation_sign,
                )

                score, position_error, normal_error = (
                    ik_candidate_score(
                        target_world,
                        q,
                    )
                )

                candidate_solutions.append(
                    (
                        score,
                        q,
                        position_error,
                        normal_error,
                        seed_name,
                    )
                )
            except Exception:
                continue

        _ik_retry_count += 1

    if not candidate_solutions:
        raise RuntimeError(
            f"No exact IK solution for target "
            f"{np.asarray(target_world)}."
        )

    (
        _,
        selected_q,
        selected_position_error,
        selected_normal_error,
        selected_seed_name,
    ) = min(
        candidate_solutions,
        key=lambda item: item[0],
    )

    if (
        _ik_retry_count <= 10
        and selected_seed_name != "previous pose"
    ):
        print(
            f"IK retry selected {selected_seed_name}: "
            f"position={selected_position_error * 1000.0:.2f} mm, "
            f"normal={selected_normal_error:.2f} deg"
        )

    _last_joint_state = selected_q.copy()
    return selected_q.copy()


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

mesh_vis = raw_mesh.copy(deep=True)
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

    ax.set_title("Clicked half-sphere temperature history", color="black", fontsize=13)
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
            "Click the half-sphere on the left\nto add temperature histories",
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


current_turntable_angle = 0.0


def make_turntable_matrix(angle_rad):
    rotation = _rotation_z(
        angle_rad
    )

    translate_to_origin = np.eye(4)
    translate_to_origin[:3, 3] = (
        -SPHERE_CENTER
    )

    rotate = np.eye(4)
    rotate[:3, :3] = rotation

    translate_back = np.eye(4)
    translate_back[:3, 3] = (
        SPHERE_CENTER
    )

    return (
        translate_back
        @ rotate
        @ translate_to_origin
    )


def apply_turntable_angle(angle_rad):
    global current_turntable_angle

    current_turntable_angle = float(
        angle_rad
    )

    matrix = make_turntable_matrix(
        current_turntable_angle
    )

    set_actor_matrix(
        heatmap_actor,
        matrix,
    )

    for marker_actor in globals().get(
        "picked_marker_actors",
        [],
    ):
        set_actor_matrix(
            marker_actor,
            matrix,
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


    if "nozzle_actor" in globals():
        set_actor_matrix(
            nozzle_actor,
            get_tool0_transform_world(q),
        )

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

def make_pressure_jet(
    jet_origin,
    impact_center,
    surface_normal,
    t,
):
    """
    Generate a coherent jet from the physical nozzle tip to the
    curved surface. No flat-plate z assumption is used.
    """
    jet_origin = np.asarray(jet_origin, dtype=float)
    impact_center = np.asarray(impact_center, dtype=float)
    surface_normal = np.asarray(surface_normal, dtype=float)

    surface_normal /= (
        np.linalg.norm(surface_normal)
        + 1.0e-12
    )

    p0 = jet_origin.copy()

    # End just outside the surface to prevent visual z-fighting.
    p2_center = (
        impact_center
        + 0.003 * surface_normal
    )

    axis = p2_center - p0
    axis_length = np.linalg.norm(axis)

    if axis_length < 1.0e-8:
        raise ValueError(
            "Jet origin and impact point are coincident."
        )

    direction = axis / axis_length

    reference = np.array(
        [0.0, 0.0, 1.0],
        dtype=float,
    )

    if abs(np.dot(direction, reference)) > 0.95:
        reference = np.array(
            [1.0, 0.0, 0.0],
            dtype=float,
        )

    tangent_u = np.cross(direction, reference)
    tangent_u /= (
        np.linalg.norm(tangent_u)
        + 1.0e-12
    )

    tangent_v = np.cross(direction, tangent_u)
    tangent_v /= (
        np.linalg.norm(tangent_v)
        + 1.0e-12
    )

    top_radius = JET_CORE_RADIUS * _r_unit
    middle_radius = JET_MID_RADIUS * _r_unit

    bottom_radius = (
        JET_EXIT_RADIUS * _r_unit
        + 0.0005 * np.sin(
            5.0 * t + _phase
        )
    )

    top_offsets = (
        top_radius[:, None]
        * np.cos(_theta)[:, None]
        * tangent_u[None, :]
        + top_radius[:, None]
        * np.sin(_theta)[:, None]
        * tangent_v[None, :]
    )

    middle_angle = (
        _theta
        + 0.008
        * np.sin(3.0 * t + _phase)
    )

    middle_offsets = (
        middle_radius[:, None]
        * np.cos(middle_angle)[:, None]
        * tangent_u[None, :]
        + middle_radius[:, None]
        * np.sin(middle_angle)[:, None]
        * tangent_v[None, :]
    )

    bottom_angle = (
        _theta
        + 0.010 * np.sin(2.5 * t)
    )

    bottom_offsets = (
        bottom_radius[:, None]
        * np.cos(bottom_angle)[:, None]
        * tangent_u[None, :]
        + bottom_radius[:, None]
        * np.sin(bottom_angle)[:, None]
        * tangent_v[None, :]
    )

    p_top = p0[None, :] + top_offsets

    p_middle_center = (
        0.50 * p0
        + 0.50 * p2_center
    )

    p_middle = (
        p_middle_center[None, :]
        + middle_offsets
    )

    p_bottom = (
        p2_center[None, :]
        + bottom_offsets
    )

    points = np.vstack(
        [p_top, p_middle, p_bottom]
    ).astype(np.float32)

    line_data = []

    for index in range(N_JET_LINES):
        line_data.extend(
            [
                2,
                index,
                index + N_JET_LINES,
            ]
        )

        line_data.extend(
            [
                2,
                index + N_JET_LINES,
                index + 2 * N_JET_LINES,
            ]
        )

    return pv.PolyData(
        points,
        lines=np.asarray(
            line_data,
            dtype=np.int64,
        ),
    )

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

# Visible physical nozzle from tool0 to the calculated nozzle exit.
nozzle_mesh_local = pv.Cylinder(
    center=(
        0.0,
        0.0,
        -0.5 * NOZZLE_LENGTH,
    ),
    direction=(0.0, 0.0, -1.0),
    radius=0.012,
    height=NOZZLE_LENGTH,
    resolution=32,
)

nozzle_actor = plotter.add_mesh(
    nozzle_mesh_local,
    color="silver",
    smooth_shading=True,
    specular=0.5,
    specular_power=20,
)

nozzle_actor.SetPickable(False)



def set_spray_head_position(pos_world):
    M = np.eye(4)
    M[:3, 3] = np.array(pos_world)
    set_actor_matrix(spray_head_actor, M)

# initial pose + pressure jet actor
q_init = solve_ik_direct(np.array(waypoints[0]))
_last_joint_state = q_init.copy()
update_ur5e_pose(q_init)

tool0_init = get_tool0_world(q_init)
nozzle_tip_init = get_nozzle_tip_world(q_init)

impact_init = np.array(
    impact_waypoints[0],
    dtype=float,
)

impact_normal_init = np.array(
    waypoint_surface_normals[0],
    dtype=float,
)

set_spray_head_position(nozzle_tip_init)

spray_poly = make_pressure_jet(
    nozzle_tip_init,
    impact_init,
    impact_normal_init,
    0.0,
)
spray_actor = plotter.add_mesh(
    spray_poly,
    color='cyan',
    line_width=4,
    opacity=0.95,
    render_lines_as_tubes=True,
)
spray_actor.SetPickable(False)

# Keep spray and temporary robot pose hidden during IK planning.
spray_actor.SetVisibility(False)
spray_head_actor.SetVisibility(False)
nozzle_actor.SetVisibility(False)

for actors in link_actors.values():
    for actor in actors:
        actor.SetVisibility(False)

plotter.camera_position = [
    (1.12, -1.05, 0.72),
    (0.45, 0.0, -0.02),
    (0.0, 0.0, 1.0),
]
plotter.camera.SetClippingRange(0.01, 10.0)
# clickable plate temperature-history callback
print("Enabling clickable half-sphere temperature-history picking...")

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
    """
    Add a marker at the world-space click while retrieving temperature
    history from the corresponding material-space triangular face.
    """
    p_click_world = np.asarray(
        p_click,
        dtype=float,
    )

    if p_click_world.shape != (3,) \
            or not np.isfinite(
                p_click_world
            ).all():
        print("Ignored invalid surface click.")
        return

    # Undo the current table rotation to identify the material face.
    p_click_material = (
        SPHERE_CENTER
        + _rotation_z(
            -current_turntable_angle
        )
        @ (
            p_click_world
            - SPHERE_CENTER
        )
    )

    distances = np.linalg.norm(
        plate_pick_centers
        - p_click_material[None, :],
        axis=1,
    )

    face_idx = int(
        np.argmin(distances)
    )

    if distances[face_idx] > 0.040:
        print(
            "Ignored click too far from the part: "
            f"{distances[face_idx]:.3f} m"
        )
        return

    material_normal = np.asarray(
        direct_face_normals[face_idx],
        dtype=float,
    )

    material_normal /= (
        np.linalg.norm(material_normal)
        + 1.0e-12
    )

    radial_material = (
        p_click_material
        - SPHERE_CENTER
    )

    if np.dot(
        material_normal,
        radial_material,
    ) < 0.0:
        material_normal = -material_normal

    marker_center_material = (
        p_click_material
        + 0.010 * material_normal
    )

    marker_mesh = pv.Sphere(
        radius=0.009,
        center=marker_center_material,
        theta_resolution=24,
        phi_resolution=24,
    )

    marker_number = len(
        picked_marker_actors
    )

    plotter.subplot(0, 0)

    marker_actor = plotter.add_mesh(
        marker_mesh,
        name=f"surface_click_{marker_number}",
        color="dodgerblue",
        lighting=False,
        smooth_shading=True,
    )

    marker_actor.SetPickable(False)

    set_actor_matrix(
        marker_actor,
        make_turntable_matrix(
            current_turntable_angle
        ),
    )

    try:
        _force_on_top(marker_actor)
    except Exception:
        pass

    picked_marker_actors.append(
        marker_actor
    )

    click_base = (
        p_click_world
        - ROBOT_BASE
    )

    label = (
        f"({click_base[0] * 1000:.0f}, "
        f"{click_base[1] * 1000:.0f}, "
        f"{click_base[2] * 1000:.0f}) mm from base"
    )

    if face_idx not in picked_face_ids:
        history_curves.append(
            T_history[:, face_idx]
        )

        history_labels.append(
            label
        )

        picked_face_ids.append(
            face_idx
        )

        refresh_history_panel()

        print(
            f"Added temperature history: "
            f"{label}, face={face_idx}"
        )
    else:
        print(
            f"Added marker at {label}; "
            f"face {face_idx} curve already exists."
        )

    plotter.render()


def on_click_position_for_history(
    pick_world,
    picker=None,
):
    p_click = np.asarray(
        pick_world,
        dtype=float,
    )

    print(
        f"[surface click] point={p_click}"
    )

    add_temperature_curve_from_point(
        p_click
    )


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

print("Clickable half-sphere temperature-history picking enabled.")

plotter.show(auto_close=False, interactive_update=True)

# animation loop


print("Animating...")

# More frames = smoother visual path.
IK_ANIMATION_FRAMES = 360

skip = max(
    1,
    int(
        np.ceil(
            steps / IK_ANIMATION_FRAMES
        )
    ),
)
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
    # The thermal simulation clamps its pose index at the final waypoint.
    # Apply the same limit here so the visual path never indexes beyond
    # waypoint 383.
    path_pos = min(
        frame / float(steps_per_move),
        float(n_waypoints - 1),
    )

    i0 = min(
        int(np.floor(path_pos)),
        n_waypoints - 1,
    )

    i1 = min(
        i0 + 1,
        n_waypoints - 1,
    )

    tau = np.clip(
        path_pos - i0,
        0.0,
        1.0,
    )

    # Smoother than ordinary smoothstep: reduces start/stop jerk.
    s = tau**3 * (
        10.0
        - 15.0 * tau
        + 6.0 * tau**2
    )

    normal_0 = np.asarray(
        waypoint_surface_normals[i0],
        dtype=np.float32,
    )

    normal_1 = np.asarray(
        waypoint_surface_normals[i1],
        dtype=np.float32,
    )

    blended_normal = (
        (1.0 - s) * normal_0
        + s * normal_1
    )

    blended_normal /= (
        np.linalg.norm(blended_normal)
        + 1.0e-12
    )

    # Remain at a constant radial standoff from the curved surface.
    raw_target = (
        SPHERE_CENTER
        + (
            SPHERE_RADIUS
            + HEMISPHERE_STANDOFF
        ) * blended_normal
    )

    # Low-pass the target to soften ring transitions.
    visual_alpha = 0.09

    visual_target = (
        (1.0 - visual_alpha) * visual_target
        + visual_alpha * raw_target
    )

    # Low-pass interpolation shortens the radius slightly. Reproject onto
    # the correct tool path so the nozzle never drifts into the dome.
    visual_radial = (
        visual_target
        - SPHERE_CENTER
    )

    visual_radial /= (
        np.linalg.norm(visual_radial)
        + 1.0e-12
    )

    visual_target = (
        SPHERE_CENTER
        + (
            SPHERE_RADIUS
            + HEMISPHERE_STANDOFF
        ) * visual_radial
    )

    visual_targets.append(
        visual_target.astype(
            np.float32,
        ).copy()
    )

visual_targets = np.array(visual_targets)

# ------------------------------------------------------------------
# Precompute orientation-aware IK path.
# ------------------------------------------------------------------
# IK TRAJECTORY CACHE
_ik_cache_hash = hashlib.sha256()

_ik_cache_hash.update(
    np.ascontiguousarray(
        visual_targets
    ).tobytes()
)

_ik_cache_hash.update(
    np.ascontiguousarray(
        turntable_angles
    ).tobytes()
)

_ik_cache_hash.update(
    np.asarray(
        [
            NOZZLE_LENGTH,
            SPHERE_RADIUS,
            HEMISPHERE_STANDOFF,
            len(chain.links),
        ],
        dtype=np.float64,
    ).tobytes()
)

_ik_cache_path = os.path.join(
    RUNTIME_DIR,
    "half_sphere_turntable_ik_"
    + _ik_cache_hash.hexdigest()[:16]
    + ".npz",
)

_ik_cache_loaded = False

if os.path.isfile(_ik_cache_path):
    cached_ik = np.load(
        _ik_cache_path,
        allow_pickle=False,
    )

    candidate_path = np.asarray(
        cached_ik["q_anim_path"],
        dtype=float,
    )

    if (
        candidate_path.shape
        == (
            len(visual_targets),
            len(chain.links),
        )
        and np.isfinite(
            candidate_path
        ).all()
    ):
        q_anim_path = candidate_path
        _ik_cache_loaded = True

        print("Loaded cached exact IK trajectory:")
        print(f"  {_ik_cache_path}")
        print(
            f"Animation frames: {len(q_anim_path)}"
        )

if not _ik_cache_loaded:
    print("Precomputing exact normal-aligned IK joint path...")

    spray_actor.SetVisibility(False)
    spray_head_actor.SetVisibility(False)

    # Calibrate using the first actual animated target.
    _last_joint_state = solve_ik_direct(
        visual_targets[0]
    )

    q_anim_path = [
        _last_joint_state.copy()
    ]

    for target_index in range(
        1,
        len(visual_targets),
    ):
        target = visual_targets[target_index]

        try:
            q = solve_ik(target)
        except Exception:
            print()
            print("FAILED EXACT IK TARGET")
            print(f"  index:    {target_index}")
            print(f"  position: {np.asarray(target)}")
            raise

        q_anim_path.append(
            q.copy()
        )

    q_anim_path = np.asarray(
        q_anim_path,
        dtype=float,
    )

    if not np.isfinite(q_anim_path).all():
        raise FloatingPointError(
            "Exact IK trajectory contains non-finite values."
        )

    # Validate exact unsmoothed solutions. This is the path that will be shown.
    ik_position_errors = np.empty(
        len(q_anim_path),
        dtype=float,
    )

    ik_normal_errors_deg = np.empty(
        len(q_anim_path),
        dtype=float,
    )

    for index, (target, q) in enumerate(
        zip(
            visual_targets,
            q_anim_path,
        )
    ):
        (
            ik_position_errors[index],
            ik_normal_errors_deg[index],
        ) = ik_solution_errors(
            target,
            q,
        )

    worst_position_index = int(
        np.argmax(ik_position_errors)
    )

    worst_normal_index = int(
        np.argmax(ik_normal_errors_deg)
    )

    print("Exact IK trajectory validation:")
    print(
        f"  orientation sign:      "
        f"{_ik_orientation_sign:+.0f}"
    )
    print(
        f"  retry events:          "
        f"{_ik_retry_count}"
    )
    print(
        f"  mean position error:   "
        f"{ik_position_errors.mean() * 1000.0:.2f} mm"
    )
    print(
        f"  max position error:    "
        f"{ik_position_errors.max() * 1000.0:.2f} mm "
        f"at target {worst_position_index}"
    )
    print(
        f"  mean normal error:     "
        f"{ik_normal_errors_deg.mean():.2f} deg"
    )
    print(
        f"  max normal error:      "
        f"{ik_normal_errors_deg.max():.2f} deg "
        f"at target {worst_normal_index}"
    )
    print(
        f"  poses above 30 mm:     "
        f"{np.count_nonzero(ik_position_errors > 0.030)}"
    )
    print(
        f"  poses above 15 deg:    "
        f"{np.count_nonzero(ik_normal_errors_deg > 15.0)}"
    )

    if ik_position_errors.max() > 0.050:
        index = worst_position_index

        raise RuntimeError(
            "Exact IK position validation failed: "
            f"target {index}, "
            f"error={ik_position_errors[index] * 1000.0:.2f} mm, "
            f"position={visual_targets[index]}."
        )

    if ik_normal_errors_deg.max() > 20.0:
        index = worst_normal_index

        raise RuntimeError(
            "Exact IK normal validation failed: "
            f"target {index}, "
            f"error={ik_normal_errors_deg[index]:.2f} deg, "
            f"position={visual_targets[index]}."
        )

    print(f"Animation frames: {len(frame_list)}")
    print("Done precomputing normal-aligned animation path.")

    np.savez_compressed(
        _ik_cache_path,
        q_anim_path=np.asarray(
            q_anim_path,
            dtype=float,
        ),
    )

    print("Saved exact IK trajectory cache:")
    print(f"  {_ik_cache_path}")



# Turn spray on only after trajectory planning is complete.
spray_actor.SetVisibility(True)
spray_head_actor.SetVisibility(False)

# set first pose before display loop starts
q_first = q_anim_path[0]
update_ur5e_pose(q_first)
nozzle_actor.SetVisibility(True)

for actors in link_actors.values():
    for actor in actors:
        actor.SetVisibility(True)

tool0_first = get_tool0_world(q_first)
nozzle_tip_first = get_nozzle_tip_world(q_first)

set_spray_head_position(nozzle_tip_first)

# ------------------------------------------------------------------
# Replay cached trajectory.
# ------------------------------------------------------------------
while True:
    for j, frame in enumerate(frame_list):
        t_start = time.time()
        sim_time = frame * dt_sim

        scheduled_pose_index = int(
            pose_idx_history[frame]
        )

        apply_turntable_angle(
            turntable_angles[
                scheduled_pose_index
            ]
        )

        spray_is_enabled = bool(
            spray_enabled_schedule[
                scheduled_pose_index
            ]
        )

        spray_actor.SetVisibility(
            spray_is_enabled
        )

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
        nozzle_tip_world = get_nozzle_tip_world(q)
        jet_origin = nozzle_tip_world.copy()

        # Impact point comes from nearest mesh surface, not hardcoded plate z.
        # This is what will matter on sphere / irregular 3D shapes.
        impact_center, surf_normal = nearest_surface_point_and_normal(target)

        set_spray_head_position(jet_origin)

        if j == 0:
            err_world = np.linalg.norm(tool0_world - target)
            alignment = np.dot(get_nozzle_direction_world(q), -surf_normal)
            print(f"Start-of-pass tool0 tracking error: {err_world:.3f} m")
            print(f"Start-of-pass spray-normal alignment: {alignment:.3f}")

        new_spray = make_pressure_jet(
            jet_origin,
            impact_center,
            surf_normal,
            sim_time,
        )
        spray_poly.points = new_spray.points
        spray_poly.lines = new_spray.lines
        spray_poly.Modified()
        spray_actor.mapper.dataset.Modified()

        plotter.camera.SetClippingRange(0.01, 10.0)
        plotter.update(stime=1, force_redraw=True)

        elapsed = time.time() - t_start
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)