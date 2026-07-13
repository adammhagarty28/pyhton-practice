"""
POC: Robot Spray Cooling with JAX-PULSE on 3D Mesh — PyVista single window.

Robot arm follows zigzag path above plate. End effector position feeds
into deposit() which computes spray distribution. Heat equation evolves
temperature per triangle face. All rendered in one PyVista scene:
robot + spray droplets + heatmap plate.

Physics:
    dT/dt = alpha * laplacian_approx(T) - (h_local / rho_c) * (T - T_ambient)
    h_local = h_ambient + deposit(end_effector_pose) * h_spray_scale
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
from jax_pulse import Pulse

# =============================================================================
# parameters
# =============================================================================
T_initial     = 900.0
k             = 50.0
T_ambient     = 25.0
h_spray_scale = 200000.0
h_ambient     = 10.0
c             = 500.0
rho           = 7800.0
rho_c         = rho * c
alpha         = k / rho_c

dt_sim        = 0.1
t_end         = 600.0
steps         = int(t_end / dt_sim)

sigma         = 0.25
a             = 1.0
ref_dist      = 1.0
resolution    = 64
fov           = 90.0

mesh_path = '/mnt/c/Users/Owner/Downloads/jax-pulse-master/jax-pulse-master/data/meshes/obj/refined_plate.obj'

# =============================================================================
# load mesh + JAX-PULSE
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
print(f"Mesh loaded. Faces: {n_faces}")

# =============================================================================
# zigzag waypoints
# =============================================================================
x_range  = np.linspace(0.1, 2.9, 20)
y_range  = np.linspace(0.1, 2.9, 20)
z_height = 1.5

waypoints = []
for i, y in enumerate(y_range):
    xs = x_range if i % 2 == 0 else x_range[::-1]
    for x in xs:
        waypoints.append(np.array([x, y, z_height]))

n_waypoints    = len(waypoints)
steps_per_move = max(1, steps // n_waypoints)
print(f"Total waypoints: {n_waypoints}")

# =============================================================================
# precompute h_fields
# =============================================================================
print("Precomputing spray distributions...")
rot = jnp.array([1.0, 0.0, 0.0, 0.0])

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

# =============================================================================
# face adjacency
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
    pose_idx = jnp.minimum(step_idx // steps_per_move, n_waypoints - 1)
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
    T_new = T + dTdt * dt_sim

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

# =============================================================================
# forward kinematics — 3-DOF arm (base yaw + 2-link vertical plane)
# =============================================================================
BASE_POS = np.array([-0.3, 1.5, 0.0])
BASE_H   = 0.2
L1, L2   = 2.5, 2.5

def compute_arm_joints(target, base=BASE_POS):
    rel = target - base
    yaw = np.arctan2(rel[1], rel[0])

    shoulder = base + np.array([0.0, 0.0, BASE_H])
    dx = np.sqrt(rel[0]**2 + rel[1]**2)
    dz = rel[2] - BASE_H
    d  = np.clip(np.sqrt(dx**2 + dz**2), 0.05, L1 + L2 - 0.02)

    theta = np.arctan2(dz, dx)
    phi   = np.arccos(np.clip((L1**2 + d**2 - L2**2) / (2 * L1 * d), -1, 1))
    shoulder_angle = theta + phi

    elbow_r = L1 * np.cos(shoulder_angle)
    elbow_z = L1 * np.sin(shoulder_angle)
    elbow = shoulder + np.array([elbow_r * np.cos(yaw),
                                  elbow_r * np.sin(yaw),
                                  elbow_z])
    return np.array([base, shoulder, elbow, target], dtype=np.float32)

# =============================================================================
# transform helper — build 4x4 matrix aligning +Z with a direction and placing
# at a target position (used to move cylinders each frame)
# =============================================================================
def align_z_transform(from_point, to_point):
    """4x4 matrix: aligns a unit-height cylinder centered at origin with axis
    along +Z, to span from from_point to to_point."""
    from_point = np.asarray(from_point, dtype=np.float64)
    to_point   = np.asarray(to_point,   dtype=np.float64)
    vec        = to_point - from_point
    height     = np.linalg.norm(vec)
    if height < 1e-6:
        M = np.eye(4)
        M[:3, 3] = from_point
        return M

    direction = vec / height
    z_axis    = np.array([0.0, 0.0, 1.0])
    axis      = np.cross(z_axis, direction)
    axis_len  = np.linalg.norm(axis)

    if axis_len < 1e-8:
        R = np.eye(3) if direction[2] > 0 else np.diag([1, -1, -1])
    else:
        axis = axis / axis_len
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
        c, s  = np.cos(angle), np.sin(angle)
        x, y, z = axis
        R = np.array([
            [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
            [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
            [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)  ],
        ])

    S = np.diag([1.0, 1.0, height, 1.0])
    T = np.eye(4)
    T[:3, 3] = (from_point + to_point) / 2
    M = np.eye(4)
    M[:3, :3] = R
    M = T @ M @ S
    return M

def translate_transform(pos):
    M = np.eye(4)
    M[:3, 3] = pos
    return M

# =============================================================================
# BUILD ARM ONCE — unit primitives; move them via SetUserMatrix each frame
# =============================================================================
LINK_COLOR   = 'goldenrod'
JOINT_COLOR  = 'dimgray'
BASE_COLOR   = 'darkslategray'
NOZZLE_COLOR = 'lightgray'
TIP_COLOR    = 'cyan'

print("Setting up visualization...")

mesh_vis = pv.read(mesh_path).triangulate()
mesh_vis.cell_data['temperature'] = T_history[0]

plotter = pv.Plotter(off_screen=False, title='JAX-PULSE Robot Spray Cooling')
plotter.set_background('black')
plotter.add_text('Spray Cooling — Robot + Plate + JAX-PULSE Physics',
                  font_size=12, color='white')

ground = pv.Plane(center=(1.5, 1.5, -0.01), direction=(0, 0, 1),
                   i_size=6.0, j_size=6.0)
plotter.add_mesh(ground, color='#111111', show_edges=False)

heatmap_actor = plotter.add_mesh(
    mesh_vis, scalars='temperature', cmap='inferno',
    show_edges=False, clim=[T_ambient, T_initial],
    scalar_bar_args={'title': 'Temperature (°C)'}
)

# unit cylinder (height 1, along +Z, centered at origin) — will be transformed
unit_cyl = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1),
                        radius=1.0, height=1.0, resolution=24)
unit_sph = pv.Sphere(radius=1.0)

# base cylinder (short, fixed to floor) — never moves
base_geom = pv.Cylinder(center=BASE_POS + np.array([0, 0, 0.075]),
                         direction=(0, 0, 1),
                         radius=0.18, height=0.15, resolution=24)
plotter.add_mesh(base_geom, color=BASE_COLOR)

# reusable actors — each holds a unit primitive; we transform them per frame
def add_scaled_sphere(radius, color):
    poly = pv.Sphere(radius=radius)
    return plotter.add_mesh(poly, color=color)

def add_scaled_cylinder(radius, color):
    # unit-height cylinder scaled radially by `radius`; height scale is set
    # per-frame via SetUserMatrix
    poly = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1),
                       radius=radius, height=1.0, resolution=24)
    return plotter.add_mesh(poly, color=color)

# joint spheres (positioned via SetUserMatrix each frame — just translate)
shoulder_actor = add_scaled_sphere(0.13, JOINT_COLOR)
elbow_actor    = add_scaled_sphere(0.11, JOINT_COLOR)
wrist_actor    = add_scaled_sphere(0.08, JOINT_COLOR)
tip_actor      = add_scaled_sphere(0.05, TIP_COLOR)

# links (unit-height cylinders, stretched + rotated per frame)
upper_actor    = add_scaled_cylinder(0.09, LINK_COLOR)
forearm_actor  = add_scaled_cylinder(0.07, LINK_COLOR)

def set_actor_matrix(actor, M):
    """Apply a 4x4 numpy matrix to a vtk actor."""
    import vtk
    vtk_m = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_m.SetElement(i, j, M[i, j])
    actor.SetUserMatrix(vtk_m)

def update_arm(joints):
    base_pt, shoulder, elbow, tip = joints
    wrist_pos = tip + np.array([0, 0, 0.15])

    # spheres: just translate
    set_actor_matrix(shoulder_actor, translate_transform(shoulder))
    set_actor_matrix(elbow_actor,    translate_transform(elbow))
    set_actor_matrix(wrist_actor,    translate_transform(wrist_pos))
    set_actor_matrix(tip_actor,      translate_transform(tip))

    # cylinders: align + stretch
    set_actor_matrix(upper_actor,   align_z_transform(shoulder, elbow))
    set_actor_matrix(forearm_actor, align_z_transform(elbow, wrist_pos))

# =============================================================================
# nozzle geometry — cone tapering to orifice
# =============================================================================
NOZZLE_ORIFICE_R = 0.03   # spray exit hole radius
NOZZLE_BODY_R    = 0.06   # wider top of nozzle
NOZZLE_HEIGHT    = 0.15

def build_nozzle_geom(tip_pos):
    """Return a tapered cone nozzle: body_r at top, orifice_r at bottom (tip)."""
    top    = tip_pos + np.array([0, 0, NOZZLE_HEIGHT])
    bottom = tip_pos
    #two-cap frustum built via pyvista.Cylinder + tapered geometry
    #simplest: use pv.PolyData with a lofted frustum
    n_seg = 24
    theta = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    top_ring    = np.column_stack([
        top[0] + NOZZLE_BODY_R * np.cos(theta),
        top[1] + NOZZLE_BODY_R * np.sin(theta),
        np.full(n_seg, top[2])
    ])
    bottom_ring = np.column_stack([
        bottom[0] + NOZZLE_ORIFICE_R * np.cos(theta),
        bottom[1] + NOZZLE_ORIFICE_R * np.sin(theta),
        np.full(n_seg, bottom[2])
    ])
    points = np.vstack([top_ring, bottom_ring])
    faces = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        # side quad -> two triangles
        faces += [3, i, j, n_seg + j]
        faces += [3, i, n_seg + j, n_seg + i]
    return pv.PolyData(points, np.array(faces))

# =============================================================================
# droplet system — Fibonacci lattice + persistent falling motion
# =============================================================================
PLATE_Z            = 0.02
N_DROPLETS         = 300
DROPLET_SPEED      = 3.0    # m/s downward
SPRAY_RADIUS_PLATE = sigma * (z_height - PLATE_Z)

# fibonacci lattice for uniform disk sampling — computed once, reused
golden_angle = np.pi * (3 - np.sqrt(5))
_i = np.arange(N_DROPLETS)
_radial_frac = np.sqrt((_i + 0.5) / N_DROPLETS)     # uniform disk radius
_angles      = _i * golden_angle
_lattice_r_unit = _radial_frac                       # normalized to 0..1
_lattice_theta  = _angles

# each droplet has a persistent height (its "phase" of falling)
_droplet_heights = np.random.uniform(PLATE_Z, z_height, N_DROPLETS)

def advance_droplets(nozzle_pos, dt):
    """Move each droplet downward by dt*speed; respawn at nozzle when hitting plate."""
    global _droplet_heights
    _droplet_heights -= DROPLET_SPEED * dt
    #respawn droplets that hit the plate
    respawn_mask = _droplet_heights <= PLATE_Z
    _droplet_heights = np.where(respawn_mask, nozzle_pos[2], _droplet_heights)

    #radius grows linearly as droplet falls — spray cone shape
    z_frac = (nozzle_pos[2] - _droplet_heights) / (nozzle_pos[2] - PLATE_Z + 1e-8)
    r      = SPRAY_RADIUS_PLATE * z_frac * _lattice_r_unit

    x = nozzle_pos[0] + r * np.cos(_lattice_theta)
    y = nozzle_pos[1] + r * np.sin(_lattice_theta)
    return np.column_stack([x, y, _droplet_heights]).astype(np.float32)

# initial arm + droplets
init_joints = compute_arm_joints(np.array(waypoints[0]))
update_arm(init_joints)

# nozzle geometry actor — rebuild-per-frame is OK since it's ~50 triangles
nozzle_geom  = build_nozzle_geom(init_joints[3])
nozzle_geom_actor = plotter.add_mesh(nozzle_geom, color=NOZZLE_COLOR,
                                     smooth_shading=True)

droplets_poly = pv.PolyData(advance_droplets(init_joints[3], 0.0))
plotter.add_mesh(
    droplets_poly, color='cyan', point_size=6,
    render_points_as_spheres=True, opacity=0.85
)

plotter.camera_position = [(5.5, -3.5, 4.5), (1.5, 1.5, 0.3), (0, 0, 1)]
plotter.show(auto_close=False, interactive_update=True)

# =============================================================================
# animation loop
# =============================================================================
print("Animating...")
skip       = max(1, steps // 150)
frame_list = list(range(0, steps, skip))
frame_dt   = skip * dt_sim   # real elapsed time per frame in physics units

while True:
    for frame in frame_list:
        pidx   = int(pose_idx_history[frame])
        target = np.array(waypoints[pidx], dtype=np.float32)

        # plate temperature
        mesh_vis.cell_data['temperature'] = T_history[frame]
        heatmap_actor.mapper.dataset.cell_data['temperature'] = T_history[frame]
        heatmap_actor.mapper.dataset.Modified()

        # arm — matrix transforms only
        joints = compute_arm_joints(target)
        update_arm(joints)

        # nozzle body — cheap rebuild
        plotter.remove_actor(nozzle_geom_actor)
        nozzle_geom = build_nozzle_geom(joints[3])
        nozzle_geom_actor = plotter.add_mesh(nozzle_geom, color=NOZZLE_COLOR,
                                             smooth_shading=True)

        # droplets — deterministic lattice + persistent falling motion
        droplets_poly.points = advance_droplets(joints[3], frame_dt * 0.3)

        plotter.render()