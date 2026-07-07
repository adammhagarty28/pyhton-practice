"""
POC: Raycasting Spray Distribution using PULSE + NVIDIA Warp

Loads a real 3D mesh (refined_plate.obj) and simulates spray deposition
from a zigzag nozzle path above the surface. Uses PULSE raycasting to
determine which triangles get hit and how much spray lands on each one.
Visualizes spray thickness distribution on the 3D mesh using pyvista.

No heat equation yet — this establishes spray distribution on real 3D
geometry before coupling to thermal simulation.
"""

import numpy as np
import warp as wp
import pyvista as pv

from pulse.lib.pulse import Pulse, PulseParams
from pulse.lib.warp_kernels import Pose
from pulse.utils.utils import get_project_mesh_dict

# =============================================================================
# load mesh
# =============================================================================
print("Loading mesh...")
mesh_dict = get_project_mesh_dict()
mesh_path = str(mesh_dict["refined_plate"]["obj_path"])
print(f"Mesh path: {mesh_path}")

params = PulseParams(
    sigma_u=0.15,
    sigma_v=0.15,
    ref_dist=1.0,
    samples_per_pulse=1000,
    volumetric_flow_rate=1.0e-5
)

p = Pulse(params=params)
p.load_mesh(mesh_path)
mesh_check = pv.read(mesh_path).triangulate()
print(f"Mesh bounds: {mesh_check.bounds}")
print(f"Mesh center: {mesh_check.center}")
print(f"Number of faces: {mesh_check.n_faces}")
print("Mesh loaded.")

# =============================================================================
# define zigzag nozzle path above the plate
# =============================================================================
# refined_plate sits roughly in -0.15 to 0.15 range in x and y
# nozzle is 0.1m above the plate (z=0.1)
z_height  = 1.0
x_range = np.linspace(0.3, 2.7, 20)
y_range = np.linspace(0.3, 2.7, 20)

poses = []
for i, y in enumerate(y_range):
    xs = x_range if i % 2 == 0 else x_range[::-1]
    for x in xs:
        pose = Pose()
        pose.position = wp.vec3(x, y, z_height)
        pose.rotation = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), wp.pi)
        poses.append(pose)

print(f"Total nozzle poses: {len(poses)}")

# =============================================================================
# evaluate spray distribution
# =============================================================================
print("Running raycasting...")
poses_wp   = wp.array(poses, dtype=Pose)
thicknesses = p.evaluate_pulses(poses_wp)
scaled      = p.get_scaled_thicknesses(thicknesses)
print("Done.")
print(f"Max thickness: {scaled.max():.6f}")
print(f"Min thickness: {scaled.min():.6f}")
print(f"Mean thickness: {scaled.mean():.6f}")
print(f"Std deviation: {scaled.std():.6f}")

# =============================================================================
# visualize spray distribution on mesh
# =============================================================================
print("Visualizing...")
mesh_pv = pv.read(mesh_path).triangulate()

# assign thickness to each triangle face
mesh_pv.cell_data["spray_thickness"] = scaled

plotter = pv.Plotter()
plotter.add_mesh(
    mesh_pv,
    scalars="spray_thickness",
    cmap="plasma",
    show_edges=False,
    scalar_bar_args={"title": "Spray Thickness"}
)

# add nozzle path as a line
path_points = np.array([[pose.position[0], pose.position[1], pose.position[2]]
                         for pose in poses])
path_pv = pv.Spline(path_points, 200)
plotter.add_mesh(path_pv, color="cyan", line_width=3, label="Nozzle path")

plotter.add_legend()
plotter.set_background("black")
plotter.add_title("Raycasting: Spray Distribution on Refined Plate")
plotter.show()