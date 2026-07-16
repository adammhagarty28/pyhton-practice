"""Verify PyVista can render UR5e meshes from the URDF."""
from robot_descriptions.loaders.yourdfpy import load_robot_description
import pyvista as pv
import numpy as np

urdf = load_robot_description("ur5e_description")

joint_config = {
    "shoulder_pan_joint":  0.0,
    "shoulder_lift_joint": -1.2,
    "elbow_joint":          1.5,
    "wrist_1_joint":       -1.5,
    "wrist_2_joint":       -1.5,
    "wrist_3_joint":        0.0,
}
urdf.update_cfg(joint_config)

plotter = pv.Plotter(lighting='three lights')
plotter.set_background('#1a1a1a')

# ground plane for context
plotter.add_mesh(
    pv.Plane(center=(0, 0, -0.001), i_size=2, j_size=2),
    color='#333333'
)

# scene.dump() returns every mesh already in world coordinates
transformed_meshes = urdf.scene.dump()
print(f"Total meshes in scene: {len(transformed_meshes)}")

for i, tm in enumerate(transformed_meshes):
    if not hasattr(tm, 'vertices') or len(tm.vertices) == 0:
        continue

    verts = np.array(tm.vertices)
    faces_flat = np.hstack(
        [np.full((len(tm.faces), 1), 3, dtype=np.int64), tm.faces]
    ).flatten()
    pv_mesh = pv.PolyData(verts, faces_flat)

    plotter.add_mesh(
        pv_mesh,
        color='#4a90d9',
        smooth_shading=True,
        specular=0.5,
        specular_power=15,
        ambient=0.2,
    )
    print(f"added mesh {i}: {len(verts)} verts, {len(tm.faces)} faces")

plotter.camera_position = 'iso'
plotter.show()