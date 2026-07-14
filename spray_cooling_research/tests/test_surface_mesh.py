from pathlib import Path

import numpy as np

from spray_cooling.geometry.surface_mesh import load_surface_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_flat_plate_geometry() -> None:
    output_path = (
        PROJECT_ROOT
        / "artifacts"
        / "generated"
        / "_surface_mesh_test.obj"
    )

    surface = load_surface_mesh(
        PROJECT_ROOT / "data" / "meshes" / "refined_plate.obj",
        target_x_extent_m=0.4,
        center_m=(0.6, 0.0, -0.15),
        runtime_path=output_path,
    )

    try:
        assert surface.n_faces == 16384

        x_extent = surface.bounds[1] - surface.bounds[0]
        assert np.isclose(x_extent, 0.4, atol=1.0e-9)

        assert np.all(surface.face_areas > 0.0)
        assert np.allclose(
            np.linalg.norm(surface.face_normals, axis=1),
            1.0,
            atol=1.0e-7,
        )

        assert np.allclose(
            np.asarray(surface.polydata.center),
            np.array([0.6, 0.0, -0.15]),
            atol=1.0e-8,
        )
    finally:
        output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_flat_plate_geometry()
    print("PASS: flat-plate surface geometry")
