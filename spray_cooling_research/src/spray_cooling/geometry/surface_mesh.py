"""
Reusable triangular surface-mesh loading and geometric preprocessing.

This module knows nothing about a flat plate, half-sphere, robot, spray,
thermal solver, or visualization. It converts a triangular surface into a
common representation used by the rest of the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv


class SurfaceMeshError(ValueError):
    """Raised when a surface mesh cannot be prepared safely."""


@dataclass
class SurfaceMesh:
    """Geometry shared by planning, spray deposition, physics, and display."""

    source_path: Path
    runtime_path: Path | None
    polydata: pv.PolyData

    points: np.ndarray
    faces: np.ndarray

    face_v0: np.ndarray
    face_v1: np.ndarray
    face_v2: np.ndarray

    face_centers: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray

    scale_factor: float

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return tuple(float(value) for value in self.polydata.bounds)


def load_surface_mesh(
    source_path: str | Path,
    *,
    target_x_extent_m: float | None = None,
    center_m: tuple[float, float, float] | np.ndarray | None = None,
    runtime_path: str | Path | None = None,
) -> SurfaceMesh:
    """
    Load and prepare a triangular surface.

    Parameters
    ----------
    source_path:
        Original OBJ/STL/PLY or other PyVista-readable mesh.

    target_x_extent_m:
        Optional desired world-frame x extent. The entire mesh is scaled
        uniformly, preserving its shape and aspect ratios.

    center_m:
        Optional world-frame location to which the mesh center is translated.

    runtime_path:
        Optional location at which the transformed triangular mesh is written.
        JAX-PULSE currently requires a mesh file on disk, so the flat-plate
        application uses this path.

    Notes
    -----
    Uniform scaling and recentering work for both the plate and half-sphere.
    Geometry-specific path planning belongs elsewhere.
    """
    source = Path(source_path).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Surface mesh does not exist: {source}")

    mesh = pv.read(source).triangulate()

    if mesh.n_points < 3:
        raise SurfaceMeshError(
            f"Surface mesh has too few points: {mesh.n_points}"
        )

    if mesh.n_cells < 1:
        raise SurfaceMeshError("Surface mesh contains no triangular cells.")

    face_data = np.asarray(mesh.faces)

    if face_data.size % 4 != 0:
        raise SurfaceMeshError(
            "Triangulated PyVista face data did not contain 4 entries per face."
        )

    face_rows = face_data.reshape(-1, 4)

    if not np.all(face_rows[:, 0] == 3):
        raise SurfaceMeshError(
            "The prepared surface contains non-triangular cells."
        )

    scale_factor = 1.0

    if target_x_extent_m is not None:
        target_x_extent_m = float(target_x_extent_m)

        if target_x_extent_m <= 0.0:
            raise SurfaceMeshError(
                "target_x_extent_m must be greater than zero."
            )

        bounds = mesh.bounds
        original_x_extent = float(bounds[1] - bounds[0])

        if original_x_extent <= 0.0:
            raise SurfaceMeshError(
                "The source mesh has zero x extent and cannot be scaled."
            )

        scale_factor = target_x_extent_m / original_x_extent
        mesh.points = np.asarray(mesh.points) * scale_factor

    if center_m is not None:
        requested_center = np.asarray(center_m, dtype=float)

        if requested_center.shape != (3,):
            raise SurfaceMeshError(
                "center_m must contain exactly three coordinates."
            )

        mesh.points = (
            np.asarray(mesh.points)
            - np.asarray(mesh.center)
            + requested_center
        )

    output_path: Path | None = None

    if runtime_path is not None:
        output_path = Path(runtime_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Weld coincident OBJ vertices before geometric preprocessing.
        #
        # Some OBJ files store each triangle with its own three vertex IDs,
        # even when neighboring triangles occupy identical coordinates.
        # Without welding, shared-edge adjacency and thermal conduction fail.
        cell_count_before_cleaning = mesh.n_cells
        point_count_before_cleaning = mesh.n_points

        mesh = mesh.clean(
            point_merging=True,
            tolerance=1.0e-10,
            absolute=True,
        ).triangulate()

        if mesh.n_cells != cell_count_before_cleaning:
            raise RuntimeError(
                "Mesh cleaning unexpectedly changed the triangle count: "
                f"{cell_count_before_cleaning} -> {mesh.n_cells}"
            )

        print(
            "Mesh topology:"
            f" welded {point_count_before_cleaning} points "
            f"to {mesh.n_points} shared points"
        )

        mesh.save(output_path)

    points = np.asarray(mesh.points, dtype=float).copy()
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:].astype(
        np.int32,
        copy=True,
    )

    face_v0 = points[faces[:, 0]]
    face_v1 = points[faces[:, 1]]
    face_v2 = points[faces[:, 2]]

    cross_products = np.cross(
        face_v1 - face_v0,
        face_v2 - face_v0,
    )

    twice_area = np.linalg.norm(cross_products, axis=1)
    face_areas = 0.5 * twice_area

    if np.any(face_areas <= 1.0e-14):
        bad_count = int(np.count_nonzero(face_areas <= 1.0e-14))
        raise SurfaceMeshError(
            f"Surface contains {bad_count} degenerate triangular faces."
        )

    face_normals = cross_products / twice_area[:, None]
    face_centers = (face_v0 + face_v1 + face_v2) / 3.0

    return SurfaceMesh(
        source_path=source,
        runtime_path=output_path,
        polydata=mesh,
        points=points,
        faces=faces,
        face_v0=face_v0,
        face_v1=face_v1,
        face_v2=face_v2,
        face_centers=face_centers,
        face_normals=face_normals,
        face_areas=face_areas,
        scale_factor=scale_factor,
    )
