"""Shared functionality extracted from the geometry demos."""

from __future__ import annotations

import numpy as np
import os


def nearest_surface_point_and_normal(face_centers_np, face_normals_np, nozzle_world):
    """
    Return nearest mesh face center and outward normal relative to nozzle_world.

    For the flat plate:
      surface point ~= [x, y, 0]
      normal ~= [0, 0, 1]

    For future curved geometry:
      normal becomes the local triangle normal, flipped so it points toward the nozzle.
    """
    nozzle_world = np.array(nozzle_world, dtype=float)
    diff = face_centers_np - nozzle_world[None, :]
    idx = int(np.argmin(np.sum(diff * diff, axis=1)))
    p_surf = face_centers_np[idx].copy()
    n = face_normals_np[idx].copy()
    n = n / (np.linalg.norm(n) + 1e-08)
    to_nozzle = nozzle_world - p_surf
    if np.dot(n, to_nozzle) < 0.0:
        n = -n
    return (p_surf, n)


def resolve_mesh_path(urdf_dir, filename_attr):
    """Convert URDF mesh filename to an absolute path on disk."""
    if filename_attr.startswith('package://'):
        stripped = filename_attr[len('package://'):]
        parts = stripped.split('/', 1)
        if len(parts) == 2:
            from robot_descriptions import ur5e_description as ur_mod
            candidate = os.path.join(ur_mod.REPOSITORY_PATH, parts[1])
            if os.path.exists(candidate):
                return candidate
            target_name = os.path.basename(parts[1])
            for r, _, files in os.walk(ur_mod.REPOSITORY_PATH):
                if target_name in files:
                    return os.path.join(r, target_name)
    elif filename_attr.startswith('file://'):
        return filename_attr[len('file://'):]
    else:
        candidate = os.path.join(urdf_dir, filename_attr)
        if os.path.exists(candidate):
            return candidate
    return None
