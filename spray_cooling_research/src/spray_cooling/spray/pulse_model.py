"""Shared functionality extracted from the geometry demos."""

from __future__ import annotations

from relevant_PULSE_files.jax_kernels import Pose, deposit


def compute_h_for_pose(a, face_normals, face_v0, face_v1, face_v2, fov, h_ambient, h_spray_scale, n_faces, ref_dist, resolution, sigma, pos, rot):
    weight = deposit(pos, rot, sigma, a, ref_dist, resolution, face_v0, face_v1, face_v2, face_normals, n_faces, fov)
    return h_ambient + weight * h_spray_scale
