"""Shared JAX-PULSE spray influence model."""

from __future__ import annotations

from relevant_PULSE_files.jax_kernels import deposit


def make_compute_h_for_pose(
    *,
    a,
    face_normals,
    face_v0,
    face_v1,
    face_v2,
    fov,
    h_ambient,
    h_spray_scale,
    n_faces,
    ref_dist,
    resolution,
    sigma,
):
    """Create a pose-to-heat-transfer-field function for one fixed surface."""

    def compute_h_for_pose(pos, rot):
        weight = deposit(
            pos,
            rot,
            sigma,
            a,
            ref_dist,
            resolution,
            face_v0,
            face_v1,
            face_v2,
            face_normals,
            n_faces,
            fov,
        )

        return (
            h_ambient
            + weight * h_spray_scale
        )

    return compute_h_for_pose
