"""Curved spray path generation for a hemispherical shell."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HemispherePath:
    """Discrete surface, tool, normal, and JAX-PULSE pose data."""

    surface_points: np.ndarray
    surface_normals: np.ndarray
    tool_positions: np.ndarray
    pulse_quaternions_xyzw: np.ndarray


def _quaternion_plus_z_to(direction: np.ndarray) -> np.ndarray:
    """
    Return an XYZW quaternion rotating local +Z onto `direction`.

    JAX-PULSE's original flat-plate quaternion [1, 0, 0, 0]
    rotates +Z onto world -Z.
    """
    target = np.asarray(direction, dtype=float)
    target /= np.linalg.norm(target) + 1.0e-12

    source = np.array([0.0, 0.0, 1.0], dtype=float)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if dot > 1.0 - 1.0e-10:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    if dot < -1.0 + 1.0e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    xyz = np.cross(source, target)

    quaternion = np.array(
        [xyz[0], xyz[1], xyz[2], 1.0 + dot],
        dtype=float,
    )

    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def build_hemisphere_path(
    *,
    sphere_center: np.ndarray,
    sphere_radius: float,
    ring_count: int,
    points_per_ring: int,
    polar_min_deg: float,
    polar_max_deg: float,
    standoff: float,
) -> HemispherePath:
    """
    Build alternating circular passes over a hemispherical dome.

    polar angle:
        0 degrees  = dome apex
        90 degrees = dome rim

    tool position:
        surface point + standoff * outward normal

    spray direction:
        inward, opposite the outward surface normal
    """
    if ring_count < 2:
        raise ValueError("ring_count must be at least 2.")

    if points_per_ring < 4:
        raise ValueError("points_per_ring must be at least 4.")

    if not 0.0 <= polar_min_deg < polar_max_deg < 90.0:
        raise ValueError(
            "Polar angles must satisfy "
            "0 <= minimum < maximum < 90 degrees."
        )

    center = np.asarray(sphere_center, dtype=float)
    radius = float(sphere_radius)
    standoff = float(standoff)

    polar_angles = np.linspace(
        np.deg2rad(polar_min_deg),
        np.deg2rad(polar_max_deg),
        ring_count,
    )

    surface_points: list[np.ndarray] = []
    surface_normals: list[np.ndarray] = []
    tool_positions: list[np.ndarray] = []
    pulse_rotations: list[np.ndarray] = []

    for ring_index, polar in enumerate(polar_angles):
        azimuths = np.linspace(
            0.0,
            2.0 * np.pi,
            points_per_ring,
            endpoint=False,
        )

        # Alternating direction prevents a full-circle jump between rings.
        if ring_index % 2 == 1:
            azimuths = azimuths[::-1]

        for azimuth in azimuths:
            normal = np.array(
                [
                    np.sin(polar) * np.cos(azimuth),
                    np.sin(polar) * np.sin(azimuth),
                    np.cos(polar),
                ],
                dtype=float,
            )

            surface_point = center + radius * normal
            tool_position = surface_point + standoff * normal

            surface_points.append(surface_point)
            surface_normals.append(normal)
            tool_positions.append(tool_position)

            # PULSE local +Z is aligned with the inward spray direction.
            pulse_rotations.append(
                _quaternion_plus_z_to(-normal)
            )

    return HemispherePath(
        surface_points=np.asarray(surface_points),
        surface_normals=np.asarray(surface_normals),
        tool_positions=np.asarray(tool_positions),
        pulse_quaternions_xyzw=np.asarray(pulse_rotations),
    )
