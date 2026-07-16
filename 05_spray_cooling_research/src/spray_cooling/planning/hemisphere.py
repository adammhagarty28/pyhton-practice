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
    points_per_ring: int | np.ndarray,
    polar_min_deg: float,
    polar_max_deg: float,
    standoff: float,
    azimuth_center_rad: float | None = None,
    azimuth_half_span_deg: float | None = None,
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

    if np.isscalar(points_per_ring):
        ring_point_counts = np.full(
            ring_count,
            int(points_per_ring),
            dtype=int,
        )
    else:
        ring_point_counts = np.asarray(
            points_per_ring,
            dtype=int,
        )

        if ring_point_counts.shape != (ring_count,):
            raise ValueError(
                "Variable points_per_ring must contain exactly "
                f"{ring_count} entries."
            )

    minimum_points = (
        2
        if azimuth_half_span_deg is not None
        else 4
    )

    if np.any(ring_point_counts < minimum_points):
        raise ValueError(
            "Every ring must contain at least "
            f"{minimum_points} points."
        )

    if (
        azimuth_center_rad is None
    ) != (
        azimuth_half_span_deg is None
    ):
        raise ValueError(
            "azimuth_center_rad and azimuth_half_span_deg "
            "must either both be provided or both be omitted."
        )

    surface_points: list[np.ndarray] = []
    surface_normals: list[np.ndarray] = []
    tool_positions: list[np.ndarray] = []
    pulse_rotations: list[np.ndarray] = []

    for ring_index, polar in enumerate(polar_angles):
        point_count = int(
            ring_point_counts[ring_index]
        )

        if azimuth_half_span_deg is None:
            azimuths = np.linspace(
                0.0,
                2.0 * np.pi,
                point_count,
                endpoint=False,
            )
        else:
            half_span_rad = np.deg2rad(
                float(azimuth_half_span_deg)
            )

            if not 0.0 < half_span_rad <= np.pi:
                raise ValueError(
                    "azimuth_half_span_deg must be in (0, 180]."
                )

            # Slightly inset the endpoints so a later numerical
            # accessibility check does not reject points exactly on
            # the angular boundary.
            angular_epsilon = 1.0e-8

            azimuths = np.linspace(
                float(azimuth_center_rad)
                - half_span_rad
                + angular_epsilon,
                float(azimuth_center_rad)
                + half_span_rad
                - angular_epsilon,
                point_count,
                endpoint=True,
            )

        # Alternate direction to keep successive rings spatially connected.
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
