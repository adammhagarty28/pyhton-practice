"""Shared thermal time-stepping kernels.

The flat-plate and half-sphere benchmarks currently use different
thermal spatial discretizations. Their existing behavior is preserved
here while moving thermal integration out of the geometry demos.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def make_flat_plate_step(
    *,
    h_fields,
    steps_per_move,
    n_waypoints,
    neighbors,
    conductance,
    thermal_mass,
    volumetric_heat_capacity,
    thickness_m,
    ambient_temperature_c,
    dt_s,
):
    def step(carry, _):
        T, step_idx = carry
        pose_idx = jnp.minimum(
            step_idx // steps_per_move,
            n_waypoints - 1,
        )
        h_field = h_fields[pose_idx]

        safe_neighbors = jnp.where(
            neighbors >= 0,
            neighbors,
            0,
        )
        T_neighbors = T[safe_neighbors]
        valid = (
            neighbors >= 0
        ).astype(jnp.float32)

        conductive_power = jnp.sum(
            conductance
            * valid
            * (T_neighbors - T[:, None]),
            axis=1,
        )

        dTdt_conduction = (
            conductive_power
            / thermal_mass
        )

        dTdt_spray = -(
            h_field
            / (
                volumetric_heat_capacity
                * thickness_m
            )
        ) * (
            T - ambient_temperature_c
        )

        T_new = (
            T
            + (
                dTdt_conduction
                + dTdt_spray
            )
            * dt_s
        )

        T_new = jnp.maximum(
            T_new,
            ambient_temperature_c,
        )

        return (
            T_new,
            step_idx + 1,
        ), (
            T_new,
            pose_idx,
        )

    return step


def make_conservative_thin_shell_step(
    *,
    h_fields,
    steps_per_move,
    n_waypoints,
    edge_i,
    edge_j,
    edge_conductance,
    face_area,
    thermal_capacity,
    ambient_temperature_c,
    thermal_substeps,
    thermal_substep_dt,
):
    def step(carry, _):
        temperature, step_index = carry

        pose_index = jnp.minimum(
            step_index // steps_per_move,
            n_waypoints - 1,
        )

        h_field = h_fields[pose_index]

        def thermal_substep(
            _,
            temperature_sub,
        ):
            edge_temperature_difference = (
                temperature_sub[edge_j]
                - temperature_sub[edge_i]
            )

            edge_power = (
                edge_conductance
                * edge_temperature_difference
            )

            conductive_power = (
                jnp.zeros_like(
                    temperature_sub
                )
            )

            conductive_power = (
                conductive_power.at[
                    edge_i
                ].add(edge_power)
            )

            conductive_power = (
                conductive_power.at[
                    edge_j
                ].add(-edge_power)
            )

            convective_power = (
                h_field
                * face_area
                * (
                    temperature_sub
                    - ambient_temperature_c
                )
            )

            net_power = (
                conductive_power
                - convective_power
            )

            temperature_next = (
                temperature_sub
                + thermal_substep_dt
                * net_power
                / thermal_capacity
            )

            return temperature_next

        temperature_new = (
            jax.lax.fori_loop(
                0,
                thermal_substeps,
                thermal_substep,
                temperature,
            )
        )

        return (
            temperature_new,
            step_index + 1,
        ), (
            temperature_new,
            pose_index,
        )

    return step
