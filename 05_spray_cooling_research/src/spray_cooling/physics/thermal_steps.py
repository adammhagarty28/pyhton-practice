"""
Thermal time-stepping kernels for the spray-cooling POC family.

POC 3 volume-mesh physics: IMEX backward Euler on a 3D prism-cell adjacency
graph. Physics module lifted from POC 2 (pulse_spray_surface_and_volume_mesh)
and adapted for face-type-dependent boundary conditions on real UR5e geometry.

Governing equation (per cell i):
    rho c_p V_i dT_i/dt = sum_j G_ij (T_j - T_i)
                          - h_face,i A_face,i (T_i - T_inf)
                          - eps_i sigma A_exposed,i (T_i^4 - T_amb^4)

Divided by (rho c_p V_i) to get dT/dt in K/s.
Temperatures in Kelvin internally. All coefficient fields lagged at T_old
so the implicit solve stays linear (matrix-free conjugate gradient).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Volume-mesh IMEX backward-Euler step factory (POC 3 main)
# ---------------------------------------------------------------------------

def make_flat_plate_step(
    *,
    # Spray footprint history (per-pose, per-surface-triangle)
    spray_masks_surf,           # [n_poses, n_surface] in [0, 1]
    steps_per_move,             # int
    n_waypoints,                # int
    n_surface,                  # int (top-face triangles)
    n_cells,                    # int (n_surface * n_layers)
    # 3D adjacency
    neighbors_3d,               # [n_cells, MAX_NB_3D] int, -1 = invalid
    G_matrix,                   # [n_cells, MAX_NB_3D] float, A_ij/(d_ij V_i) [1/m^2]
    # Per-cell geometry / masks
    cell_volume,                # [n_cells] [m^3]
    face_area_top,              # [n_cells] [m^2] (only non-zero on top layer)
    face_area_bottom,           # [n_cells]
    face_area_side,             # [n_cells]
    is_top_mask,                # [n_cells] 1.0 on top layer, 0 elsewhere
    surface_areas,              # [n_surface] [m^2] (for hysteresis zone average)
    # Material property tables (T-dependent)
    temp_table_k,               # [n_props] [K]
    k_table_w_mk,               # [n_props] [W/(m K)]
    cp_table_j_kgk,             # [n_props] [J/(kg K)]
    emissivity_table,           # [n_props] [-]
    density_kg_m3,              # scalar [kg/m^3]
    stefan_boltzmann,           # scalar [W/(m^2 K^4)]
    # HTCs
    h_ambient_w_m2k,            # scalar [W/(m^2 K)]
    boiling_temp_table_c,       # [n_boil] [C]
    boiling_h_table_w_m2k,      # [n_boil] [W/(m^2 K)]
    # Ambient / target
    ambient_temperature_c,      # scalar [C]
    # Time integration
    dt_s,                       # scalar [s]
    # Adaptive hysteresis
    delta_off_fraction,         # scalar [-]
    delta_on_fraction,          # scalar [-]
    # S-curve valve
    spray_ramp_tau_s,           # scalar [s]
):
    """Return a jitted step function for a volume-mesh IMEX BE integrator.

    Carry:
        (T_K, step_idx, valve_target, spray_stage1, spray_actual)
    Emits per-step:
        (T_new_C, pose_idx, spray_actual, T_top_C, T_bot_C)
    """
    # Convert scalars/arrays to JAX
    T_amb_K = jnp.float64(ambient_temperature_c + 273.15)
    T_table_K_j = jnp.array(temp_table_k)
    K_table_j   = jnp.array(k_table_w_mk)
    CP_table_j  = jnp.array(cp_table_j_kgk)
    EPS_table_j = jnp.array(emissivity_table)
    boil_T_C_j  = jnp.array(boiling_temp_table_c)
    boil_H_j    = jnp.array(boiling_h_table_w_m2k)
    surface_areas_j = jnp.array(surface_areas)

    n_layers = n_cells // n_surface

    def k_of_T(T):     return jnp.interp(T, T_table_K_j, K_table_j)
    def cp_of_T(T):    return jnp.interp(T, T_table_K_j, CP_table_j)
    def eps_of_T(T):   return jnp.interp(T, T_table_K_j, EPS_table_j)
    def h_spray_of_T(T_K):
        return jnp.interp(T_K - 273.15, boil_T_C_j, boil_H_j)

    # -----------------------------------------------------------------------
    # 3D graph Laplacian and IMEX BE step
    # -----------------------------------------------------------------------
    def volume_laplacian(T, neighbors_arr, G_mat):
        def gather_col(nb_col, G_col):
            safe = jnp.maximum(nb_col, 0)
            T_nb = jnp.where(nb_col >= 0, T[safe], T)
            return G_col * (T_nb - T)
        contribs = jax.vmap(gather_col, in_axes=1, out_axes=0)(neighbors_arr, G_mat)
        return jnp.sum(contribs, axis=0)

    def linear_operator(T, alpha_field, beta_field):
        return T - dt_s * alpha_field * volume_laplacian(T, neighbors_3d, G_matrix) + dt_s * beta_field * T

    def imex_step(T_old, alpha_field, beta_field, gamma_field):
        rad = gamma_field * (T_old**4 - T_amb_K**4)
        rhs = T_old + dt_s * beta_field * T_amb_K - dt_s * rad
        A   = lambda T: linear_operator(T, alpha_field, beta_field)
        T_new, _ = jax.scipy.sparse.linalg.cg(A, rhs, x0=T_old, tol=1e-8, maxiter=400)
        return T_new

    def step(carry, _):
        T_old, step_idx, valve_target, spray_stage1, spray_actual = carry

        # ----- Choose current spray footprint from waypoint schedule -----
        pose_idx        = jnp.minimum(step_idx // steps_per_move, n_waypoints - 1)
        spray_mask_surf = spray_masks_surf[pose_idx]                # [n_surface]

        # ----- Adaptive hysteresis on top-layer sprayed zone -----
        T_top       = T_old[:n_surface]
        mask_weight = spray_mask_surf * surface_areas_j
        mask_wsum   = jnp.sum(mask_weight) + 1e-12
        T_zone      = jnp.sum(T_top * mask_weight) / mask_wsum
        T_plate_avg = jnp.sum(T_old * cell_volume) / jnp.sum(cell_volume)

        headroom  = jnp.maximum(T_plate_avg - T_amb_K, 1.0)
        delta_off = delta_off_fraction * headroom
        delta_on  = delta_on_fraction  * headroom
        has_zone  = (mask_wsum > 1e-6).astype(jnp.float64)
        turn_off  = valve_target         * (T_zone < T_plate_avg - delta_off).astype(jnp.float64) * has_zone
        turn_on   = (1.0 - valve_target) * (T_zone > T_plate_avg - delta_on ).astype(jnp.float64) * has_zone
        valve_target_new = valve_target - turn_off + turn_on

        # ----- S-curve valve (two cascaded lags) + smoothstep h response -----
        alpha_ramp        = dt_s / (spray_ramp_tau_s + dt_s)
        spray_stage1_new  = spray_stage1 + alpha_ramp * (valve_target_new - spray_stage1)
        spray_actual_new  = spray_actual + alpha_ramp * (spray_stage1_new - spray_actual)
        s_clipped         = jnp.clip(spray_actual_new, 0.0, 1.0)
        spray_smooth      = s_clipped * s_clipped * (3.0 - 2.0 * s_clipped)

        # ----- T-dependent material properties (lagged at T_old) -----
        k_field   = k_of_T(T_old)
        cp_field  = cp_of_T(T_old)
        eps_field = eps_of_T(T_old)
        rhocp     = density_kg_m3 * cp_field
        alpha_fld = k_field / rhocp

        # ----- Boiling curve h(T_top) evaluated at TRUE top-layer T -----
        h_spray_field = h_spray_of_T(T_old)

        # Extend surface spray mask across all cells (nonzero only on top layer)
        spray_mask_full = jnp.concatenate([
            spray_mask_surf,
            jnp.zeros(n_cells - n_surface, dtype=spray_mask_surf.dtype),
        ])

        # Per-face h on top: spray-weighted plus ambient background
        h_top_face = (spray_mask_full * spray_smooth * (h_spray_field - h_ambient_w_m2k)
                      + h_ambient_w_m2k) * is_top_mask

        beta_top    = h_top_face      * face_area_top    / (rhocp * cell_volume)
        beta_bottom = h_ambient_w_m2k * face_area_bottom / (rhocp * cell_volume)
        beta_side   = h_ambient_w_m2k * face_area_side   / (rhocp * cell_volume)
        beta_field  = beta_top + beta_bottom + beta_side

        A_rad     = face_area_top + face_area_bottom + face_area_side
        gamma_fld = eps_field * stefan_boltzmann * A_rad / (rhocp * cell_volume)

        # ----- IMEX BE solve -----
        T_new = imex_step(T_old, alpha_fld, beta_field, gamma_fld)

        # ----- Emit per-step diagnostics -----
        T_new_C = T_new - 273.15
        T_top_C = T_new_C[:n_surface]
        T_bot_C = T_new_C[(n_layers - 1) * n_surface :]

        return (T_new, step_idx + 1, valve_target_new, spray_stage1_new, spray_actual_new), \
               (T_new_C, pose_idx, spray_actual_new, T_top_C, T_bot_C)

    return step


# ---------------------------------------------------------------------------
# Legacy thin-shell step (kept for half_sphere demo)
# ---------------------------------------------------------------------------

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
        pose_index = jnp.minimum(step_index // steps_per_move, n_waypoints - 1)
        h_field = h_fields[pose_index]

        def thermal_substep(_, temperature_sub):
            edge_dT = temperature_sub[edge_j] - temperature_sub[edge_i]
            edge_power = edge_conductance * edge_dT
            conductive_power = jnp.zeros_like(temperature_sub)
            conductive_power = conductive_power.at[edge_i].add(edge_power)
            conductive_power = conductive_power.at[edge_j].add(-edge_power)
            convective_power = h_field * face_area * (temperature_sub - ambient_temperature_c)
            net_power = conductive_power - convective_power
            return temperature_sub + thermal_substep_dt * net_power / thermal_capacity

        temperature_new = jax.lax.fori_loop(0, thermal_substeps, thermal_substep, temperature)
        return (temperature_new, step_index + 1), (temperature_new, pose_index)

    return step
