"""
POC 1 - Pulsed Spray Cooling on a 2D Rectangular Grid (Flat Plate)

WHAT THIS FILE IS
    A 2D finite-difference simulation of pulsed water-spray cooling on a flat
    steel plate. A stationary Gaussian spray footprint sits at the plate center.
    The spray turns on and off adaptively based on plate temperature, producing
    the sawtooth thermal history that characterizes real pulsed quenching:
    sharp drops while spraying, partial rebounds while off as heat conducts in
    from surrounding hot material.

GOVERNING EQUATION
    rho c_p dT/dt = k grad^2(T) - (h_field/L)(T - T_inf) - (eps sigma/L)(T^4 - T_amb_rad^4)

    Divided by (rho c_p):
        dT/dt = alpha grad^2(T) - beta_field (T - T_inf) - gamma (T^4 - T_amb^4)

    where alpha = k/(rho c_p), beta = h/(rho c_p L), gamma = eps sigma/(rho c_p L).
    Temperatures are in Kelvin internally (required for the T^4 radiation term)
    and converted to Celsius only for display.

PHYSICAL MECHANISMS MODELED
    1. Conduction (k grad^2 T)
       Heat spreading through the plate. This is what causes the REBOUND during
       spray-off periods: cold zone under the spray pulls heat in from the
       surrounding hot plate material. Without conduction, temperature would
       be flat during off-periods. This is the signature Mike asked to see.

    2. Spray convection (h_spray * (T - T_inf))
       Direct heat removal by the water spray, dominant when spray is on.
       Represented as a Gaussian footprint (spray_mask) centered at the plate
       middle, scaled by spray intensity in [0, 1].

    3. Ambient convection (h_ambient * (T - T_inf))
       Natural convection to still air (~15 W/m^2 K), active everywhere on the
       plate at all times. Small compared to spray but non-zero.

    4. Radiation (eps sigma (T^4 - T_amb^4))
       Blackbody radiation to surroundings. Significant at 900 C (~85 kW/m^2),
       negligible below ~200 C. Cools the plate uniformly, spray-independent.
       Especially important in regions the spray never touches.

    5. Temperature-dependent material properties (k(T), c_p(T), eps(T))
       Real steel properties vary strongly across 25-900 C. k drops by ~2x,
       c_p rises by ~1.5x, emissivity rises slightly. Interpolated from
       Incropera Table A.1 (AISI 1010, closest tabulated proxy to AISI 1045).
       Without this, conduction and heat capacity would be wrong by factors
       of ~2 at high temperature.

    6. Boiling curve h_spray(T_surface) - piecewise-linear model
       Water spray heat transfer coefficient is NOT constant. It depends
       strongly on surface temperature due to boiling regime transitions:
         - T_surf > 400 C : film boiling (vapor blanket)     h ~  2,000
         - ~300 C         : Leidenfrost peak (max)           h ~ 40,000
         - 100-250 C      : nucleate boiling                 h ~ 30,000
         - T_surf < 100 C : single-phase liquid              h ~  3,000
       This is what causes the ACCELERATED COOLING once the surface drops
       below the Leidenfrost point - h jumps 20x, and cooling suddenly
       becomes very fast. Real quench curves have this cliff. Generic
       piecewise-linear values used here (not derived from a specific
       correlation or measured for a specific nozzle setup).

    7. Adaptive hysteretic spray control
       Spray turns on/off based on relative temperature of the spray zone
       vs. the plate average, not a fixed time schedule. Spray turns OFF
       when the sprayed region falls 100 K below plate average, back ON
       when it recovers to within 20 K of plate average. This produces
       natural pulsing that continues across the whole cool-down instead
       of dying out once the plate cools below a fixed threshold.

    8. Smooth valve dynamics (S-curve ramp + smoothstep h response)
       Real spray solenoid valves do not open/close instantly. Modeled here
       as two cascaded first-order lags (S-curve response, ~1s per stage),
       plus a smoothstep function (3s^2 - 2s^3) applied to how spray
       intensity maps to h. Added specifically to eliminate sharp,
       non-differentiable spikes at spray on/off transitions in the
       temperature profile - both for physical realism (real valves have
       inertia) and for future gradient-based path optimization where
       smooth derivatives matter.

NUMERICAL METHOD
    IMEX (implicit-explicit) backward Euler. Conduction and convection are
    solved implicitly each step via matrix-free conjugate gradient. Radiation
    (T^4) and all temperature-dependent coefficients are lagged: evaluated
    at T_old to keep the linear system linear. Unconditionally stable, so
    dt is chosen for accuracy (0.1s) rather than stability. Zero-flux
    (adiabatic) boundary conditions on all edges: represents a small patch
    of a larger plate.

LIMITATIONS
    - Lumped through-thickness (2D surface only, no volume mesh).
      The Biot number Bi = h_spray * L / k = 2000 * 0.01 / 45 = 0.44 exceeds
      the textbook lumped-capacitance criterion (Bi <= 0.1). Reported
      temperatures represent a DEPTH-AVERAGED value across the 10mm plate
      thickness, not the true surface temperature. Real surface during spray
      is colder than the depth-average; real core is hotter. Resolved by
      moving to a 3D volume mesh (next POC evolution).

    - Boiling curve h(T) is a generic piecewise-linear model chosen for
      physics-demo purposes, NOT derived from a validated correlation for
      a specific nozzle geometry, water mass flux, or surface condition.
      The Wendelstorf 2008 correlation would give more defensible values
      but requires committing to specific spray parameters. Also: because
      of the lumped limitation above, h(T) is evaluated at the depth-
      averaged T rather than the true surface T, so Leidenfrost onset
      appears later in the sim than it would in reality.

    - No latent heat of phase transformation. Real steel undergoes
      austenite -> martensite/bainite/pearlite transitions during quench,
      releasing ~80 kJ/kg of latent heat that creates a temperature plateau
      around 200-400 C. Deferred to post-symposium research work.

    - No spray hydrodynamics. Vapor blanket dynamics, water pooling,
      droplet impingement, and re-flashing are all lumped into the h(T)
      curve. Physically real but well beyond POC scope.

    - No back-face or edge convection. The 2D grid represents the top
      surface only; the plate is treated as adiabatic on all boundaries
      of the 2D domain. Meaningful only once a volume mesh distinguishes
      top, bottom, and sides.

    - Radiation view factor assumed = 1.0 (plate radiates to a large room
      at ambient). Correct for flat plate; will matter on curved geometry
      where surfaces radiate to themselves.

    - Constant plate density (rho = 7850 kg/m^3). Real thermal expansion
      is small over this range but nonzero; ignored here.
"""

import jax
from jax import config
config.update("jax_enable_x64", True)  # numerics matter here

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Material properties: AISI 1045 nominal constants
RHO       = 7850.0
SIGMA_SB  = 5.670374419e-8
T_TABLE_K = jnp.array([300.0, 400.0, 600.0, 800.0, 1000.0, 1200.0])
K_TABLE   = jnp.array([ 63.9,  58.7,  48.8,  39.2,   31.3,   28.0])
CP_TABLE  = jnp.array([434.0, 487.0, 559.0, 685.0, 1168.0,  600.0])
EPS_TABLE = jnp.array([ 0.70,  0.73,  0.78,  0.82,   0.85,   0.85])
def k_of_T(T):     return jnp.interp(T, T_TABLE_K, K_TABLE)
def cp_of_T(T):    return jnp.interp(T, T_TABLE_K, CP_TABLE)
def eps_of_T(T):   return jnp.interp(T, T_TABLE_K, EPS_TABLE)
def alpha_of_T(T): return k_of_T(T) / (RHO * cp_of_T(T))
ALPHA_REF = float(k_of_T(jnp.array(500.0)) / (RHO * cp_of_T(jnp.array(500.0))))

# Geometry
PLATE_SIZE = 0.20                 # m x m
PLATE_L    = 0.010                # m thickness (lumped through)
NX, NY     = 40, 40
DX         = PLATE_SIZE / NX      # 5 mm cell

# Heat transfer coefficients (PHYSICAL, W/(m^2 K))
H_AMBIENT  = 15.0
H_BOIL_T_C = jnp.array([  25.0,  100.0,  200.0,  300.0,  400.0,  500.0,  900.0])
H_BOIL_VAL = jnp.array([3000.0, 3000.0,30000.0,40000.0,15000.0, 3000.0, 2000.0])
def h_spray_of_T(T_K):
    return jnp.interp(T_K - 273.15, H_BOIL_T_C, H_BOIL_VAL)

# Temperatures (Kelvin internally)
T_INIT_C     = 900.0
T_AMBIENT_C  = 25.0
T_INIT_K     = T_INIT_C    + 273.15
T_AMBIENT_K  = T_AMBIENT_C + 273.15

# Event-driven pulsed control
T_COOL_ENOUGH_C   = 300.0     # global stop criterion (avg plate temp)

# Adaptive hysteresis (relative to plate average — pulses at every temperature)
DELTA_OFF_K       = 100.0     # spray OFF when spray zone is this much colder than plate avg
DELTA_ON_K        =  20.0     # spray ON  when spray zone recovers to within this of plate avg

# Smooth on/off ramp (spray valve dynamics, for differentiability)
# S-curve = two cascaded 1st-order lags. Total 10-90% rise time ~= 3.36 * tau.
SPRAY_RAMP_TAU_S  = 1.0       # per-stage lag; two stages give S-curve

DT        = 0.1
T_MAX     = 900.0             # simulate up to 15 minutes if needed
MAX_STEPS = int(T_MAX / DT)


# Spray footprint (Gaussian, 4 cm radius)
SPRAY_RADIUS_M     = 0.04
SPRAY_RADIUS_NODES = SPRAY_RADIUS_M / DX
CX, CY = NX // 2, NY // 2

# Physics: pure functions (lift to shared module for POC 2/3 later)

def laplacian_2d_zero_flux(T, dx):
    """5-point discrete Laplacian, zero-flux (Neumann) BCs via edge padding."""
    Tp = jnp.pad(T, 1, mode='edge')
    return (Tp[2:, 1:-1] + Tp[:-2, 1:-1]
          + Tp[1:-1, 2:] + Tp[1:-1, :-2]
          - 4.0 * Tp[1:-1, 1:-1]) / (dx * dx)

def linear_operator(T, dt, alpha_field, dx, beta_field):
    """(I - dt*alpha*Lap + dt*diag(beta_field)) applied to T. alpha_field per-node ok."""
    return T - dt * alpha_field * laplacian_2d_zero_flux(T, dx) + dt * beta_field * T

def radiation_flux(T, gamma_field, T_amb_rad):
    """Radiation sink [K/s], gamma_field may be per-node (T-dependent emissivity)."""
    return gamma_field * (T**4 - T_amb_rad**4)

def imex_backward_euler_step(T_old, dt, alpha_field, dx,
                             beta_field, T_amb_conv,
                             gamma_field, T_amb_rad):
    """IMEX BE step. All coefficient fields evaluated at T_old (lagged) upstream."""
    rad = radiation_flux(T_old, gamma_field, T_amb_rad)
    rhs = T_old + dt * beta_field * T_amb_conv - dt * rad
    A   = lambda T: linear_operator(T, dt, alpha_field, dx, beta_field)
    T_new, _ = jax.scipy.sparse.linalg.cg(A, rhs, x0=T_old, tol=1e-8, maxiter=300)
    return T_new

# Precompute weights for measuring "spray zone" average temperature.
# Weight by the spray mask itself — this is the region we're controlling on.
# Precompute spray mask (Gaussian footprint centered at CX, CY)
yy_np, xx_np = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij')
dist_nodes   = np.sqrt((xx_np - CX)**2 + (yy_np - CY)**2)
sigma_spray  = SPRAY_RADIUS_NODES / 2.0
spray_mask   = jnp.array(np.exp(-dist_nodes**2 / (2 * sigma_spray**2)))
spray_zone_weights = spray_mask / jnp.sum(spray_mask)

# Initial temperature field
T_init = jnp.full((NY, NX), T_INIT_K)

# Diagnostic prints
print("Running event-driven IMEX backward-Euler simulation...")
print(f"  alpha (ref, T=500K)  = {ALPHA_REF:.3e} m^2/s")
print(f"  k(300K)={float(k_of_T(jnp.array(300.0))):.1f}, k(900C)={float(k_of_T(T_INIT_K)):.1f} W/(m K)")
print(f"  cp(300K)={float(cp_of_T(jnp.array(300.0))):.0f}, cp(900C)={float(cp_of_T(T_INIT_K)):.0f} J/(kg K)")
print(f"  h_spray(900C)={float(h_spray_of_T(T_INIT_K)):.0f} W/(m^2 K)")
print(f"  h_spray(300C)={float(h_spray_of_T(jnp.array(573.15))):.0f} W/(m^2 K)  [Leidenfrost peak]")
print(f"  h_spray(150C)={float(h_spray_of_T(jnp.array(423.15))):.0f} W/(m^2 K)  [nucleate boiling]")
print(f"  eps(900C)={float(eps_of_T(T_INIT_K)):.2f}")


@jax.jit
def step(carry, _):
    # carry = (T_old, valve_target, spray_stage1, spray_actual)
    #   valve_target : discrete 0/1 control command from hysteresis
    #   spray_stage1 : intermediate ramp variable (1st lag output)
    #   spray_actual : final smoothed spray intensity in [0, 1] (2nd lag output)
    #                  S-curve response: zero slope at both endpoints.
    T_old, valve_target, spray_stage1, spray_actual = carry

    # Measure current spray-zone average and plate average (Kelvin)
    T_zone      = jnp.sum(T_old * spray_zone_weights)
    T_plate_avg = jnp.mean(T_old)

    # Adaptive hysteresis relative to plate average (pulses at all temps)
    turn_off = valve_target         * (T_zone < T_plate_avg - DELTA_OFF_K).astype(jnp.float64)
    turn_on  = (1.0 - valve_target) * (T_zone > T_plate_avg - DELTA_ON_K ).astype(jnp.float64)
    valve_target_new = valve_target - turn_off + turn_on

    # Two-stage cascaded lag => S-curve ramp (zero slope at start & end of transition)
    alpha_ramp = DT / (SPRAY_RAMP_TAU_S + DT)
    spray_stage1_new = spray_stage1 + alpha_ramp * (valve_target_new - spray_stage1)
    spray_actual_new = spray_actual + alpha_ramp * (spray_stage1_new - spray_actual)

    # T-dependent material properties (evaluated at T_old — lagged, keeps solve linear)
    k_field   = k_of_T(T_old)
    cp_field  = cp_of_T(T_old)
    eps_field = eps_of_T(T_old)
    rhocp     = RHO * cp_field
    alpha_fld = k_field / rhocp

    # Boiling-curve h evaluated at T_old (surface T proxy = lumped T for POC)
    h_spray_field = h_spray_of_T(T_old)

    # Smooth spray intensity, then apply to boiling-curve h
    s = jnp.clip(spray_actual_new, 0.0, 1.0)
    spray_smooth = s * s * (3.0 - 2.0 * s)   # C1 smoothstep
    h_field    = H_AMBIENT + spray_mask * spray_smooth * (h_spray_field - H_AMBIENT)
    beta_field = h_field / (rhocp * PLATE_L)
    gamma_fld  = eps_field * SIGMA_SB / (rhocp * PLATE_L)

    T_new = imex_backward_euler_step(
        T_old, DT, alpha_fld, DX,
        beta_field, T_AMBIENT_K,
        gamma_fld, T_AMBIENT_K,
    )
    return (T_new, valve_target_new, spray_stage1_new, spray_actual_new), (T_new, spray_actual_new)

print("Running event-driven IMEX backward-Euler simulation...")
init_carry = (T_init, jnp.float64(1.0), jnp.float64(1.0), jnp.float64(1.0))   # start with spray ON
_, (T_history_K, spray_history) = jax.lax.scan(step, init_carry, None, length=MAX_STEPS)
T_history_K = np.array(T_history_K)
T_history_C = T_history_K - 273.15
spray_history = np.array(spray_history)

# Truncate to when plate reached "cool enough"
plate_avg_C = T_history_C.mean(axis=(1, 2))
cool_idx = np.argmax(plate_avg_C < T_COOL_ENOUGH_C)
if cool_idx == 0 and plate_avg_C[-1] >= T_COOL_ENOUGH_C:
    print(f"WARNING: plate did not reach {T_COOL_ENOUGH_C} C within {T_MAX}s")
    print(f"  Final avg temp: {plate_avg_C[-1]:.1f} C")
    end_idx = MAX_STEPS
else:
    end_idx = min(cool_idx + int(5.0/DT), MAX_STEPS)  # 5s of margin
    print(f"Plate reached {T_COOL_ENOUGH_C} C avg at t = {cool_idx * DT:.1f} s")

T_history_C   = T_history_C[:end_idx]
spray_history = spray_history[:end_idx]
STEPS = end_idx
T_END = STEPS * DT

# Visualization (unchanged interaction: heatmap + click-to-plot history)
t_axis = np.linspace(0, T_END, STEPS)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax, ax_history = axes

im = ax.imshow(T_history_C[0], cmap='inferno', origin='lower',
               vmin=T_AMBIENT_C, vmax=T_INIT_C)
fig.colorbar(im, ax=ax, label='Temperature (°C)')
ax.set_title('Pulsed spray cooling (SI, IMEX BE)')
ax.set_xlabel('x (nodes)')
ax.set_ylabel('y (nodes)')

time_text  = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=10)
spray_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, color='cyan',  fontsize=10)

circle = plt.Circle((CX, CY), SPRAY_RADIUS_NODES, color='cyan',
                    fill=False, linewidth=2, label='Spray zone')
ax.add_patch(circle)
ax.legend(loc='upper right', fontsize=9)

ax_history.set_xlabel('Time (s)')
ax_history.set_ylabel('Temperature (°C)')
ax_history.set_title('Thermal history (click nodes on heatmap)')
ax_history.set_xlim(0, T_END)
ax_history.set_ylim(T_AMBIENT_C - 10, T_INIT_C + 10)
ax_history.grid(True, alpha=0.3)

# Shade actual spray-ON periods from history
in_on = False
on_start = 0
labeled = False
for i, s in enumerate(spray_history):
    if s > 0.5 and not in_on:
        on_start = i * DT
        in_on = True
    elif s <= 0.5 and in_on:
        ax_history.axvspan(on_start, i * DT, alpha=0.1, color='cyan',
                           label='Spray ON' if not labeled else '')
        labeled = True
        in_on = False
if in_on:
    ax_history.axvspan(on_start, STEPS * DT, alpha=0.1, color='cyan',
                       label='Spray ON' if not labeled else '')

skip = max(1, STEPS // 300)

def animate(frame):
    actual = frame * skip
    im.set_data(T_history_C[actual])
    t = actual * DT
    on = spray_history[actual] > 0.5
    time_text.set_text(f't = {t:.1f}s')
    spray_text.set_text('SPRAY ON' if on else 'SPRAY OFF')
    return [im, time_text, spray_text]

def on_click(event):
    if event.inaxes != ax:
        return
    col = int(np.clip(round(event.xdata), 0, NX - 1))
    row = int(np.clip(round(event.ydata), 0, NY - 1))
    ax_history.plot(t_axis, T_history_C[:, row, col],
                    linewidth=1.5, label=f'({col},{row})')
    ax_history.legend(fontsize=8, loc='upper right')
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', on_click)

ani = animation.FuncAnimation(fig, animate, frames=range(0, STEPS // skip),interval=60, blit=False)
plt.tight_layout()
plt.show()

# Summary
print(f"\nFinal results (t = {T_END:.0f}s):")
print(f"  Peak temp:    {T_history_C[-1].max():.1f} C")
print(f"  Min temp:     {T_history_C[-1].min():.1f} C")
print(f"  Spread:       {T_history_C[-1].max() - T_history_C[-1].min():.1f} C")
print(f"  Avg temp:     {T_history_C[-1].mean():.1f} C")