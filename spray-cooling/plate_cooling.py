"""
plate_cooling.py
2D steel plate spray cooling simulation.
Physics: heat diffusion + Leidenfrost effect + uniform spray cooling
Units: temperature in C, position in m, time in s
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Plate setup ---
n = 10
L = 0.1
positions = np.linspace(0, L, n)
dx = positions[1] - positions[0]

# --- Real water properties at ~100C (Bergman Appendix A) ---
mu_l = 2.82e-4        # dynamic viscosity of liquid water (N·s/m²)
h_fg = 2257000.0      # latent heat of vaporization (J/kg)
rho_l = 957.9         # liquid water density (kg/m³)
rho_v = 0.596         # vapor density (kg/m³)
sigma = 0.0589        # surface tension (N/m)
Cp_l = 4217.0         # specific heat of liquid water (J/kg·K)
Pr_l = 1.76           # Prandtl number of liquid water
C_sf = 0.013          # surface fluid constant water/steel (Bergman Table 10.1)
n_exp = 1.0           # exponent for water (Bergman Table 10.1)
g = 9.81              # gravity (m/s²)
T_sat = 100.0         # saturation temperature at 1 atm (°C)
h_film = 300.0        # film boiling coefficient W/m²·K
h_conv = 1000.0       # forced convection coefficient W/m²·K
A_point = (L/n)**2    # area per grid point (m²)

# --- Steel properties ---
rho_solid = 7800.0    # density of steel (kg/m³)
cp_solid = 500.0      # specific heat of steel (J/kg·K)
V_point = A_point * (L/10)  # volume per grid point (m³)

# --- Material properties (steel) ---
alpha = 1.17e-5
dt = 0.1
safe_temp = 300.0

# --- Non-uniform initial temperature: hot center, cooler edges ---
cx, cy = L/2, L/2
T = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist = np.sqrt((positions[i]-cx)**2 + (positions[j]-cy)**2)
        T[i, j] = 900.0 - 5000.0 * dist
T = np.clip(T, 400.0, 900.0)

# --- Physical constants ---
T_water = 20.0

# --- Storage ---
time_history = []
mean_history = []
max_history = []
min_history = []
T_snapshots = []
snapshot_times = []

# --- Simulation ---
t = 0.0
step = 0
T_snapshots.append(T.copy())
snapshot_times.append(t)

print(f"{'Time(s)':<10} {'Max(C)':<10} {'Mean(C)':<10} {'Min(C)':<10} {'Regime':<20}")
print("-" * 60)

while np.max(T) > safe_temp and step < 10000:

    # --- 2D heat diffusion ---
    d2Tdx2 = np.zeros((n, n))
    d2Tdy2 = np.zeros((n, n))
    d2Tdx2[1:-1, :] = (T[2:, :] - 2*T[1:-1, :] + T[:-2, :]) / dx**2
    d2Tdy2[:, 1:-1] = (T[:, 2:] - 2*T[:, 1:-1] + T[:, :-2]) / dx**2
    laplacian = d2Tdx2 + d2Tdy2
    T = T + alpha * dt * laplacian

  # --- Real spray cooling: local regime per point (Bergman Ch.10) ---
    delta_Te = np.maximum(T - T_sat, 0.0)
    q_total = np.zeros((n, n))

    film_mask = T > 200.0
    nucleate_mask = (T > T_sat) & (T <= 200.0)
    conv_mask = T <= T_sat

    # Film boiling: Newton's law with film boiling h
    q_total[film_mask] = h_film * (T[film_mask] - T_sat)

    # Nucleate boiling: Rohsenow correlation (Bergman Eq. 10.5)
    term1 = mu_l * h_fg
    term2 = (g * (rho_l - rho_v) / sigma) ** 0.5
    term3 = (Cp_l * delta_Te[nucleate_mask] / (C_sf * h_fg * Pr_l**n_exp)) ** 3
    q_total[nucleate_mask] = term1 * term2 * term3

    # Single phase convection
    q_total[conv_mask] = h_conv * (T[conv_mask] - T_water)

    # Convert heat flux to temperature drop
    Q_removed = q_total * A_point * dt
    dT_cooling = Q_removed / (rho_solid * cp_solid * V_point)
    T = T - dT_cooling
    T = np.clip(T, T_water, None)

    # Regime label for terminal output (dominant regime)
    if np.any(film_mask):
        regime_label = "film boiling"
    elif np.any(nucleate_mask):
        regime_label = "nucleate boiling"
    else:
        regime_label = "convection"

    # --- Store history ---
    time_history.append(t)
    mean_history.append(T.mean())
    max_history.append(T.max())
    min_history.append(T.min())

    # --- Print every 50 steps ---
    if step % 50 == 0:
        print(f"{t:<10.1f} {T.max():<10.1f} {T.mean():<10.1f} {T.min():<10.1f} {regime_label:<20}")

    # --- Snapshots ---
    if step == 50 or step == 200:
        T_snapshots.append(T.copy())
        snapshot_times.append(t)

    t += dt
    step += 1

T_snapshots.append(T.copy())
snapshot_times.append(t)

print(f"\nSimulation complete: {step} steps, {t:.1f} seconds")
print(f"Final max temp: {T.max():.1f} C | Final mean temp: {T.mean():.1f} C")

# --- Plot 1: heatmap subplots ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles = [f"t = {snapshot_times[i]:.1f}s" for i in range(len(T_snapshots))]
vmin_val=T_snapshots[0].min()
vmax_val=T_snapshots[0].max()
for idx, ax in enumerate(axes):
    if idx < len(T_snapshots):
        im=ax.imshow(T_snapshots[idx], cmap="hot", vmin=vmin_val,vmax=vmax_val)
        ax.set_title(titles[idx])
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        plt.colorbar(im, ax=ax, label="Temp (C)")
plt.tight_layout()

# --- Plot 2: temperature history ---
plt.figure()
plt.plot(time_history, max_history, color="red", label="Max Temp")
plt.plot(time_history, mean_history, color="orange", label="Mean Temp")
plt.plot(time_history, min_history, color="blue", label="Min Temp")
plt.axhline(y=safe_temp, color="green", linestyle="--", label="Safe Threshold")
plt.title("Plate Temperature Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (C)")
plt.legend()
plt.grid(True)

plt.show()


"""To summarize: a 10x10 grid of points representing a steel plate that starts hot in the center (900 degrees celsius) and cooler
at the edges (400 degrees celsius). Every time step, heat diffuses through the steeel plate and spray cooling removes heat
from each grid point. The cooling regime is now determined locally at each grid point: film boiling betwen 100 and 200 degrees celsius,
and cnvetion below 100 degrees celsius. The code converts local heat flux energy removed, then into a temperature drop using the
steel density, heat capacity, and grid-cell volume. """