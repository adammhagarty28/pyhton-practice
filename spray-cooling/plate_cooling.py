"""
plate_cooling_2d.py
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

    # --- Uniform spray cooling with Leidenfrost ---
    T_avg = T.mean()
    if T_avg > 200.0:
        Q_uniform = 500.0 * (T - T_water) * dt / 10000.0
        regime_label = "film boiling"
    elif T_avg > 100.0:
        Q_uniform = np.full((n, n), 5.0 * dt)
        regime_label = "nucleate boiling"
    else:
        Q_uniform = 0.5 * (T - T_water) * dt / 1000.0
        regime_label = "convection"

    T = T - Q_uniform
    T = np.clip(T, T_water, None)

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
for idx, ax in enumerate(axes):
    if idx < len(T_snapshots):
        im = ax.imshow(T_snapshots[idx], cmap="hot", vmin=T_water, vmax=900)
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

"""
To summarize: a 10x10 grid of points representing a steel plate that starts hot in the center (900 degrees celsius) and cooler
at the edges (400 degrees celsius). Every time step, two tings happen simultaneously:heat diffuses between neighboring points
according to the 2D heat equaiton (hot center spreading outward), and uniform spray cooling removes heat from every point on 
the plate at a rate determined by the Leidenfrost condition (film boiling while the plate is above 200 degrees celsius average).
The while loop keeps running until the hottest point drops below 300 degrees celsius, at whcih the plate is considered safely cooled
"""