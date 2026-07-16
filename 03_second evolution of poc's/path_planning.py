"""
POC: Path Planning - Greedy Hottest-First Nozzle Policy in JAX

Instead of a fixed zigzag pattern, the nozzle adaptively targets the
hottest region of the part at each move decision. Full heat equation
with conduction/diffusion + convective spray cooling. Animated heatmap
with nozzle trail and interactive clicking for thermal history.

Physics:
    dT/dt = alpha * laplacian(T) - (h_field / rho_c) * (T - T_ambient)

Material: generic carbon steel
    k    = 50   W/mK
    rho  = 7800 kg/m3
    c    = 500  J/kgK
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#parameters
nx, ny        = 40, 40
Lx, Ly        = 0.04, 0.04
dx            = Lx / (nx - 1)

k             = 50.0
rho           = 7800.0
c             = 500.0
rho_c         = rho * c
alpha         = k / rho_c

h_spray       = 5000.0
h_ambient     = 10.0
T_ambient     = 25.0
T_initial     = 900.0

spray_radius  = 4

dt            = 0.001
t_end         = 30.0
steps         = int(t_end / dt)
steps_per_move = 50

stability = alpha * dt / dx**2
print(f"Stability number: {stability:.4f}  (must be < 0.25)")
if stability >= 0.25:
    raise ValueError("Unstable! Reduce dt or increase dx.")

# initial condition: Gaussian hot spot centered on grid
x_vals = jnp.linspace(0, Lx, nx)
y_vals = jnp.linspace(0, Ly, ny)
XX, YY = jnp.meshgrid(x_vals, y_vals)
cx, cy  = Lx / 2, Ly / 2
sigma_g = 0.008
T_init  = T_ambient + (T_initial - T_ambient) * jnp.exp(
    -((XX - cx)**2 + (YY - cy)**2) / (2 * sigma_g**2)
)

#precompute spray masks for every possible nozzle position
yy_np, xx_np = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
spray_masks = np.zeros((ny * nx, ny, nx), dtype=np.float32)
for row in range(ny):
    for col in range(nx):
        dist = np.sqrt((xx_np - col)**2 + (yy_np - row)**2)
        spray_masks[row * nx + col] = (dist <= spray_radius).astype(np.float32)
spray_masks_jax = jnp.array(spray_masks)

xx_np_jax = jnp.array(xx_np)
yy_np_jax = jnp.array(yy_np)

#JAX step function: greedy hottest-first policy


@jax.jit
def step(carry, _):
    T, step_idx, nozzle_row, nozzle_col = carry

    dist_from_nozzle = jnp.sqrt(
        (xx_np_jax - nozzle_col)**2 + (yy_np_jax - nozzle_row)**2
    )
    score    = T + 200.0 * dist_from_nozzle / (nx + ny)
    flat_idx = jnp.argmax(score)
    hot_row  = flat_idx // nx
    hot_col  = flat_idx  % nx

    should_move = (step_idx % steps_per_move) == 0
    nozzle_row  = jnp.where(should_move, hot_row, nozzle_row)
    nozzle_col  = jnp.where(should_move, hot_col, nozzle_col)

    mask_idx   = nozzle_row * nx + nozzle_col
    spray_mask = spray_masks_jax[mask_idx]
    h_field    = jnp.where(spray_mask, h_spray, h_ambient)

    d2Tdx2 = jnp.zeros_like(T)
    d2Tdy2 = jnp.zeros_like(T)
    d2Tdx2 = d2Tdx2.at[1:-1, :].set(
        (T[2:, :] - 2*T[1:-1, :] + T[:-2, :]) / dx**2
    )
    d2Tdy2 = d2Tdy2.at[:, 1:-1].set(
        (T[:, 2:] - 2*T[:, 1:-1] + T[:, :-2]) / dx**2
    )
    laplacian = d2Tdx2 + d2Tdy2

    dTdt  = alpha * laplacian - (h_field / rho_c) * (T - T_ambient)
    T_new = T + dTdt * dt

    carry_new = (T_new, step_idx + 1, nozzle_row, nozzle_col)
    output    = (T_new, nozzle_row, nozzle_col)
    return carry_new, output

#run simulation
print("Running simulation...")
init_carry = (T_init, jnp.int32(0), jnp.int32(ny // 2), jnp.int32(nx // 2))
_, (T_history, nozzle_row_hist, nozzle_col_hist) = jax.lax.scan(
    step, init_carry, None, length=steps
)
T_history       = np.array(T_history)
nozzle_row_hist = np.array(nozzle_row_hist)
nozzle_col_hist = np.array(nozzle_col_hist)
print("Done.")

#animation + interactive clicking
trail_length   = 40
clicked_points = []

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax_heat    = axes[0]
ax_history = axes[1]

im = ax_heat.imshow(
    T_history[0], cmap='inferno', origin='lower',
    vmin=T_ambient, vmax=T_initial,
    extent=[0, nx, 0, ny]
)
fig.colorbar(im, ax=ax_heat, label='Temperature (°C)')
ax_heat.set_title('Path planning: hottest-first nozzle policy')
ax_heat.set_xlabel('x (nodes)')
ax_heat.set_ylabel('y (nodes)')
time_text = ax_heat.text(0.02, 0.95, '', transform=ax_heat.transAxes,
                          color='white', fontsize=10)

nozzle_dot, = ax_heat.plot([], [], 'c^', markersize=10, zorder=6, label='Nozzle')

trail_dots = [
    ax_heat.plot([], [], 'o', color='cyan',
                 alpha=(i + 1) / trail_length * 0.7,
                 markersize=4, zorder=5)[0]
    for i in range(trail_length)
]
ax_heat.legend(loc='upper right', fontsize=8)
scatter = ax_heat.scatter([], [], c='yellow', s=40, zorder=7)

ax_history.set_xlabel('Time (s)')
ax_history.set_ylabel('Temperature (°C)')
ax_history.set_title('Thermal history (click nodes on heatmap)')
ax_history.set_xlim(0, t_end)
ax_history.set_ylim(T_ambient - 10, T_initial + 10)
ax_history.grid(True, alpha=0.3)
t_axis = np.linspace(0, t_end, steps)

skip = max(1, steps // 150)

def animate(frame):
    actual = frame * skip
    im.set_data(T_history[actual])
    time_text.set_text(f't = {actual * dt:.2f}s')
    nozzle_dot.set_data([nozzle_col_hist[actual]], [nozzle_row_hist[actual]])
    for i, dot in enumerate(trail_dots):
        trail_frame = actual - (trail_length - i) * skip
        if trail_frame >= 0:
            dot.set_data([nozzle_col_hist[trail_frame]], [nozzle_row_hist[trail_frame]])
        else:
            dot.set_data([], [])
    return [im, time_text, nozzle_dot] + trail_dots

def on_click(event):
    if event.inaxes != ax_heat:
        return
    col = int(round(event.xdata))
    row = int(round(event.ydata))
    col = np.clip(col, 0, nx - 1)
    row = np.clip(row, 0, ny - 1)
    node_history = T_history[:, row, col]
    label = f'({col},{row})'
    ax_history.plot(t_axis, node_history, linewidth=1.5, label=label)
    ax_history.legend(fontsize=8, loc='upper right')
    clicked_points.append((col, row))
    xs = [p[0] for p in clicked_points]
    ys = [p[1] for p in clicked_points]
    scatter.set_offsets(np.c_[xs, ys])
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', on_click)

ani = animation.FuncAnimation(
    fig, animate,
    frames=range(0, steps // skip),
    interval=10,
    blit=False
)

plt.tight_layout()
plt.show()

print(f"\nFinal results:")
print(f"  Peak temp:    {T_history[-1].max():.1f} C")
print(f"  Min temp:     {T_history[-1].min():.1f} C")
print(f"  Temp spread:  {T_history[-1].max() - T_history[-1].min():.1f} C")
print(f"  Avg temp:     {T_history[-1].mean():.1f} C")