"""
POC: Snake path spray cooling with conduction added in JAX

Uniform hot plate cooled by zigzag spray nozzle. Adds conduction/diffusion
between nodes on top of the original spray cooling. Lumped h coefficients
kept for visible results. JAX lax.scan replaces Python for loop.

Physics:
    dT/dt = alpha * laplacian(T) - h_field * (T - T_ambient)
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#parameters
nx, ny       = 40, 40
T_initial    = 900.0
T_ambient    = 25.0
h_spray      = 0.15
h_ambient    = 0.005
alpha        = 0.01
spray_radius = 3
dt           = 0.5
row_step     = 2

#initial temperature
T_init = jnp.full((ny, nx), T_initial)

#zigzag path
path = []
for row in range(0, ny, row_step):
    cols = range(nx) if (row // row_step) % 2 == 0 else range(nx-1, -1, -1)
    for col in cols:
        path.append((col, row))

steps = len(path)
steps_per_move = 1

#precompute spray masks
yy_np, xx_np = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
spray_masks = np.zeros((len(path), ny, nx), dtype=np.float32)
for i, (px, py) in enumerate(path):
    dist = np.sqrt((xx_np - px)**2 + (yy_np - py)**2)
    spray_masks[i] = (dist <= spray_radius).astype(np.float32)
spray_masks_jax = jnp.array(spray_masks)

#JAX step function
@jax.jit
def step(carry, _):
    T, step_idx = carry

    path_idx   = jnp.minimum(step_idx, len(path) - 1)
    spray_mask = spray_masks_jax[path_idx]
    h_field    = jnp.where(spray_mask, h_spray, h_ambient)

    # laplacian in x and y
    d2Tdx2 = jnp.zeros_like(T)
    d2Tdy2 = jnp.zeros_like(T)
    d2Tdx2 = d2Tdx2.at[:, 1:-1].set(
        (T[:, 2:] - 2*T[:, 1:-1] + T[:, :-2])
    )
    d2Tdy2 = d2Tdy2.at[1:-1, :].set(
        (T[2:, :] - 2*T[1:-1, :] + T[:-2, :])
    )
    laplacian = d2Tdx2 + d2Tdy2

    dTdt  = alpha * laplacian - h_field * (T - T_ambient)
    T_new = T + dTdt * dt

    return (T_new, step_idx + 1), (T_new, path_idx)

#run simulation
print("Running simulation...")
_, (T_history, path_idx_history) = jax.lax.scan(
    step, (T_init, jnp.int32(0)), None, length=steps
)
T_history        = np.array(T_history)
path_idx_history = np.array(path_idx_history)
print("Done.")

#animation + interactive clicking
trail_length   = 20
clicked_points = []
t_axis         = np.arange(len(path))

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax         = axes[0]
ax_history = axes[1]

im = ax.imshow(T_history[0], cmap='inferno', origin='lower',
               vmin=T_ambient, vmax=T_initial)
fig.colorbar(im, ax=ax, label='Temperature (°C)')
ax.set_title('Snake path spray cooling trail')
ax.set_xlabel('x (nodes)')
ax.set_ylabel('y (nodes)')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                    color='white', fontsize=10)

nozzle_dot, = ax.plot([], [], 'c^', markersize=12, zorder=6, label='Nozzle')

trail_dots = [
    ax.plot([], [], 'o', color='cyan',
            alpha=(i+1)/trail_length * 0.8,
            markersize=5, zorder=5)[0]
    for i in range(trail_length)
]
ax.legend(loc='upper right', fontsize=9)

ax_history.set_xlabel('Step')
ax_history.set_ylabel('Temperature (°C)')
ax_history.set_title('Thermal history (click nodes on heatmap)')
ax_history.set_xlim(0, len(path))
ax_history.set_ylim(T_ambient - 10, T_initial + 10)
ax_history.grid(True, alpha=0.3)

def animate(frame):
    im.set_data(T_history[frame])
    time_text.set_text(f'step {frame}/{len(path)}')

    pidx = path_idx_history[frame]
    nozzle_dot.set_data([path[pidx][0]], [path[pidx][1]])

    for i, dot in enumerate(trail_dots):
        trail_frame = frame - (trail_length - i)
        if trail_frame >= 0:
            tidx = path_idx_history[trail_frame]
            dot.set_data([path[tidx][0]], [path[tidx][1]])
        else:
            dot.set_data([], [])

    return [im, time_text, nozzle_dot] + trail_dots

def on_click(event):
    if event.inaxes != ax:
        return
    col = int(round(event.xdata))
    row = int(round(event.ydata))
    col = np.clip(col, 0, nx - 1)
    row = np.clip(row, 0, ny - 1)

    node_history = T_history[:, row, col]
    label        = f'({col},{row})'
    ax_history.plot(t_axis, node_history, linewidth=1.5, label=label)
    ax_history.legend(fontsize=8, loc='upper right')
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', on_click)

ani = animation.FuncAnimation(
    fig, animate,
    frames=len(path),
    interval=40,
    blit=False
)

plt.tight_layout()
plt.show()

print(f"\nFinal results:")
print(f"  Peak temp:    {T_history[-1].max():.1f} C")
print(f"  Min temp:     {T_history[-1].min():.1f} C")
print(f"  Temp spread:  {T_history[-1].max() - T_history[-1].min():.1f} C")
print(f"  Avg temp:     {T_history[-1].mean():.1f} C")