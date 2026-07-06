"""
POC: Snake path cooling trail on uniform hot plate.
Simple, visual. Shows nozzle traversing zigzag path
with fading trail and cooling left behind.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- parameters ---
nx, ny       = 40, 40
T_initial    = 900.0
T_ambient    = 25.0
h_spray      = 0.15
h_ambient    = 0.005
spray_radius = 3
dt           = 0.5
row_step     = 2

# --- initial temperature ---
T = np.full((ny, nx), T_initial)

# --- zigzag path ---
path = []
for row in range(0, ny, row_step):
    cols = range(nx) if (row // row_step) % 2 == 0 else range(nx-1, -1, -1)
    for col in cols:
        path.append((col, row))

# --- precompute spray masks ---
yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
spray_masks = []
for (px, py) in path:
    dist = np.sqrt((xx - px)**2 + (yy - py)**2)
    spray_masks.append(dist <= spray_radius)
spray_masks = np.array(spray_masks)

# --- simulate: one frame = nozzle moves one step ---
T_history  = []
path_history = []
T_current = T.copy()

for pidx, (px, py) in enumerate(path):
    spray_mask = spray_masks[pidx]
    h_field    = np.where(spray_mask, h_spray, h_ambient)
    dTdt       = -h_field * (T_current - T_ambient)
    T_current  = T_current + dTdt * dt
    T_history.append(T_current.copy())
    path_history.append(pidx)

T_history = np.array(T_history)

# --- animation + interactive clicking ---
trail_length = 20
clicked_points = []
t_axis = np.arange(len(path))

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax = axes[0]
ax_history = axes[1]

im = ax.imshow(T_history[0], cmap='inferno', origin='lower',
               vmin=T_ambient, vmax=T_initial)
fig.colorbar(im, ax=ax, label='Temperature (°C)')
ax.set_title('Snake path spray cooling trail')
ax.set_xlabel('x (nodes)')
ax.set_ylabel('y (nodes)')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                    color='white', fontsize=10)

# nozzle
nozzle_dot, = ax.plot([], [], 'c^', markersize=12, zorder=6, label='Nozzle')

# fading trail
trail_dots = [
    ax.plot([], [], 'o', color='cyan',
            alpha=(i+1)/trail_length * 0.8,
            markersize=5, zorder=5)[0]
    for i in range(trail_length)
]
ax.legend(loc='upper right', fontsize=9)

# thermal history subplot
ax_history.set_xlabel('Step')
ax_history.set_ylabel('Temperature (°C)')
ax_history.set_title('Thermal history (click nodes on heatmap)')
ax_history.set_xlim(0, len(path))
ax_history.set_ylim(T_ambient - 10, T_initial + 10)
ax_history.grid(True, alpha=0.3)

def animate(frame):
    im.set_data(T_history[frame])
    time_text.set_text(f'step {frame}/{len(path)}')

    nozzle_dot.set_data([path[frame][0]], [path[frame][1]])

    for i, dot in enumerate(trail_dots):
        trail_frame = frame - (trail_length - i)
        if trail_frame >= 0:
            dot.set_data([path[trail_frame][0]], [path[trail_frame][1]])
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
    label = f'({col},{row})'
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