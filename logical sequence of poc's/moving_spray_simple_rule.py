import numpy as np
import matplotlib.pyplot as plt

#grid setup
nx, ny = 40, 40
T_initial = 900.0
T_ambient = 25.0

h_spray = 0.15        # cooling coefficient under the spray
h_ambient = 0.01      # cooling coefficient everywhere else
spray_radius = 5      # radius of spray footprint

dt = 0.5
t_end = 120.0
steps = int(t_end / dt)

#temperature field
T = np.full((ny, nx), T_initial)

#build zigzag path: nozzle visits a sequence of (x, y) positions
# sweep left-to-right on one row, then right-to-left on the next, moving down
row_step = 4   # how many grid rows to skip per pass (controls scan density)
path = []
for row in range(0, ny, row_step):
    cols = range(nx) if (row // row_step) % 2 == 0 else range(nx - 1, -1, -1)
    for col in cols:
        path.append((col, row))

#map simulation steps to path positions
#nozzle advances one path point every few timesteps
steps_per_move = max(1, steps // len(path))

yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

#simulate
path_idx = 0
for step in range(steps):
    if step % steps_per_move == 0 and path_idx < len(path) - 1:
        path_idx += 1

    nozzle_x, nozzle_y = path[path_idx]
    dist = np.sqrt((xx - nozzle_x)**2 + (yy - nozzle_y)**2)
    spray_mask = dist <= spray_radius

    h_field = np.where(spray_mask, h_spray, h_ambient)
    dTdt = -h_field * (T - T_ambient)
    T = T + dTdt * dt

#plot
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(T, cmap='inferno', origin='lower')
ax.set_title(f'2D part temperature after {t_end:.0f}s (zigzag scan spray)')
ax.set_xlabel('x')
ax.set_ylabel('y')

#overlay the path the nozzle took
path_x = [p[0] for p in path]
path_y = [p[1] for p in path]
ax.plot(path_x, path_y, color='cyan', linewidth=1, alpha=0.6, label='Nozzle path')
ax.legend(loc='upper right')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Temperature (°C)')

plt.tight_layout()
plt.show()

print(f"Hottest point: {T.max():.1f} C")
print(f"Coolest point: {T.min():.1f} C")
print(f"Temperature spread: {T.max() - T.min():.1f} C")
print(f"Average temperature: {T.mean():.1f} C")