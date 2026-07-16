import numpy as np
import matplotlib.pyplot as plt


# --- grid setup ---
nx, ny = 40, 40
T_initial = 900.0
T_ambient = 25.0
 
h_spray = 0.15
h_ambient = 0.01
spray_radius = 5
 
dt = 0.5
t_end = 120.0
steps = int(t_end / dt)


#define L-shape mask
#True=pat exists here, False=empty space
mask=np.zeros((ny,nx), dtype=bool)

#vertical bar of the L (left side, full height)
mask[0:40,0:20]=True

#horizontal bar of the L (bottom, full width)
mask[0:15, 0:40]=True

#temperature field: only initialize nodes that are part of the L
T=np.where(mask, T_initial, np.nan)

#zigzag path
row_step=4
path = []
for row in range(0, ny, row_step):
    cols = range(nx) if (row // row_step) % 2 == 0 else range(nx - 1, -1, -1)
    for col in cols:
        path.append((col, row))

steps_per_move = max(1, steps // len(path))

yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

#simulate
path_idx = 0
for step in range(steps):
    if step % steps_per_move == 0 and path_idx < len(path) - 1:
        path_idx += 1
 
    nozzle_x, nozzle_y = path[path_idx]
 
    # only spray if nozzle is over the actual part
    if mask[nozzle_y, nozzle_x]:
        dist = np.sqrt((xx - nozzle_x)**2 + (yy - nozzle_y)**2)
        spray_mask = (dist <= spray_radius) & mask
    else:
        spray_mask = np.zeros((ny, nx), dtype=bool)
 
    h_field = np.where(spray_mask, h_spray, h_ambient)

#only update temperature where part exists
    dTdt = -h_field * (T - T_ambient)
    T = np.where(mask, T + dTdt * dt, np.nan)

#plot
fig,ax=plt.subplots(figsize=(7,6))

#mask NaN values so empty space shows as grey
T_plot = np.ma.masked_invalid(T)
cmap = plt.cm.inferno.copy()
cmap.set_bad(color='lightgrey')
 
im = ax.imshow(T_plot, cmap=cmap, origin='lower')
ax.set_title(f'L-shaped part temperature after {t_end:.0f}s (zigzag spray)')
ax.set_xlabel('x (nodes)')
ax.set_ylabel('y (nodes)')
 
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Temperature (°C)')
 
plt.tight_layout()
plt.show()
 
#results
valid_temps = T[mask]
print(f"Hottest point:      {np.nanmax(T):.1f} C")
print(f"Coolest point:      {np.nanmin(T):.1f} C")
print(f"Temperature spread: {np.nanmax(T) - np.nanmin(T):.1f} C")
print(f"Average temp:       {np.nanmean(T):.1f} C")
