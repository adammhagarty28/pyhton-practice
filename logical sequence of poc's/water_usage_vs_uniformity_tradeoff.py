import numpy as np
import matplotlib.pyplot as plt
 
#grid setup
nx, ny = 40, 40
T_ambient = 25.0
T_initial = 900.0
 
dt = 0.5
t_end = 120.0
steps = int(t_end / dt)
 
spray_radius = 5

#L-shape mask
mask = np.zeros((ny, nx), dtype=bool)
mask[0:40, 0:20] = True   #vertical bar (left, full height)
mask[0:15, 0:40] = True   #horizontal bar (bottom, full width)

 
#thickness map: controls how fast each node cools
#thicker regions resist cooling (lower effective h)
#vertical bar is thick, horizontal bar is thin
thickness = np.ones((ny, nx))
thickness[0:40, 0:20] = 2.5    #vertical bar: thick
thickness[0:15, 0:40] = 1.0    #horizontal bar: thin
#where there's no part, thickness doesn't matter
thickness[~mask] = 1.0

#cooling coefficients scaled by thickness
h_spray_base = 0.15
h_ambient_base = 0.01
h_spray = h_spray_base / thickness
h_ambient = h_ambient_base / thickness

#temperature field
T = np.where(mask, T_initial, np.nan)

#water usage tracker
water_used = np.zeros((ny, nx))

#zigzag path: only visit nodes on the actual part
row_step = 4
path = []
for row in range(0, ny, row_step):
    cols = range(nx) if (row // row_step) % 2 == 0 else range(nx - 1, -1, -1)
    for col in cols:
        if mask[row, col]:   #only add to path if node is on the part
            path.append((col, row))
 
steps_per_move = max(1, steps // len(path))
 
yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

#simulate with adaptive spray
#spray intensity depends on local temperature:
#above 600C: full spray
#300-600C:   half spray
#below 300C: no spray
path_idx = 0
for step in range(steps):
    if step % steps_per_move == 0 and path_idx < len(path) - 1:
        path_idx += 1
 
    nozzle_x, nozzle_y = path[path_idx]
    nozzle_temp = T[nozzle_y, nozzle_x]
 
    #decide spray intensity based on temperature at nozzle location
    if nozzle_temp > 600:
        spray_intensity = 1.0
    elif nozzle_temp > 300:
        spray_intensity = 0.5
    else:
        spray_intensity = 0.0
 
    dist = np.sqrt((xx - nozzle_x)**2 + (yy - nozzle_y)**2)
    spray_mask = (dist <= spray_radius) & mask
 
    #apply spray intensity to cooling coefficient
    h_field = np.where(spray_mask, h_spray * spray_intensity, h_ambient)
 
    #track water usage
    if spray_intensity > 0:
        water_used += spray_mask * spray_intensity
 
    dTdt = -h_field * (T - T_ambient)
    T = np.where(mask, T + dTdt * dt, np.nan)


#plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
 
#left plot: final temperature
cmap = plt.cm.inferno.copy()
cmap.set_bad(color='lightgrey')
 
T_plot = np.ma.masked_invalid(T)
im1 = axes[0].imshow(T_plot, cmap=cmap, origin='lower')
axes[0].set_title(f'Final temperature after {t_end:.0f}s')
axes[0].set_xlabel('x (nodes)')
axes[0].set_ylabel('y (nodes)')
fig.colorbar(im1, ax=axes[0], label='Temperature (°C)')

# overlay nozzle path
path_x = [p[0] for p in path]
path_y = [p[1] for p in path]
axes[0].plot(path_x, path_y, color='cyan', linewidth=0.8, alpha=0.5, label='Nozzle path')
axes[0].legend(loc='upper right', fontsize=9)

 
#right plot: water usage per node
water_plot = np.ma.masked_array(water_used, mask=~mask)
cmap2 = plt.cm.Blues.copy()
cmap2.set_bad(color='lightgrey')
im2 = axes[1].imshow(water_plot, cmap=cmap2, origin='lower')
axes[1].set_title('Water usage per node (spray steps)')
axes[1].set_xlabel('x (nodes)')
axes[1].set_ylabel('y (nodes)')
fig.colorbar(im2, ax=axes[1], label='Spray intensity accumulated')
 
plt.tight_layout()
plt.show()
 
#results
print(f"Hottest point:      {np.nanmax(T):.1f} C")
print(f"Coolest point:      {np.nanmin(T):.1f} C")
print(f"Temperature spread: {np.nanmax(T) - np.nanmin(T):.1f} C")
print(f"Average temp:       {np.nanmean(T):.1f} C")
print(f"Total water used:   {water_used[mask].sum():.1f} units")
 

