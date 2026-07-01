import numpy as np
import matplotlib.pyplot as plt

#grid setup
nx, ny = 40, 40              # grid resolution
T_initial = 900.0            # initial part temp (C)
T_ambient = 25.0             # ambient temp (C)

h_spray = 0.15                # cooling coefficient where spray hits
h_ambient = 0.01              # cooling coefficient everywhere else (just air)

dt = 0.5
t_end = 120.0
steps = int(t_end / dt)

#temperature field
T = np.full((ny, nx), T_initial)

#spray location: circular patch in center
cx, cy = nx // 2, ny // 2
radius = 8
yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
print(f"yy={yy}")
print(f"xx={xx}")
dist_from_center = np.sqrt((xx - cx)**2 + (yy - cy)**2)
spray_mask = dist_from_center <= radius

#cooling coefficient field: high where spray hits, low elsewhere
h_field = np.where(spray_mask, h_spray, h_ambient)

#simulate
for step in range(steps):
    dTdt = -h_field * (T - T_ambient)
    T = T + dTdt * dt

#plot
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(T, cmap='inferno', origin='lower')
ax.set_title(f'2D part temperature after {t_end:.0f}s (localized spray)')
ax.set_xlabel('x')
ax.set_ylabel('y')

#outline the spray zone
circle = plt.Circle((cx, cy), radius, color='cyan', fill=False, linewidth=2, label='Spray zone')
ax.add_patch(circle)
ax.legend(loc='upper right')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Temperature (°C)')

plt.tight_layout()
plt.show()

print(f"Hottest point: {T.max():.1f} C")
print(f"Coolest point: {T.min():.1f} C")
print(f"Temperature spread: {T.max() - T.min():.1f} C")