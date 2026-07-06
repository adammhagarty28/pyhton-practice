"""
POC: 2D Heat Diffusion in JAX
 
Simulates heat diffusing across a flat 2D grid from a Gaussian initial
temperature distribution. Uses JAX (jit + lax.scan) for GPU-accelerated
time integration. Includes animated heatmap showing temperature evolution
over time and interactive point-clicking to view thermal history at any node.
 
Physics: transient heat diffusion equation + surface convection
    dT/dt = alpha * laplacian(T) - (h / rho_c) * (T - T_ambient)
    alpha = k / (rho * c)  -- thermal diffusivity
 
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

#jax.config.updatte("jax_enable_x64", True)

#parameters
nx, ny = 40, 40
Lx, Ly = 0.04, 0.04 # physical size of grid in meters (40mm x 40mm)
dx = Lx / (nx - 1) # node spacing in meters

# material: carbon steel
k= 50.0 # thermal conductivity W/mK
rho= 7800.0 # density kg/m3
c= 500.0 # specific heat J/kgK
rho_c= rho * c # volumetric heat capacity
alpha= k / rho_c # thermal diffusivity m2/s

# convection to ambient air (no spray yet)
h= 10.0 # convective heat transfer coefficient W/m2K
T_ambient= 25.0 #ambient temperature C

dt=0.001 #timestep in seconds
t_end=10.0 #total simulation time in seconds

steps=int(t_end/dt) #mumber of timesteps

#stability check: must be less than .25 for explicity scheme
stability = alpha * dt / dx**2
print(f"Stability number: {stability:.4f}  (must be < 0.25)")
if stability >= 0.25:
    raise ValueError("Unstable! Reduce dt or increase dx.")

#Gaussian initial condition
T_peak=900.0 #peak temperature at center C
sigma_g=.006 #Gaussian widt in meters (6mm)

#initial temperature field: Gaussian hot spot centered on grid
x_vals = jnp.linspace(0, Lx, nx)
y_vals = jnp.linspace(0, Ly, ny)
XX, YY = jnp.meshgrid(x_vals, y_vals)   # shape (ny, nx)
 
cx, cy = Lx / 2, Ly / 2
T_init = T_ambient + (T_peak - T_ambient) * jnp.exp(
    -((XX - cx)**2 + (YY - cy)**2) / (2 * sigma_g**2)
)

@jax.jit
def step(T, _):
    """One forward Euler timestep of the heat equation."""
 
    #Laplacian via finite differences (interior nodes only)
    d2Tdx2 = jnp.zeros_like(T)
    d2Tdy2 = jnp.zeros_like(T)
 
    d2Tdx2 = d2Tdx2.at[1:-1, :].set(
        (T[2:, :] - 2*T[1:-1, :] + T[:-2, :]) / dx**2
    )
    d2Tdy2 = d2Tdy2.at[:, 1:-1].set(
        (T[:, 2:] - 2*T[:, 1:-1] + T[:, :-2]) / dx**2
    )
    laplacian = d2Tdx2 + d2Tdy2
 
    #full heat equation: diffusion + surface convection
    dTdt = alpha * laplacian - (h / rho_c) * (T - T_ambient)
 
    T_new = T + dTdt * dt
    return T_new, T_new   # (carry, output)
 

# run simulation via lax.scan

print("Running simulation...")
T_final, T_history = jax.lax.scan(step, T_init, None, length=steps)
T_history = np.array(T_history)   # shape: (steps, ny, nx)
print("Done.")
 

# animation

# store clicked point thermal histories
clicked_points = []
 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax_heat = axes[0]
ax_history = axes[1]
 
# heatmap
im = ax_heat.imshow(
    T_history[0], cmap='inferno', origin='lower',
    vmin=T_ambient, vmax=T_peak,
    extent=[0, nx, 0, ny]
)
cbar = fig.colorbar(im, ax=ax_heat)
cbar.set_label('Temperature (°C)')
ax_heat.set_title('Temperature field (click any point)')
ax_heat.set_xlabel('x (nodes)')
ax_heat.set_ylabel('y (nodes)')
time_text = ax_heat.text(0.02, 0.95, '', transform=ax_heat.transAxes,color='white', fontsize=10)
 
# thermal history plot (right side)
ax_history.set_xlabel('Time (s)')
ax_history.set_ylabel('Temperature (°C)')
ax_history.set_title('Thermal history (click nodes on heatmap)')
ax_history.set_xlim(0, t_end)
ax_history.set_ylim(T_ambient - 10, T_peak + 10)
ax_history.grid(True, alpha=0.3)
t_axis = np.linspace(0, t_end, steps)
 
# scatter for clicked points
scatter = ax_heat.scatter([], [], c='cyan', s=40, zorder=5)
 
def animate(frame):
    im.set_data(T_history[frame])
    time_text.set_text(f't = {frame * dt:.1f}s')
    return [im, time_text]
 
def on_click(event):
    if event.inaxes != ax_heat:
        return
    # convert click coords to node indices
    col = int(round(event.xdata))
    row = int(round(event.ydata))
    col = np.clip(col, 0, nx - 1)
    row = np.clip(row, 0, ny - 1)
 
    # get thermal history at this node
    node_history = T_history[:, row, col]
    label = f'({col}, {row})'
    ax_history.plot(t_axis, node_history, linewidth=1.5, label=label)
    ax_history.legend(fontsize=8, loc='upper right')
    ax_history.set_title('Thermal history (click nodes on heatmap)')
 
    clicked_points.append((col, row))
    xs = [p[0] for p in clicked_points]
    ys = [p[1] for p in clicked_points]
    scatter.set_offsets(np.c_[xs, ys])
 
    fig.canvas.draw()
 
fig.canvas.mpl_connect('button_press_event', on_click)
 
ani = animation.FuncAnimation(
    fig, animate,
    frames=range(0, steps, 100), 
    interval=30,
    blit=False
)
 
plt.tight_layout()
plt.show()
 
print(f"\nFinal results:")
print(f"  Peak temp:    {T_history[-1].max():.1f} C")
print(f"  Min temp:     {T_history[-1].min():.1f} C")
print(f"  Temp spread:  {T_history[-1].max() - T_history[-1].min():.1f} C")