"""
One fixed hot spot on a uniform 2D plate. Spray turns on for 5 seconds,
then off for 5 seconds, repeating. With conduction working, residual heat
from surrounding material conducts back into the cooled spot during the
off period — visible as a temperature rebound in the thermal history plot.

Physics:
    dT/dt = alpha * laplacian(T) - h_field * (T - T_ambient)

    h_field = h_spray when spray is ON and node is within spray radius
    h_field = h_ambient always elsewhere

Demonstrates: conduction rebound effect — proof that heat diffusion
between nodes is physically working.
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
spray_radius = 16
dt           = 0.5

#spray timing
t_on         = 5.0    #seconds spray is ON
t_off        = 5.0    #seconds spray is OFF
t_end        = 60.0   #total simulation time
steps        = int(t_end / dt)

#spray fixed at center of grid
cx, cy = nx // 2, ny // 2

#initial temperature
T_init = jnp.full((ny, nx), T_initial)

#precompute fixed spray mask
yy_np, xx_np = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
dist = np.sqrt((xx_np - cx)**2 + (yy_np - cy)**2)
sigma_spray = spray_radius / 2.0
spray_mask_np = np.exp(-dist**2 / (2 * sigma_spray**2)).astype(np.float32)
spray_mask_jax = jnp.array(spray_mask_np)

#spray schedule: precompute which steps have spray ON
#spray is ON during [0, t_on], OFF during [t_on, t_on+t_off], repeat
cycle = t_on + t_off
spray_on_schedule = np.array([
    (t % cycle) < t_on
    for t in np.arange(steps) * dt
], dtype=np.float32)
spray_on_jax = jnp.array(spray_on_schedule)

#JAX step function
@jax.jit
def step(carry, spray_on):
    T, step_idx = carry

    #h_field depends on whether spray is on this timestep
    h_field = h_ambient + spray_mask_jax * h_spray * spray_on

    #laplacian
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

    return (T_new, step_idx + 1), (T_new, spray_on)

#run simulation
print("Running simulation...")
_, (T_history, spray_schedule_out) = jax.lax.scan(
    step, (T_init, jnp.int32(0)), spray_on_jax
)
T_history          = np.array(T_history)
spray_schedule_out = np.array(spray_schedule_out)
print("Done.")

#animation + interactive clicking
clicked_points = []
t_axis = np.linspace(0, t_end, steps)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax         = axes[0]
ax_history = axes[1]

im = ax.imshow(T_history[0], cmap='inferno', origin='lower',vmin=T_ambient, vmax=T_initial)
fig.colorbar(im, ax=ax, label='Temperature (°C)')
ax.set_title('Pulsed spray cooling')
ax.set_xlabel('x (nodes)')
ax.set_ylabel('y (nodes)')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,color='white', fontsize=10)
spray_text = ax.text(0.02, 0.88, '', transform=ax.transAxes,color='cyan', fontsize=10)

#spray zone circle
circle = plt.Circle((cx, cy), spray_radius, color='cyan',fill=False, linewidth=2, label='Spray zone')
ax.add_patch(circle)
ax.legend(loc='upper right', fontsize=9)

ax_history.set_xlabel('Time (s)')
ax_history.set_ylabel('Temperature (°C)')
ax_history.set_title('Thermal history (click nodes on heatmap)')
ax_history.set_xlim(0, t_end)
ax_history.set_ylim(T_ambient - 10, T_initial + 10)
ax_history.grid(True, alpha=0.3)

#shade spray ON periods on history plot
for i in range(int(t_end / cycle) + 1):
    ax_history.axvspan(i * cycle, i * cycle + t_on,alpha=0.1, color='cyan', label='Spray ON' if i == 0 else '')
ax_history.legend(fontsize=8, loc='upper right')

skip = max(1, steps // 300)

def animate(frame):
    actual = frame * skip
    im.set_data(T_history[actual])
    t = actual * dt
    on = (t % cycle) < t_on
    time_text.set_text(f't = {t:.1f}s')
    spray_text.set_text('SPRAY ON' if on else 'SPRAY OFF')
    return [im, time_text, spray_text]

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
    frames=range(0, steps // skip),
    interval=30,
    blit=False
)

plt.tight_layout()
plt.show()

print(f"\nFinal results:")
print(f"  Peak temp:    {T_history[-1].max():.1f} C")
print(f"  Min temp:     {T_history[-1].min():.1f} C")
print(f"  Temp spread:  {T_history[-1].max() - T_history[-1].min():.1f} C")
print(f"  Avg temp:     {T_history[-1].mean():.1f} C")