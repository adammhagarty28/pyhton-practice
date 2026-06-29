import numpy as np
import matplotlib.pyplot as plt

# --- parameters ---
T_initial = 900.0       # initial part temperature (C), typical after heat treatment
T_ambient = 25.0        # ambient / coolant temperature (C)
h = 0.05                # cooling coefficient (1/s), tunable
dt = 0.5                # time step (s)
t_end = 120.0           # total simulation time (s)

# --- time array ---
t = np.arange(0, t_end + dt, dt)

# --- simulate cooling: Newton's law of cooling ---
# dT/dt = -h * (T - T_ambient)
# solved numerically with forward Euler
T = np.zeros(len(t))
T[0] = T_initial

for i in range(1, len(t)):
    dTdt = -h * (T[i-1] - T_ambient)
    T[i] = T[i-1] + dTdt * dt

# --- plot ---
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(t, T, color='firebrick', linewidth=2, label='Part temperature')
ax.axhline(T_ambient, color='steelblue', linewidth=1.2, linestyle='--', label=f'Ambient ({T_ambient}°C)')

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Temperature (°C)', fontsize=12)
ax.set_title('POC 1: Single point cooling (Newton\'s law)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# annotate start and rough equilibrium
ax.annotate(f'Start: {T_initial}°C', xy=(0, T_initial), xytext=(5, T_initial - 60),
            fontsize=10, color='firebrick')

plt.tight_layout()
plt.show()
print(f"Start temp:  {T[0]:.1f} C")
print(f"End temp:    {T[-1]:.1f} C after {t_end}s")