"""
This script simulates a 1D rod with 10 points being cooled by spray cooling using the actual discretized heat equaiton,
calculating both the temperature evolution over time and its spatial derivatives (slope and curvature) using np.diff(). 
It tracks summary statistics across 30 time steps (mean temp, max temp, count of points still above a safe threshold),
then produces multiple plots: temperature snapshots over time, mean temp and hot-point-count history, and a stacked
plot showing temperature alongside its first and second derivatives at a chosen step. 

Overall: heat_equation_rod.py
1D spray cooling simulation using the discretized heat equation. 
Combines numpy (vectorized math, derivatives) and matplotlib(multiple plot types)

"""

import numpy as np
import matplotlib.pyplot as plt

n_points=10
length=1
positions=np.linspace(0, length, n_points)
spacing=positions[1]-positions[0]

temperatures=np.array([500,480,460,440,420,400,380,360,340,320],dtype=float)

alpha=0.05
spray_cooling=8.0
room_temp=25.0
n_steps=30

#Storage for history (so we can plot how things evolve over time)
temp_history=np.zeros((n_steps, n_points))
mean_history=np.zeros((n_steps))
hot_count_history=np.zeros(n_steps)

for step in range(n_steps):
    #first derivative measures how temperature changes point-to-point (the gradient)
    first_derivative=np.diff(temperatures)/spacing

    #Second derivative measures how that grainet itself changes (drives diffusion)
    second_derivative=np.diff(first_derivative)/spacing

    #diffusion only applies to interior points (no neighbores past the ends)
    diffusion_term=np.zeros(n_points)
    diffusion_term[1:-1]=alpha*second_derivative

    #update temperatures becuase diffusion spreads heat iternally, spray removes heat overall
    temperatures=temperatures+diffusion_term-spray_cooling
    temperatures=np.clip(temperatures, room_temp, None)
    
    #store this step's full temperature array and summary stats
    temp_history[step]=temperatures
    mean_history[step]=temperatures.mean()
    hot_count_history[step]=np.sum(temperatures > room_temp +50)

# Print terminal output every 5 steps
    if step % 5 == 0:
        print(f"Step {step:>3} | Max: {temperatures.max():.1f}C | Mean: {temperatures.mean():.1f}C | Min: {temperatures.min():.1f}C | Hot points: {int(hot_count_history[step])}")

# After loop ends
print("\n--- Final State ---")
print(f"Max temp:  {temp_history[-1].max():.1f} C")
print(f"Mean temp: {temp_history[-1].mean():.1f} C")
print(f"Min temp:  {temp_history[-1].min():.1f} C")
print(f"Hot points remaining: {int(hot_count_history[-1])}")

# Print first and second derivative at chosen step
chosen_step = 15
fd = np.diff(temp_history[chosen_step]) / spacing
sd = np.diff(fd) / spacing
print(f"\n--- Derivatives at Step {chosen_step} ---")
print(f"First derivative (dT/dx):  {np.round(fd, 2)}")
print(f"Second derivative (d2T/dx2): {np.round(sd, 2)}")
print(f"Most negative curvature at position: {positions[1:-1][np.argmin(sd)]:.2f}m")
print(f"Most positive curvature at position: {positions[1:-1][np.argmax(sd)]:.2f}m")


# --- Plot 1: temperature snapshots at different times ---
plt.figure()
for step in [0, 10, 20, 29]:
    plt.plot(positions, temp_history[step], marker="o", label=f"Step {step}")
plt.title("Temperature Along Rod at Different Times")
plt.xlabel("Position")
plt.ylabel("Temperature (C)")
plt.legend()
plt.grid(True)

# --- Plot 2: mean temperature over time ---
plt.figure()
plt.plot(range(n_steps), mean_history, color="red", linestyle="-")
plt.title("Mean Rod Temperature Over Time")
plt.xlabel("Step")
plt.ylabel("Mean Temperature (C)")
plt.grid(True)

# --- Plot 3: number of hot points over time ---
plt.figure()
plt.plot(range(n_steps), hot_count_history, color="orange", marker="s")
plt.title("Points Still Above Safe Threshold Over Time")
plt.xlabel("Step")
plt.ylabel("Number of Hot Points")
plt.grid(True)

# --- Plot 4: temperature, first derivative, second derivative (stacked) at one step ---
chosen_step = 15
fd = np.diff(temp_history[chosen_step]) / spacing
sd = np.diff(fd) / spacing

fig, axes = plt.subplots(3, 1)

axes[0].plot(positions, temp_history[chosen_step], color="blue", marker="o")
axes[0].set_title(f"Temperature at Step {chosen_step}")
axes[0].set_ylabel("Temp (C)")
axes[0].grid(True)

axes[1].plot(positions[:-1], fd, color="green", marker="^")
axes[1].set_title("First Derivative (dT/dx)")
axes[1].set_ylabel("Slope")
axes[1].grid(True)

axes[2].plot(positions[1:-1], sd, color="purple", marker="s")
axes[2].set_title("Second Derivative (d2T/dx2)")
axes[2].set_xlabel("Position")
axes[2].set_ylabel("Curvature")
axes[2].grid(True)

plt.tight_layout()

plt.show()