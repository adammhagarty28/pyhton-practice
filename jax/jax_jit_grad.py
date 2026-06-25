"""
Demonstrates JAX core concepts: jit, grad, and vmap on a spray cooling mesh. 
Section 1: jit speed comparison
Section 2: multi-zone mesh gradient descent optimiing cool rates
Physics: 3x3 plate mesh, heat diffusion between neighbors, spray cooling sink
"""

import jax
import jax.numpy as jnp
import time

#Section 1: jit speed comparison

def simulate_cooling(temps, cooling_rate, steps):
    for step in range(steps):
        temps-=cooling_rate
    return jnp.mean(temps)

jit_simulate=jax.jit(simulate_cooling, static_argnums=(2,))

initial_temps = jnp.array([900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0, 550.0, 500.0])
cooling_rate = 10.0
steps = 100 

#without jit
start=time.time()
for step in range(1000):
    result=simulate_cooling(initial_temps, cooling_rate, steps)
end=time.time()
print(f"Without jit: {end-start:.4f} seocnds | result: {result:.1f} C")

# With jit 
start=time.time()
for step in range(1000):
    result=jit_simulate(initial_temps, cooling_rate, steps)
end=time.time()
print(f"With jit: {end-start:.4f} seconds| result: {result:.1f} C")

#Section 2: multi-zone mesh gradient descent

alpha=.05
target_temp=100.0
n_steps=20

def mesh_cooling_loss(cooling_rates, initial_temps, alpha, n_steps, target_temp):
    temps = initial_temps
    for _ in range(n_steps):
        # Heat diffusion using explicit slicing (no padding needed)
        diffusion = jnp.zeros((3, 3))
        diffusion = diffusion.at[1:-1, :].add(alpha * (temps[:-2, :] - temps[1:-1, :]))
        diffusion = diffusion.at[1:-1, :].add(alpha * (temps[2:, :] - temps[1:-1, :]))
        diffusion = diffusion.at[:, 1:-1].add(alpha * (temps[:, :-2] - temps[:, 1:-1]))
        diffusion = diffusion.at[:, 1:-1].add(alpha * (temps[:, 2:] - temps[:, 1:-1]))
        temps = temps + diffusion - cooling_rates
    return jnp.sum((temps - target_temp) ** 2)

grad_mesh = jax.grad(mesh_cooling_loss)

# 3x3 plate: hot center, cooler edges
initial_temps_2d = jnp.array([
    [400.0, 450.0, 400.0],
    [450.0, 900.0, 450.0],
    [400.0, 450.0, 400.0]
])

# Start with uniform cooling rates across all zones
cooling_rates = jnp.ones((3, 3)) * 5.0
learning_rate = 0.001

print(f"\n--- Mesh Gradient Descent: optimizing 9 zone cooling rates ---")
print(f"{'Iteration':<12} {'Loss':<15} {'Max Rate':<12} {'Min Rate':<12} {'Center Rate':<12}")
print("-" * 65)

for iteration in range(50):
    loss = mesh_cooling_loss(cooling_rates, initial_temps_2d, alpha, n_steps, target_temp)
    grads = grad_mesh(cooling_rates, initial_temps_2d, alpha, n_steps, target_temp)
    cooling_rates = cooling_rates - learning_rate * grads

    if iteration % 5 == 0:
        print(f"{iteration:<12} {loss:<15.1f} {jnp.max(cooling_rates):<12.4f} {jnp.min(cooling_rates):<12.4f} {cooling_rates[1,1]:<12.4f}")

print(f"\nFinal cooling rates (per zone):")
print(jnp.round(cooling_rates, 3))

if __name__ == "__main__":
    print("\nScript complete.")