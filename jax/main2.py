"""
cooling_loss if a funtion that simulates cooling a plate and returns how far the final temperature is from the targe-squared.
That squared difference is called a loss funciton, its zero when you hit the target exactly, and gets bigger the further you ar from it

"""

import jax
import jax.numpy as jnp

def cooling_loss(cooling_rate, target_temp, iniital_temp, steps):
    temp=initial_temp
    for step in range(steps):
        temp-=cooling_rate
    return (temp-target_temp) ** 2

grad_cooling=jax.grad(cooling_loss)

initial_temp = 900.0
target_temp = 100.0
cooling_rate = 5.0
steps = 10

loss = cooling_loss(cooling_rate, target_temp, initial_temp, steps)
grad = grad_cooling(cooling_rate, target_temp, initial_temp, steps)

print(f"Final temp: {initial_temp - cooling_rate * steps:.1f} C")
print(f"Loss (how far from target): {loss:.1f}")
print(f"Gradient: {grad:.4f}")
print(f"Meaning: increase cooling_rate by 1 and loss changes by {grad:.2f}")

print("\n--- Gradient Descent: tuning cooling rate ---")
cooling_rate = 5.0
learning_rate = .005

for iteration in range(5):
    loss = cooling_loss(cooling_rate, target_temp, initial_temp, steps)
    grad = grad_cooling(cooling_rate, target_temp, initial_temp, steps)
    cooling_rate = cooling_rate - learning_rate * grad
    final_temp = initial_temp - cooling_rate * steps
    print(f"Iteration {iteration}: cooling_rate={cooling_rate:.4f} | final_temp={final_temp:.1f} C | loss={loss:.1f}")