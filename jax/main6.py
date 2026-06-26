import jax
import jax.numpy as jnp
import time

"""
jit and value_and_grad

1.) jax.jit-compile a function for speed

jit="just in time compilation
first call: jax compiles the function (slow)
every call after: runs the compiled version (fast)
Don't change the function - just wrap it
"""

def slow_sum(x):
    return jnp.sum(x**2)

fast_sum=jax.jit(slow_sum)

x=jnp.ones(1_000_000)

#first call: includes compile time
t0=time.time()
result=fast_sum(x).block_until_ready()
t1=time.time()
print(f"jit first call: {t1-t0:.4f} sec (compiled)")

# Second call — compiled, much faster
t0 = time.time()
result = fast_sum(x).block_until_ready()
t1 = time.time()
print(f"jit second call: {t1 - t0:.4f} sec  (compiled)")


print(f"result: {result}") 

"""
value_and_grad-loss and gradient together

grad() only returns the gradient. 
value_and_grad() returns both:
-the loss vale (how wrong you were)
-the gradient (whcih direction to move)

This is what optimizers actually see
"""

T_target=500
def loss(T):
    return (T-T_target) **2

loss_and_grad=jax.value_and_grad(loss)

print("\n---value_and_grad---")
T=600.0
val, grad=loss_and_grad(T)
print(f"T={T}")
print(f"loss value= {val}")
print(f"gradient = {grad}")

"""
Put it together: a simple update step

This is gradient descent in its simplest form
new_T=old_T=learning_rate*gradient

learning_rate controls step size
small lr=slow but stable
large lr=fast but can overshoot

"""

lr=.01
T=620.0

print("\n--- gradient descent: 10 steps ---")
print(f"{'step':<6} {'T':>8} {'loss':>10} {'grad':>12}")
print("-" * 44)

for step in range(10):
    val, grad=loss_and_grad(T)
    print(f"{step:<6} {T:>8.4f} {val:>10.4f} {grad:>12.4f}")
    T=T-lr*grad



print(f"\nfinal T={T:.4f} (target was {T_target})")