"""
Autodiff is not symbolic math and not finite differnces. JAX traces your function and builds an exact derivative computation
at the same time. Print boh value and gradient at several points so you can verify it manually. 

To summarize autodiff: JAX watches your function run and simultaneously builds the derivative. Not approximated, exact.
The spray cooling exampls hsows how dT/d(flow) get smaller at high flow rates-diminishing returns, which is real physics. 
"""

import jax
import jax.numpy as jnp

def f(x):
    return x**2

df=jax.grad(f)

for x_val in [1.0, 2.0, 3.0,-4.0]:
    g=df(x_val)
    print(f"x={x_val:5.1f} f(x)={f(x_val):6.1f} grad={g:6.1f} expected=2x{2*x_val:6.1f}")

print()



