

import jax
import jax.numpy as jnp

#JAX arrays and grad

x=jnp.array([1.0,2.0,3.0])
print("x= ", x)
print("x*2= ",x*2)
print("sum of x= ", jnp.sum(x))

#jax.grad-auto diff

#grad() takes a function and returns a new function thta computes the derivatie of the original

def f(x):
    return x**2

df=jax.grad(f)

print("\n---grad example:f(x)=x^2---")
print("f(3.0) = ", f(3.0))
print("f(3.0) = ", df(3.0))
print("f(5.0) = ", df(5.0))

"""
Try it with someting closer to physics

Temperature loss: how wrong is our temp guess?
loss(T)=(T-T_target)^2
derivative tells us which direction to move T
"""

T_target=500.0 #we want 500k

def temp_loss(T):
    return (T-T_target)**2

dloss=jax.grad(temp_loss)
 
print("\n--- grad example: temperature loss ---")
T_guess = 600.0
print(f"T_guess = {T_guess}")
print(f"loss at T=600:        {temp_loss(T_guess)}")   # expected: 10000
print(f"gradient at T=600:    {dloss(T_guess)}")       # expected: 200.0  (positive = too hot, move down)
 
T_guess2 = 400.0
print(f"\nT_guess = {T_guess2}")
print(f"loss at T=400:        {temp_loss(T_guess2)}")  # expected: 10000
print(f"gradient at T=400:    {dloss(T_guess2)}")      # expected: -200.0 (negative = too cold, move up)
 
