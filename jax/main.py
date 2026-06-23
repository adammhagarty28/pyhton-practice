"""
jax is a python library developed by google. On the surface it looks identical to numpy-same array opeations, syntax, math.
However, it has three capabilities numpy completely lacks:

jit (just in time compilaiton)-takes your python function and compiles it to run on GPU automatically. Same code, dramatically faster
grad-automatically computes the derivative of any function with respec to any input.
vmap-runs a function over a batch of inputs simultaneously without writing loops

Everything I have been doing, heat equation, cooling sim- numpy runs it forward. It can tell me given these inputs, here are the outputs
. JAX can additionally tell me if I want a specfiic output, which direciton should I adjust my inputs? That is grad. In 
spray cooling terms: instead of tuning cooling rate and hoping the plate reaches safe temperature, JAX can tell me exactly how
to change the cooling rate to hit the target. \

JAX arrays work identically to numpy. Alomst every function has a JAX equivalent, so eveyrthing transfers directly

There is one imortant different and that is that jax arrays are immutable so i cannot modify arrays in place.


jax.grad()-computes the derivative of a function with respect to its inputs. This is the while reason JAX exists

jax.jit()-wraps a function to compile it for GPU/CPU speed. Same result, much faster

jnp (jax.numpy)-identical to numpy but jax aware. Eveyr numpy thing I have learned transfers directly: jnp.array(),jnp.zeros(),jnp.mean(),etc.

jax.value_and_grad()-returns both the function output and its gradient in one call. 

"""
import jax # loads the top-level jax package, which gives acces to core tools like jax.grad() and jax.jit(). acess them with jax. prefix
import jax.numpy as jnp #loads numpy-equivalent submodule of JAX

#both are libraries are needed becase jax gives grad and jit, while jnp ives array math functions

def square(x):
    return x**2

grad_square= jax.grad(square)

x=3.0
print("f(x) =", square(x))
print("f(x) =", grad_square(x))
