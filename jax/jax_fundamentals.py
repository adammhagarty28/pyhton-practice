"""
For JAX, jax.value_nad_grad ocmbined iwth time-stepping loop

Few Key notes on jax:

loss, grads=jax.value_and_grads(f)(x):
This takes a funciton f and returns both its output value and its gradient with respect ot x in one single pass
through the funciton. Without it you would have to call the funcito twice, once to get the value, once to get the 
gradient, which is wasteful. In every real optimization loop you need both, so value_and_grad is what you use in practice
instead of grad alone. The output is alwyas a tuple: (value, gradient), which is why you unpack it into two variables. 


grad_f-jax.grad(f)
gradient=grad_f(x)
retuns a new funciton that computes the gradient of f with respect to its first argument by default. If you want 
the gradient with respect to a different argumnet, use argnums

grad_f=jax.grad(f, argnums=1)
gradient with respect to second argument
Important rule:the function must return a scalar (single number). If it returns an array, JAX does not know what to 
differentiate. Your loss function always returns a single number (like jnp.sum(...) for this reason. )

"""

import jax
import jax.numpy as jnp

n_points=5
target_temp=100
n_steps=20
alpha=.1

def simulate_and_loss(spray_rates):
    temps=jnp.array([300, 400, 500, 400, 300])

    for step in range(n_steps):
        left=jnp.concatenate([temps[:1], temps[:-1]]) #join sequence of arrays together along existing axis
        right=jnp.concatenate([temps[1:],temps[-1:]])
        diffusion=alpha*(left+right-2*temps)

        temps=temps+diffusion-spray_rates
        temps=jnp.clip(temps, 20, None)
    return jnp.sum((temps-target_temp)**2)

spray_rates=jnp.ones(n_points)*10
learning_rate=.001

print(f"{"Iteration":<12} {"Loss":<15} {"Spray Rates"}")
print("-"*60)

for iteration in range(30):
    loss,grads=jax.value_and_grad(simulate_and_loss)(spray_rates)
    spray_rates=spray_rates-learning_rate*grads

    if iteration %5==0:
        print(f"{iteration:<12} {loss:<15.1f} {jnp.round(spray_rates, 2)}")

print(f"\nFinal spray rates:{jnp.round(spray_rates, 3)}")
print(f"Final Loss: {simulate_and_loss(spray_rates):.2f}")