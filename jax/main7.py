import jax
import jax.numpy as jnp
import time

# ─────────────────────────────────────────
# BLOCK 4: vmap
# ─────────────────────────────────────────

# ─────────────────────────────────────────
# 1. The problem vmap solves
#
# You write a function for ONE input.
# You have a BATCH of inputs.
# Without vmap: you write a for loop (slow).
# With vmap: JAX vectorizes it automatically (fast).
#
# vmap = "vectorized map"
# ─────────────────────────────────────────

# Function written for a single temperature value
def compute_loss(T):
    T_target = 500.0
    return (T - T_target) ** 2

# A batch of 5 temperature guesses
T_batch = jnp.array([400.0, 450.0, 500.0, 550.0, 600.0])

print("--- for loop vs vmap ---")

# Option A: for loop
losses_loop = []
for T in T_batch:
    losses_loop.append(compute_loss(T))
print(f"loop result:  {jnp.array(losses_loop)}")

# Option B: vmap — same function, no loop needed
batched_loss = jax.vmap(compute_loss)
losses_vmap = batched_loss(T_batch)
print(f"vmap result:  {losses_vmap}")

# same output, but JAX runs all 5 in parallel

# ─────────────────────────────────────────
# 2. vmap with a function that takes arrays
#
# More realistic: your function operates on
# a full temperature field (array), and you
# have a batch of different fields to evaluate.
# ─────────────────────────────────────────
def field_loss(T_field):
    # T_field is a 1D array of temperatures across N nodes
    T_target = 500.0
    return jnp.mean((T_field - T_target) ** 2)

# Simulate 4 different temperature fields, each with 6 nodes
T_fields = jnp.array([
    [480.0, 490.0, 500.0, 510.0, 520.0, 530.0],  # field 0
    [460.0, 470.0, 480.0, 490.0, 500.0, 510.0],  # field 1
    [500.0, 500.0, 500.0, 500.0, 500.0, 500.0],  # field 2 — perfect
    [550.0, 560.0, 570.0, 580.0, 590.0, 600.0],  # field 3 — too hot
])

batched_field_loss = jax.vmap(field_loss)
losses = batched_field_loss(T_fields)

print("\n--- vmap over temperature fields ---")
for i, loss in enumerate(losses):
    print(f"field {i}: loss = {loss:.2f}")

# ─────────────────────────────────────────
# 3. vmap + grad together
#
# This is where it gets powerful.
# Compute gradients for a whole batch at once.
# In your spray cooling work: evaluate how
# sensitive the loss is to temperature at
# every node, for many different initial conditions.
# ─────────────────────────────────────────

grad_fn = jax.grad(field_loss)              # grad for one field
batched_grad = jax.vmap(grad_fn)            # grad for a batch of fields

grads = batched_grad(T_fields)

print("\n--- vmap + grad: gradient for each field ---")
for i, g in enumerate(grads):
    print(f"field {i} gradients: {jnp.round(g, 2)}")

# Each row tells you: for that temperature field,
# how does loss change at each node?
# Negative = that node is below target (need more heat/less cooling)
# Positive = that node is above target (need less heat/more cooling)

# ─────────────────────────────────────────
# 4. Speed comparison
# ─────────────────────────────────────────

large_batch = jax.random.normal(jax.random.PRNGKey(0), shape=(1000, 50)) + 500.0

batched_loss_jit = jax.jit(jax.vmap(field_loss))
batched_loss_jit(large_batch).block_until_ready()  # warmup

t0 = time.time()
result = batched_loss_jit(large_batch).block_until_ready()
t1 = time.time()

print(f"\n--- speed: 1000 fields x 50 nodes ---")
print(f"vmap + jit time: {t1 - t0:.5f} sec")
print(f"mean loss across batch: {jnp.mean(result):.4f}")
