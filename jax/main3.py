import jax.numpy as jnp
import matplotlib.pyplot as plt

T = jnp.array([
    [400.0, 500.0, 400.0],
    [500.0, 900.0, 500.0],
    [400.0, 500.0, 400.0]
])

cooling_rate = 50.0
n_steps = 10
snapshots = []
snapshot_labels = []

for step in range(n_steps):
    T-=cooling_rate
    T=jnp.clip(T,20.0, None)

    if step ==0 or step ==4 or step==9:
        snapshots.append(T)
        snapshot_labels.append(f"step {step}")
    

fig,axes=plt.subplots(1,3,figsize=(10,3))
vmin=snapshots[-1].min()
vmax=snapshots[0].max()

for idx, ax in enumerate(axes):
    im =ax.imshow(snapshots[idx], cmap="hot",vmin=vmin, vmax=vmax)
    ax.set_title(snapshot_labels[idx])
    ax.set_xlabel("x zone")
    ax.set_ylabel("y zone")
    plt.colorbar(im, ax=ax, label="Temp (C)")

    plt.tight_layout()
    plt.show()