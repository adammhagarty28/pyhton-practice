import numpy as np
import matplotlib.pyplot as plt

state=np.zeros((10,10))

snapshots=[]

snapshots = []                         # will hold copies of state over time

for step in range(5):
    # each step: add heat at a random spot
    row = np.random.randint(0, 10)
    col = np.random.randint(0, 10)
    state[row, col] += 1.0

    snapshots.append(state.copy())    # save a copy — NOT a reference

# snapshots[0] has 1 hot spot, snapshots[4] has 5
# plot all 5 side by side
fig, axes = plt.subplots(1, 5, figsize=(12, 6))

for i, ax in enumerate(axes):
    im = ax.imshow(snapshots[i], cmap='hot', vmin=0, vmax=5)
    ax.set_title(f'step {i+1}')
    ax.axis('off')

fig.colorbar(im, ax=axes, label='heat')
plt.show()