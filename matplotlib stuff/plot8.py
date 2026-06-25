import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 4 * np.pi, 300)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 3))

axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('sin(x)')

axes[0, 1].plot(x, np.cos(x), color='coral')
axes[0, 1].set_title('cos(x)')

axes[1, 0].plot(x, x * np.sin(x), color='seagreen')
axes[1, 0].set_title('x · sin(x)')

axes[1, 1].plot(x, np.exp(-x / 5) * np.sin(x), color='purple')
axes[1, 1].set_title('damped oscillation')

fig.tight_layout()   # prevents titles from overlapping
plt.show()