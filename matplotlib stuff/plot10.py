import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-3,3,100)
y=np.linspace(-3,3,100)
X,Y=np.meshgrid(x,y)
Z=np.exp(-(X**2+Y**2)) # Gaussian hot spot

fig, ax=plt.subplots()
im=ax.imshow(
    Z,
    extent=[-3,3,-3,3],
    cmap='hot'
)
fig.colorbar(im,ax=ax, label="temperature (normalized)")
ax.set_title("2D Gaussian Heat source")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()