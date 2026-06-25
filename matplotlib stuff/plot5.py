import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0.10,100)
y=x**2

plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("x squared")
plt.title("My first labeled plot")
plt.show()