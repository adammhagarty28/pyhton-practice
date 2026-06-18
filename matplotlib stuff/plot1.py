import numpy as np
import matplotlib.pyplot as plt

positions=np.linspace(0,1,8)
temperatures=np.array([425, 405, 385, 395, 435, 415, 365, 345])

plt.plot(positions, temperatures)
plt.show()