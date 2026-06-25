import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 200)

plt.plot(x, np.sin(x), label='sin(x)',  color='steelblue')
plt.plot(x, np.cos(x), label='cos(x)',  color='coral',     linestyle='--')
plt.plot(x, np.tan(x), label='tan(x)',  color='seagreen',  linestyle=':')

plt.ylim(-2, 2)          # clip tan's huge spikes
plt.axhline(0, color='black', linewidth=0.5)  # zero line
plt.legend()
plt.title('Trig functions')
plt.show()