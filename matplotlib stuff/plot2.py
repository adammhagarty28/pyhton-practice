import numpy as np
import matplotlib.pyplot as plt

positions = np.linspace(0, 1, 8)
temperatures_a = np.array([425, 405, 385, 395, 435, 415, 365, 345])
temperatures_b = np.array([300, 320, 340, 310, 290, 330, 350, 360])

# --- Figure 1: basic line plot with style options ---
plt.figure()
plt.plot(positions, temperatures_a, color="red", linestyle="-", marker="o", label="Zone A")
plt.plot(positions, temperatures_b, color="blue", linestyle="--", marker="s", label="Zone B")
plt.title("Temperature Along Rod")
plt.xlabel("Position")
plt.ylabel("Temperature (C)")
plt.legend()
plt.grid(True)
plt.savefig("plot_lines.png")

# --- Figure 2: scatter plot (good for point clouds) ---
plt.figure()
plt.scatter(positions, temperatures_a, color="green", marker="^", label="Zone A points")
plt.title("Scatter of Zone A Temperatures")
plt.xlabel("Position")
plt.ylabel("Temperature (C)")
plt.legend()
plt.savefig("plot_scatter.png")

# --- Figure 3: line width and a third dataset ---
temperatures_c = temperatures_a - temperatures_b
plt.figure()
plt.plot(positions, temperatures_c, color="purple", linestyle=":", linewidth=3, label="Difference (A - B)")
plt.title("Temperature Difference")
plt.xlabel("Position")
plt.ylabel("Temp Difference (C)")
plt.legend()
plt.grid(True)
plt.savefig("plot_difference.png")

plt.show()