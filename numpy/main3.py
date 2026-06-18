"""
numpy basic file-practice uisng various built-in functions 
"""
import numpy as np

temperatures=np.array([500, 480, 460, 470, 510, 490, 440, 420])
print("Initial temperatures:", temperatures)
print("Shape:", temperatures.shape)

print("Mean temp:",temperatures.mean())
print("Max temp:",temperatures.max())
print("Min temp:", temperatures.min())
print("Total sum:", temperatures.sum())

room_temp=25
cooling_rate=15

new_temperatures=np.zeros(temperatures.shape)
print("\nEmpty placeholder array:", new_temperatures)

for step in range(5):
    temperatures-=cooling_rate
    still_hot=temperatures>room_temp
    num_hot=np.sum(still_hot)
    print(f"\n----Step {step}---")
    print("Temperatures:", temperatures)
    print("Number still above room temp", num_hot)
    print("Hottest point",temperatures.max())
    print("Coldest point",temperatures.min())

positions=np.linspace(0,1,8)
print("\nPosition of each point along the rod:",positions)

for i in range(len(positions)):
    print(f"Position {positions[i]:.2f} -> Temperature {temperatures[i]:.1f} C")

hottest_index=np.argmax(temperatures)
print(f"\nHottest point is at position {positions[hottest_index]:.2f} with temp {temperatures[hottest_index]:.1f} C")