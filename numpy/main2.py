import numpy as np

temperatures=np.array([500, 480, 460, 470, 450])
room_temp=25
cooling_rate=20

temperatures=temperatures-cooling_rate
print(temperatures>room_temp)