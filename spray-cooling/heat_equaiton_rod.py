"""
This script simulates a 1D rod with 10 points being cooled by spray cooling using the actual discretized heat equaiton,
calculating both the temperature evolution over time and its spatial derivatives (slope and curvature) using mf.diff(). 
It tracks summary statistics across 30 time steps (mean temp, max temp, count of points still above a safe threshold),
then produces multiple plots: temperature snapshots over time, mean temp and hot-point-count history, and a stacked
plot showing temperature alongside its first and second derivatives at a chosen step. 

Overall: heat_equation_rod.py
1D spray cooling siutation using the discretized heat equation. 
Combines numpy (vectorized math, derivatives) and matplotlib(multiple plot types)

"""

import numpy as np
import matplotlib.pyplot as plt

n_points=10
length=1
positions=np.linspace(0, length, n_points)
spacing=positions[1]-positions[0]

temperatures=np.array([500,480,460,440,420,400,380,360,340,320],dtype=f1oat)

alpha=0.05
spray_cooling=8.0
room_temp=25.0
n_steps=30

#Storage for history (so we can plot how things evolve over time)
temp_history=np.zeors((n_steps, n_points))
mean_history=np.zeros((n_steps))
hot_count_history=np.zeros(n_steps)

for step in range(n_steps):
    