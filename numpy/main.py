"""
Here is the roadmap of what is relevant for sim work, in order, according to claude:
creating arrays-how numpy stores numbers (replaces your lists of dictionaries)

array shape-rows/columns, how to check and change it

indexing and slicing-grabbing specific elements or chunks

vectorizing math-doing math on a whole new array at once, no loops

broadcasting-numpy's rule for doing math between differently-shaped arrays

useful built-ins: .sum(), .mean(), np.zeros(), np.linspace()- the ones that are used constantly

"""
import numpy as np

temperatures=np.array([500, 480, 460, 470, 450])
grid=np.array([
[500,480,460],
[470,450,440]

])
print(grid.shape)
print(temperatures.shape)
print(grid[0])
print(temperatures[0])
print(temperatures[-1])
print(temperatures[1:3])
print("------")
print(grid[0,0])
print(grid[1,2])
print(grid[0,:])
print(grid[1,0])