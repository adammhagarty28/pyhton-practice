import numpy as np
import matplotlib.pyplot as plt

rows=[]

for i in range(20):
    row=[]
    for j in range(20):
        value=i+j 
        row.append(value)
    rows.append(row)

grid=np.array(rows)
print(grid.shape)

fig,ax=plt.subplots()
im=ax.imshow(grid,cmap="hot", origin="lower")
fig.colorbar(im,ax=ax,label="i+j")
ax.set_title("Grid is built row by row")
plt.show()