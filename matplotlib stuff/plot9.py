import numpy as np
import matplotlib.pyplot as plt

categories = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
values     = [23, 41, 38, 55, 62]

fig, ax = plt.subplots()
bars = ax.bar(categories, values, color='steelblue')

#label the value above each bar

for bar, val in zip(bars, values):
    ax.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+1,
        str(val),
        ha="center",fontsize=9
    )
ax.set_ylabel('Units sold')
ax.set_title('Monthly sales')
ax.spines['top'].set_visible(False)    # cleaner look
ax.spines['right'].set_visible(False)
plt.show()