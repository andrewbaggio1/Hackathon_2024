import sys
import os
import pandas as pd
import numpy as np

base_dir = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
sys.path.append(base_dir)
from bsoid_umap_prelim.config import *
from bsoid_umap_prelim.utils.save import *

save = directories()
cwd = save.cwd
save.shift_dir(save.great_grandparent_dir)
save.shift_dir(save.find_closest_kw_folder_down(kw = 'outputs_mega'))
save.shift_dir(save.find_closest_kw_folder_down(kw = 'outputs_end'))
save.shift_dir(sorted(save.files)[-2] + '/bokeh')
# targ_dir = save.cwd

vals = {}
for file in sorted(save.files):
    print(file)
    vals[f'{file[:-5]}'] = save.load_pickle(file)
    
import matplotlib.pyplot as plt
import mplcursors

# Extracting x and y from vals['umap_test']
data = vals['umap_test']  # Your actual data
x = data[:, 0]  # First column (x-coordinates)
y = data[:, 1]  # Second column (y-coordinates)

# Extracting color data from vals['hbd_test']
colors = vals['hbd_test']  # Should be a 1D array or Series with the same length as vals['umap_test']

# Create a figure with specified dimensions
fig_width = 20  
fig_height = 20

plt.figure(figsize=(fig_width, fig_height))

# Plotting the data with colors
scatter = plt.scatter(x, y, c=colors, s=10, cmap='viridis')  # Scatter plot with color map
plt.title("UMAP Test Data Colored by HBD Test")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar(scatter, label='HBD Test Values')  # Adding a color bar to indicate the values
plt.grid()

# Adding interactive cursor to display data on hover
cursor = mplcursors.cursor(scatter, hover=True)

# Customizing the hover information
@cursor.connect("add")
def on_add(sel):
    original_index = sel.index  # Get the original index of the point
    color_value = colors[original_index]
    
    # Change the range based on the color value
    lower_bound = int(original_index * 6 + 4796)  # Adjusting the multiplier and offset for more variation
    upper_bound = lower_bound + 5
    
    sel.annotation.set_text(
        f"Index: {original_index}\n"
        f"X: {x[original_index]:.2f}\n"
        f"Y: {y[original_index]:.2f}\n"
        f"Color: {color_value:.2f}\n"
        f"Frame Approximation: {lower_bound}-{upper_bound}"
    )

# Show the plot
plt.show()
