import pandas as pd
import numpy as np
from typing import List, Optional
from bokeh.models import ColumnDataSource, Button, Div, HoverTool, RangeTool
from bokeh.plotting import figure, output_notebook, output_file, show
from bokeh.layouts import column, row
from bokeh.io import curdoc
from bokeh.palettes import Category10, Category20
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import sys
from pathlib import Path

nb_path = Path().resolve()
root = nb_path.parents[4]
sys.path.append(str(root))

from bsoid_umap_prelim.utils.save import *

###############################
# MAKE FRAMES NEXT TO SCATTER #
###############################

###############################
# LABEL THE AXES OF THE PLOTS #
###############################

###############################################
# UMAP TICK WIDTHS/SCALES ARE CLEARLY DEFINED #
###############################################

# Singleton to manage dashboard initialization
class Dashboard:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Dashboard, cls).__new__(cls)
            cls._doc_initialized = False
        return cls._instance

    def initialize_output(self, notebook: bool = False, file_name: str = "data_visualization_dashboard.html") -> None:
        """Initializes the output for Bokeh visualizations."""
        if not self._doc_initialized:
            if notebook:
                output_notebook()
            else:
                output_file(file_name)
            self._doc_initialized = True

dashboard = Dashboard()  # Singleton instance

# Function to prepare wedge data for donut plots
def prepare_wedge_data(data: pd.DataFrame, column_name: str, colors: List[str]) -> pd.DataFrame:
    counts = data[column_name].value_counts()
    categories = sorted(counts.index)
    sizes = counts.loc[categories].values

    angles = np.linspace(0, 2 * np.pi, len(categories) + 1)
    start_angles = angles[:-1]
    end_angles = angles[1:]

    return pd.DataFrame({
        'start_angle': start_angles,
        'end_angle': end_angles,
        'color': colors,
        column_name: categories,
        'value': sizes
    })

# Function to extract a frame from video (assumed functionality)
def extract_frame():
    # Placeholder: replace with actual frame extraction logic
    return np.zeros((480, 640, 3), dtype=np.uint8)  # Dummy BGR image

def display_frame() -> None:
    video_frame = extract_frame()  # Assuming extract_frame returns a BGR frame
    frame_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    img_rgba = pil_img.convert("RGBA")
    img_width, img_height = img_rgba.size
    img_data = np.array(img_rgba)
    
    flat_img = img_data.view(np.uint32).reshape((img_height, img_width))
    
    p = figure(width=img_width, height=img_height, x_range=(0, img_width), y_range=(0, img_height))
    p.image_rgba(image=[flat_img], x=0, y=0, dw=img_width, dh=img_height)
    
    curdoc().add_root(column(p))

def scatter_plot(df: pd.DataFrame) -> None:
    """Creates a scatter plot from the provided DataFrame."""
    if all(col in df.columns for col in ['frame', 'coordinates', 'cluster']):
        df[['x', 'y']] = pd.DataFrame(df['coordinates'].tolist(), index=df.index)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df['x'], df['y'], c=df['cluster'], cmap='viridis', alpha=0.6)
        plt.title("Scatter Plot by Cluster")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        plt.show()
    else:
        print("DataFrame must contain 'frame', 'coordinates', and 'cluster' columns.")

def scatter_with_minimap(df: pd.DataFrame) -> None:
    """Creates a scatter plot with a minimap."""
    if all(col in df.columns for col in ['frame', 'coordinates', 'cluster']):
        df[['x', 'y']] = pd.DataFrame(df['coordinates'].tolist(), index=df.index)
        
        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(df['x'], df['y'], c=df['cluster'], cmap='viridis', alpha=0.6)
        ax.set_title("Scatter Plot with Minimap")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        plt.colorbar(scatter, label='Cluster')

        ax_inset = fig.add_axes([0.65, 0.65, 0.2, 0.2])  # Position of the minimap
        ax_inset.scatter(df['x'], df['y'], c=df['cluster'], cmap='viridis', alpha=0.6)
        ax_inset.set_xlim(df['x'].mean() - 0.1, df['x'].mean() + 0.1)  # Zoom in x-axis
        ax_inset.set_ylim(df['y'].mean() - 0.1, df['y'].mean() + 0.1)  # Zoom in y-axis
        
        plt.show()
    else:
        print("DataFrame must contain 'frame', 'coordinates', and 'cluster' columns.")

def arrange(graphs: List) -> None: pass
def display() -> None: pass

def html_dashboard() -> None:
    """Creates a link to the dashboard."""
    link = Div(text='<h1>Scatter Plot Dashboard</h1>'
                    '<p><a href="http://localhost:5006" target="_blank">Open Dashboard</a></p>',
                width=400)
    
    curdoc().add_root(column(link))

def latex_dashboard(equation: str) -> None:
    """Displays a LaTeX equation in the dashboard."""
    latex_div = Div(text=f'<h1>LaTeX Equation</h1>'
                         f'<p>{equation}</p>', 
                    width=400, height=100)
    
    curdoc().add_root(latex_div)

def extra() -> None: pass

# Function to create ColumnDataSource
def create_column_data_source(data: pd.DataFrame) -> ColumnDataSource:
    return ColumnDataSource(data)

# Function to map colors to a DataFrame
def map_colors(data: pd.DataFrame, column_name: str, palette: List[str], new_column_name: str = 'color') -> pd.DataFrame:
    data[new_column_name] = data[column_name].map(lambda cluster: palette[cluster-1])
    return data

# Function to create a donut plot
def create_donut_plot(title: str, wedge_data: pd.DataFrame, column_name: str, radius: float = 0.8) -> figure:
    plot = figure(height=350, width=350, title=title, tools='hover',
                  tooltips=f"@{column_name}: @value", x_range=(-1, 1), y_range=(-1, 1))
    plot.wedge(x=0, y=0, radius=radius, start_angle='start_angle', end_angle='end_angle',
               line_color="white", fill_color='color', legend_field=column_name, source=wedge_data)
    plot.circle(x=0, y=0, radius=0.4, fill_color='white')
    plot.axis.visible = False
    plot.grid.grid_line_color = None
    return plot

# Function to create a scatter plot
def create_scatter_plot(data_source: ColumnDataSource, tooltips: List[str], width: int = 800, height: int = 400) -> figure:
    scatter = figure(
        title="Detailed Scatter Plot",
        tools="pan,wheel_zoom,box_zoom,reset,hover",
        width=width,
        height=height,
        tooltips=tooltips,
        background_fill_color="#efefef"
    )
    scatter.scatter(x='x', y='y', color='color', source=data_source, size=10)
    scatter.add_tools(HoverTool(tooltips=tooltips))
    return scatter

# Function to create a minimap
def create_minimap(data_source: ColumnDataSource, detailed_plot: figure, width: int = 800) -> figure:
    minimap = figure(
        width=width,
        height=150,
        tools="",
        toolbar_location=None,
        background_fill_color=detailed_plot.background_fill_color,
        title="Minimap"
    )
    minimap.scatter(x='x', y='y', color='color', source=data_source, size=10)
    minimap.x_range.range_padding = 0
    minimap.ygrid.grid_line_color = None

    range_tool = RangeTool(x_range=detailed_plot.x_range, y_range=detailed_plot.y_range)
    range_tool.overlay.fill_color = "darkblue"
    range_tool.overlay.fill_alpha = 0.3
    minimap.add_tools(range_tool)

    return minimap

def create_toggle_button(minimap: figure) -> Button:
    """Creates a toggle button for the minimap."""
    toggle_button = Button(label="Toggle Minimap", button_type="success")
    toggle_button.js_on_click(CustomJS(args=dict(minimap=minimap), code="""
        minimap.visible = !minimap.visible;
    """))
    return toggle_button

def create_toggleable_object(label: str, obj, button_position: str = "below") -> column:
    """Creates a layout with a toggle button for the object."""
    toggle_button = Button(label=f"Toggle {label}", button_type="success")
    toggle_button.js_on_click(CustomJS(args=dict(obj=obj), code=f"""
        obj.visible = !obj.visible;
    """))
    return column(toggle_button, obj)

# This is an entry point to the dashboard
def main():
    dashboard.initialize_output(notebook=False)  # Set to True if using Jupyter Notebook
    dirs = directories()
    data = dirs.load_pickle('/Volumes/g.goodhill/Active/AOSI/projects/B-SOID/bsoid_umap_prelim/DEBUG_SPREE_09/old notebooks/data_test_dashboard')  # Load or create your data here

    # Prepare data for the donut plot
    wedge_data = prepare_wedge_data(data, 'cluster', Category10[10])
    donut_plot = create_donut_plot("Donut Plot Example", wedge_data, 'cluster')

    # Create a ColumnDataSource for the scatter plot
    data_source = create_column_data_source(data)

    # Create the main scatter plot
    scatter_tooltips = [("X", "@x"), ("Y", "@y"), ("Color", "@color")]
    scatter_plot_main = create_scatter_plot(data_source, scatter_tooltips)

    # Create a minimap
    minimap = create_minimap(data_source, scatter_plot_main)

    # Create the layout and display
    layout = column(donut_plot, scatter_plot_main, minimap)
    curdoc().add_root(layout)
    show(layout)  # Display the dashboard

if __name__ == "__main__":
    main()
