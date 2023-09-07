import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


class Visualiser:
    def __init__(self, map_array, road_network, land_usages=None):
        self.fig, self.ax, self.major_lines, self.minor_lines = init_plot()
        self.map_array = map_array
        self.road_network = road_network
        self.land_usages = land_usages
        self.iteration_counter = 0

    def visualise(self):
        if self.iteration_counter % 100 == 0:
            visualise(self.map_array, self.road_network, self.major_lines, self.minor_lines, self.fig, self.ax, self.land_usages)
        self.iteration_counter += 1

def init_plot():
    fig, ax = plt.subplots()
    ax.axis('equal')
    major_lines = LineCollection([], linewidths=[1.0], colors=[[200, 100, 100, 1]])
    minor_lines = LineCollection([], linewidths=[0.6], colors=[[70,200, 0, 0.8]])

    ax.add_collection(major_lines)
    ax.add_collection(minor_lines)

    plt.ion()  # Turn on interactive mode
    plt.show()
    
    return fig, ax, major_lines, minor_lines


def visualise(map_array, road_network, major_lines, minor_lines, fig, ax, land_usages=None,):


    major_segment_coords = [np.array([segment.start_vert.position, segment.end_vert.position]) for segment in road_network if not segment.is_minor_road]
    minor_segment_coords = [np.array([segment.start_vert.position, segment.end_vert.position]) for segment in road_network if segment.is_minor_road]

    major_lines.set_segments(major_segment_coords)
    minor_lines.set_segments(minor_segment_coords)

    ax.clear()  # Clear previous fill polygons if you want to
    ax.imshow(map_array)
    ax.add_collection(major_lines)
    ax.add_collection(minor_lines)

    if land_usages is not None:
        for use in land_usages:
            x_coords = []
            y_coords = []
            poly = use["polygon"]
            color = "r"  # Default
            if use["land_usage"] == "residential":
                color = "purple"
            elif use["land_usage"] == "commercial":
                color = "y"
            elif use["land_usage"] == "industry":
                color = "b"

            for vertex in poly:
                x_coords.append(vertex['x'])
                y_coords.append(vertex['z'])
            ax.fill(x_coords, y_coords, color)

    ax.autoscale()
    fig.canvas.flush_events()  # Update the plot