import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from src.utilities import RoadTypes


class Visualiser:
    def __init__(self, map_array, road_network, land_usages=None):
        self.fig, self.ax, self.highways, self.bridges, self.tunnels, self.minor_lines = init_plot()
        self.map_array = map_array
        self.road_network = road_network
        self.land_usages = land_usages
        self.iteration_counter = 0

    def visualise(self):
        # while True:
        #     time.sleep(1)
        #     visualise(self.map_array, self.road_network, self.major_lines, self.minor_lines, self.fig, self.ax, self.land_usages)
        #     self.iteration_counter += 1

        self.iteration_counter += 1
        at_beginning = self.iteration_counter < 10
        visualise(self.map_array, self.road_network, self.highways, self.bridges, self.tunnels, self.minor_lines, self.fig, self.ax, at_beginning, self.land_usages)


def init_plot():
    fig, ax = plt.subplots()
    ax.axis('equal')
    highways = LineCollection([], linewidths=[1.0], colors=[[0.2, 0.6, 0.6, 1]])
    bridges = LineCollection([], linewidths=[1.0], colors=[[1, 1, 0, 1]])
    tunnels = LineCollection([], linewidths=[1.0], colors=[[0.5, 1, 0, 1]])
    minor_lines = LineCollection([], linewidths=[0.6], colors=[[0.6, 0.2, 0, 0.8]])

    plt.ion()  # Turn on interactive mode

    plt.show()

    return fig, ax, highways, bridges, tunnels, minor_lines


def visualise(map_array, road_network, highways, bridges, tunnels, minor_lines, fig, ax, at_beginning=True, land_usages=None):
    if not at_beginning:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    bridges_segment_coords = []
    tunnels_segment_coords = []
    highways_segment_coords = []
    minor_segment_coords = []
    vertex_x_coords = []
    vertex_y_coords = []

    for segment in road_network:
        vertex_x_coords.append(segment.start_vert.position[0])
        vertex_y_coords.append(segment.start_vert.position[1])
        vertex_x_coords.append(segment.end_vert.position[0])
        vertex_y_coords.append(segment.end_vert.position[1])

        coors = np.array([segment.start_vert.position, segment.end_vert.position])

        if segment.road_type == RoadTypes.BRIDGE:
            bridges_segment_coords.append(coors)
        elif segment.road_type == RoadTypes.TUNNEL:
            tunnels_segment_coords.append(coors)
        elif not segment.is_minor_road:
            highways_segment_coords.append(coors)
        else:
            minor_segment_coords.append(coors)

    bridges.set_segments(bridges_segment_coords)
    tunnels.set_segments(tunnels_segment_coords)
    highways.set_segments(highways_segment_coords)
    minor_lines.set_segments(minor_segment_coords)

    ax.clear()  # Clear previous fill polygons if you want to
    ax.imshow(map_array)
    ax.add_collection(bridges)
    ax.add_collection(tunnels)
    ax.add_collection(highways)
    ax.add_collection(minor_lines)
    # ax.scatter(vertex_x_coords, vertex_y_coords, c='purple', s=2)

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

    if at_beginning:
        ax.autoscale_view()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.canvas.flush_events()  # Update the plot
