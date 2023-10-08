import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from src.utilities import RoadTypes

HIGHWAY_COLOUR = [[251 / 255, 177 / 255, 99 / 255, 1]]
MINOR_ROAD_COLOUR = [[183 / 255, 185 / 255, 187 / 255, 1]]
TUNNER_COLOUR = [[227 / 255, 227 / 255, 227 / 255, 1]]
BRIDGE_COLOUR = [[219 / 255, 181 / 255, 0 / 255, 1]]

EDGE_COLOURS = {
    "HIGHWAY": "white",
    "TUNNEL": "gray",
    "BRIDGE": "dimgray"
}


class Visualiser:
    def __init__(self, map_array, road_network, land_usages=None):
        self.fig, self.ax, self.highways, self.bridges, self.tunnels, self.minor_lines, self.legend = init_plot()
        self.map_array = map_array
        self.road_network = road_network
        self.land_usages = land_usages
        self.iteration_counter = 0

    def visualise(self):
        self.iteration_counter += 1
        at_beginning = self.iteration_counter < 10
        visualise(self.map_array, self.road_network, self.highways, self.bridges,
                  self.tunnels, self.minor_lines, self.fig, self.ax, at_beginning, self.land_usages)

    def saveImage(self):
        self.fig.axes[0].get_xaxis().set_visible(False)
        self.fig.axes[0].get_yaxis().set_visible(False)
        self.legend.remove()
        # Save the graph 
        self.fig.savefig('output/visualisation.png', bbox_inches='tight', pad_inches=0, dpi=1200)


def init_plot():
    fig, ax = plt.subplots()
    ax.axis('equal')
    highways = LineCollection([], linewidths=[1.2], colors=HIGHWAY_COLOUR, zorder=10)
    bridges = LineCollection([], linewidths=[1.4], colors=BRIDGE_COLOUR, zorder=12)
    tunnels = LineCollection([], linewidths=[1.4], colors=TUNNER_COLOUR, zorder=11)
    minor_lines = LineCollection([], linewidths=[0.5], colors=MINOR_ROAD_COLOUR, zorder=2)

    plt.ion()  # Turn on interactive mode

    # Plot the legend
    highway_patch = mpatches.Patch(facecolor=HIGHWAY_COLOUR[0], edgecolor=EDGE_COLOURS["HIGHWAY"],
                                   label='Highway')
    minor_road_patch = mpatches.Patch(facecolor=MINOR_ROAD_COLOUR[0], edgecolor="gray",
                                      label='Minor Road')
    bridge_patch = mpatches.Patch(facecolor=BRIDGE_COLOUR[0], edgecolor=EDGE_COLOURS["BRIDGE"],
                                  label='Bridge')
    tunnel_patch = mpatches.Patch(facecolor=TUNNER_COLOUR[0], edgecolor=EDGE_COLOURS["TUNNEL"],
                                  label='Tunnel')
    legend = fig.legend(loc="upper left", handles=[
        highway_patch, minor_road_patch, bridge_patch, tunnel_patch])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return fig, ax, highways, bridges, tunnels, minor_lines, legend


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

        coors = np.array([segment.start_vert.position,
                          segment.end_vert.position])

        if segment.road_type == RoadTypes.BRIDGE:
            bridges_segment_coords.append(coors)
        elif segment.road_type == RoadTypes.TUNNEL:
            tunnels_segment_coords.append(coors)
        elif not segment.is_minor_road:
            highways_segment_coords.append(coors)
        else:
            minor_segment_coords.append(coors)

    bridges.set_segments(bridges_segment_coords)
    bridges.set_path_effects(
        [pe.Stroke(linewidth=4, foreground=EDGE_COLOURS["BRIDGE"]), pe.Normal()])
    tunnels.set_segments(tunnels_segment_coords)
    tunnels.set_path_effects(
        [pe.Stroke(linewidth=4, foreground=EDGE_COLOURS["TUNNEL"]), pe.Normal()])
    highways.set_segments(highways_segment_coords)
    highways.set_path_effects(
        [pe.Stroke(linewidth=1.5, foreground=EDGE_COLOURS["HIGHWAY"]), pe.Normal()])
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
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.canvas.flush_events()  # Update the plot
