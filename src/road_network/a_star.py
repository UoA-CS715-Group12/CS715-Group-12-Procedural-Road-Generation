from queue import PriorityQueue
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from src.config_manager import ConfigManager
from src.road_network.vertex import Vertex
from src.road_network.segment import Segment
from src.utilities import get_distance

WEIGHT_FACTOR = 30

def heuristic(point_n, point_goal):
    return get_distance(point_n, point_goal)


def a_star_search(start, goal):
    config = ConfigManager()

    # Initialize priority queue and add the start node
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    g_cost = {start: 0}
    f_value = {start: 0}

    while not frontier.empty():
        current_priority, current = frontier.get()

        # Goal found
        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        neighbors = get_neighbors(current, 15)

        for neighbor in neighbors:
            x, y = neighbor
            # Check if the neighbor is in the grid and is not an obstacle
            if 0 <= x < np.shape(config.water_map_gray)[1] and 0 <= y < np.shape(config.water_map_gray)[0] and config.water_map_gray[y][x] < 200:
                new_g_cost = g_cost[current] + 1
                priority = new_g_cost + heuristic(neighbor, goal)

                if neighbor not in f_value or priority < f_value[neighbor]:
                    g_cost[neighbor] = new_g_cost
                    f_value[neighbor] = priority
                    frontier.put((priority, neighbor))
                    came_from[neighbor] = current

    print("No A* path found")
    return None  # Path not found


def get_neighbors(current, range_n):
    """
    Get the neighbor pixels/vertices of a cell in a grid.

    :param current: Current cell
    :param range_n: Range of the neighbor pixels/vertices from current cell
    :return: An array of neighbor pixels/vertices
    """
    neighbors = []

    for dx in range(-range_n, range_n + 1):
        for dy in range(-range_n, range_n + 1):
            if dx == 0 and dy == 0:  # Skip the current cell itself
                continue
            neighbors.append((current[0] + dx, current[1] + dy))

    return neighbors


def generate_a_star_road(path):
    """
    Generate segments between each vertex in the path.

    :param path: An array of vertices
    :return: An array of segments
    """
    segments = []

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        segment = Segment(segment_start=Vertex(np.array([x1, y1])), segment_end=Vertex(np.array([x2, y2])))

        segments.append(segment)

    return segments


def get_all_a_star_roads(population_centres):
    segments = []
    edges = get_edges_mst(population_centres)
    
    for edge in edges:
        node1Idx, node2Idx = edge
        x1, y1, *_ = population_centres[node1Idx]
        x2, y2, *_ = population_centres[node2Idx]
        path = generate_a_star_road([(x1, y1), (x2, y2)]) # TODO: Add A star search between each pair of centres
        segments.append(path)
        
    return segments

def get_edges_mst(nodes):
    """ Returns a graph consisting of edges and nodes based on the population density centre nodes

    Args:
        nodes (_type_): List of nodes in the form [(x1, y1, w1), (x2, y2, w2), ...]
    return: A list of edges in the form [(n0, n2), (n1, n3)] where nx is the index of the node
    """
    # Create a graph
    G = nx.Graph()

    # Add nodes to the graph
    for i, node in enumerate(nodes):
        G.add_node(i, pos=node)

    # Calculate the distances between all pairs of nodes
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            weight_i = nodes[i][2]
            weight_j = nodes[j][2]
            distance = ((nodes[i][0] - nodes[j][0])**2 + (nodes[i][1] - nodes[j][1])**2)**0.5
            weighted_dist = distance - WEIGHT_FACTOR * (weight_i + weight_j) # Less cost for more important nodes so we make sure they are connected
            G.add_edge(i, j, weight=weighted_dist)

    # Find the Minimum Spanning Tree
    mst = nx.minimum_spanning_tree(G)
    
    return mst.edges()
