from queue import PriorityQueue
import numpy as np
import networkx as nx

from src.config_manager import ConfigManager
from src.road_network.growth_rules.cost_function import check_curvature, cost_function, get_road_type
from src.road_network.vertex import Vertex
from src.road_network.segment import Segment
from src.utilities import get_distance, RoadTypes


WEIGHT_FACTOR = 30
NEIGHBOR_RANGE = 7        # Tweak this. Higher = more time, roads can take more angles
BOUNDED_RELAXATION = 2   # Tweak this. Higher = Greedy search Faster, lower >= 1 optimal path
MIN_TUNNEL_LEN = 5
MIN_BRIDGE_LEN = 6



def heuristic(point_n, point_goal):
    return BOUNDED_RELAXATION*get_distance(point_n, point_goal)


def a_star_search(start, goal):
    """
    A* search algorithm to find the shortest path from start point to goal point.

    :param start: Start point
    :param goal: Goal point
    :return: Shortest path from start to goal
    """
    config = ConfigManager()

    # Initialize priority queue and add the start node
    frontier = PriorityQueue()
    frontier.put((0, start))
    closed_set = set()
    came_from = {start: None}
    g_cost = {start: 0}
    road_types = {start: ""}

    count = 0
    while not frontier.empty():
        current_priority, current = frontier.get()
        count += 1

        # Goal found
        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append((current, road_types[current]))
                current = came_from[current]
            path.reverse()
            print("A* path found in ", count, " iterations, cost: ", current_priority)
            return path

        closed_set.add(current)
        neighbors = get_neighbors(current, NEIGHBOR_RANGE, RoadTypes.HIGHWAY)

        for neighbor in neighbors:
            # Check if the neighbor is in the grid and the road is not too curvy
            if neighbor not in closed_set and config.is_in_the_map(neighbor) and not check_curvature(came_from[current], current, neighbor, 90): ## TODO NICK Move to Cost Fn
                cost, road_type = cost_function(current, neighbor, config)
                new_g_cost = g_cost[current] + cost
                priority = new_g_cost + heuristic(neighbor, goal)

                if neighbor not in g_cost or new_g_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_g_cost
                    frontier.put((priority, neighbor))
                    came_from[neighbor] = current
                    road_types[neighbor] = road_type

    print("No A* path found")
    return None  # Path not found


def get_neighbors(current, range_n, type):
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

            if type == RoadTypes.HIGHWAY:
                if dx ** 2 + dy ** 2 <= range_n ** 2:
                    neighbors.append((current[0] + dx, current[1] + dy))
            elif type == RoadTypes.TUNNEL:
                if dx ** 2 + dy ** 2 <= range_n ** 2 and dx ** 2 + dy ** 2 >= MIN_TUNNEL_LEN ** 2:
                    neighbors.append((current[0] + dx, current[1] + dy))
            elif type == RoadTypes.BRIDGE:
                if dx ** 2 + dy ** 2 <= range_n ** 2 and dx ** 2 + dy ** 2 >= MIN_BRIDGE_LEN ** 2:
                    neighbors.append((current[0] + dx, current[1] + dy))
    
    # go through the neighbors and prune same ratio neighbors, such as 2,4 and 4,8, prune 4,8
    result_set = []
    final_neighbors = []
    for neighbor in neighbors:
        # reduce the neighbor to having a gcd of 1
        x = neighbor[0] - current[0]
        y = neighbor[1] - current[1]
        gcd = np.gcd(x, y)
        reduced_neighbor = (x // gcd, y // gcd)
        if reduced_neighbor not in result_set:
            result_set.append(reduced_neighbor)
            final_neighbors.append((reduced_neighbor[0] + current[0], reduced_neighbor[1] + current[1]))


    return final_neighbors



def generate_a_star_road(path):
    """
    Generate segments between each vertex in the path.

    :param path: An array of vertices
    :return: An array of segments
    """
    config = ConfigManager()
    segments = []

    for i in range(len(path) - 1):
        x1, y1 = path[i][0]
        x2, y2 = path[i + 1][0]
        road_type = path[i + 1][1]
        segment = Segment(segment_start=Vertex(np.array([x1, y1])), segment_end=Vertex(np.array([x2, y2])), road_type=road_type)

        segments.append(segment)

    return segments


def get_all_a_star_roads(population_centres):
    """
    Generate roads/segments between each pair of population density centres using A*.

    :param population_centres: Population density centres
    :return: An array of segments
    """
    segments = []
    edges = get_edges_mst(population_centres)

    for edge in edges:
        node1Idx, node2Idx = edge
        x1, y1, *_ = population_centres[node1Idx]
        x2, y2, *_ = population_centres[node2Idx]
        print("processing search from ", (x1, y1), " to ", (x2, y2))
        path = generate_a_star_road(a_star_search((x1, y1), (x2, y2)))
        segments.append(path)

    return segments


def get_edges_mst(nodes):
    """
    Returns a graph consisting of edges and nodes based on the population density centre nodes

    :param nodes: List of nodes in the form [(x1, y1, w1), (x2, y2, w2), ...]
    :return: A list of edges in the form [(n0, n2), (n1, n3)] where nx is the index of the node
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
            distance = get_distance(nodes[i], nodes[j])
            weighted_dist = distance - WEIGHT_FACTOR * (weight_i + weight_j)  # Less cost for more important nodes so we make sure they are connected
            G.add_edge(i, j, weight=weighted_dist)

    # Find the Minimum Spanning Tree
    mst = nx.minimum_spanning_tree(G)

    return mst.edges()
