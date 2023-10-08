import math
from queue import PriorityQueue

import networkx as nx
import numpy as np
from relativeNeighborhoodGraph import returnRNG
from scipy.spatial import distance_matrix

from src.config_manager import ConfigManager
from src.road_network.growth_rules.cost_function import check_curvature, check_water, linear_interpolate_points
from src.road_network.segment import Segment
from src.road_network.vertex import Vertex
from src.utilities import get_distance, RoadTypes, get_change_in_height, get_height

# Minimum Spanning Tree related params
WEIGHT_FACTOR = 30

# Heuristic related params
BOUNDED_RELAXATION = 2  # Tweak this. Higher = Greedy search Faster, lower >= 1 optimal path

# Neighbours related params
NEIGHBOR_RANGE = 15 #15  # Tweak this. Higher = more time, roads can take more angles, has to be bigger than MIN_TUNNEL_LEN and MIN_BRIDGE_LEN
MIN_TUNNEL_LEN = 5  #5 Tweak this
MIN_BRIDGE_LEN = 6  #6 Tweak this
MIN_HIGHWAY_LEN = 0

# Cost function related params (Road cost $?M/m)
HIGHWAY_COST = 0.0264  # Tweak this parameter
TUNNEL_COST = 0.625  # Tweak this parameter
BRIDGE_COST = 3.33  # Tweak this parameter
GRADIENT_COST_FACTOR = 10  # Tweak this parameter
GRADIENT_CUTOFF = 2  # Tweak this parameter


def cost_function(point1, point2, config, road_type: RoadTypes):
    """
    Cost function to determine the cost (g value) of a segment between two points.

    :param point1: Start point of the segment
    :param point2: End point of the segment
    :param config: ConfigManager
    :param road_type: Road type of the segment
    :return: The cheapest cost of the segment and the type of road
    """
    if road_type == RoadTypes.HIGHWAY:
        return get_highway_cost(point1, point2, config)
    elif road_type == RoadTypes.TUNNEL:
        return get_tunnel_cost(point1, point2, config)
    elif road_type == RoadTypes.BRIDGE:
        return get_bridge_cost(point1, point2, config)


def get_highway_cost(point1, point2, config):
    """
    Get the cost of a highway segment between two points.

    :param point1: Start point of the segment
    :param point2: End point of the segment
    :param config: ConfigManager
    :return: The cost of the highway segment
    """
    delta_height = get_change_in_height(point1, point2, config.height_map_gray)
    distance = get_distance(point1, point2)
    gradient = delta_height / distance

    # Check if the gradient is too steep
    if gradient < -GRADIENT_CUTOFF or GRADIENT_CUTOFF < gradient:
        return math.inf

    # Check if the segment in on water
    if check_water(Segment(segment_array=[point1, point2]), config.water_map_gray):
        return math.inf

    return HIGHWAY_COST * distance * (1 + abs(gradient) * GRADIENT_COST_FACTOR)


def get_tunnel_cost(point1, point2, config):
    """
    Get the cost of a tunnel segment between two points.

    :param point1: Start point of the segment
    :param point2: End point of the segment
    :param config: ConfigManager
    :return: The cost of the tunnel segment
    """
    delta_height = get_change_in_height(point1, point2, config.height_map_gray)
    distance = get_distance(point1, point2)
    gradient = delta_height / distance

    # Check if the gradient is too steep
    if gradient < -GRADIENT_CUTOFF or GRADIENT_CUTOFF < gradient:
        return math.inf

    # Check if the segment in on water
    if check_water(Segment(segment_array=[point1, point2]), config.water_map_gray):
        return math.inf

    return TUNNEL_COST * distance


def get_bridge_cost(point1, point2, config):
    """
    Get the cost of a bridge segment between two points.

    :param point1: Start point of the segment
    :param point2: End point of the segment
    :param config: ConfigManager
    :return: The cost of the bridge segment
    """
    delta_height = get_change_in_height(point1, point2, config.height_map_gray)
    distance = get_distance(point1, point2)
    gradient = delta_height / distance

    # Check if the gradient is too steep
    if gradient < -GRADIENT_CUTOFF or GRADIENT_CUTOFF < gradient:
        return math.inf

    points = linear_interpolate_points(point1, point2)
    start_height = get_height(point1, config.height_map_gray)
    end_height = get_height(point2, config.height_map_gray)

    bridge_section_heights = np.linspace(start_height, end_height, len(points))
    section_dist = distance / len(points)
    cost = 0

    # Check no terrain is above the bridge
    for i in range(len(points)):
        curr_ground_altitude = get_height(points[i], config.height_map_gray)
        if curr_ground_altitude > bridge_section_heights[i]:
            return math.inf

        cost += BRIDGE_COST * section_dist

    return cost


def heuristic(point_n, point_goal):
    return BOUNDED_RELAXATION * get_distance(point_n, point_goal)


def a_star_search(start, goal):
    """
    A* search algorithm to find the shortest path from start point to goal point.

    :param start: Start point
    :param goal: Goal point
    :return: Shortest path from start to goal
    """

    def process_neighbors(neighbors, road_type: RoadTypes):
        """
        Process the neighbors of the current node.

        :param neighbors:
        :param road_type: Road type of all the neighbors
        """
        for neighbor in neighbors:
            # Check if the neighbor is in the grid
            if neighbor not in closed_set and config.is_in_the_map(neighbor):
                new_g_cost = g_cost[current] + cost_function(current, neighbor, config, road_type)
                priority = new_g_cost + heuristic(neighbor, goal)

                # If the neighbor cost is cheaper, update the cost and add it to the frontier
                if neighbor not in g_cost or new_g_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_g_cost
                    frontier.put((priority, neighbor))
                    came_from[neighbor] = current
                    road_types[neighbor] = road_type

    config = ConfigManager()

    # Initialize priority queue and add the start node
    frontier = PriorityQueue()
    frontier.put((0, start))
    closed_set = set()
    came_from = {start: None}
    g_cost = {start: 0}
    road_types = {start: RoadTypes.NULL}
    neighbors_mask_highway, neighbors_mask_tunnel, neighbors_mask_bridge = get_neighbors_masks(NEIGHBOR_RANGE)

    count = 0
    while not frontier.empty():
        current_priority, current = frontier.get()
        count += 1
        if count > 10000:
            print("A* path not found in 10000 iterations")
            return None
        # Current node has been visited before with a cheaper cost
        if current in closed_set:
            continue

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

        # Get neighbors for all road types
        neighbors_highway, neighbors_tunnel, neighbors_bridge = (
            get_neighbors(came_from[current], current,
                          neighbors_mask_highway, neighbors_mask_tunnel, neighbors_mask_bridge
                          )
        )

        process_neighbors(neighbors_highway, RoadTypes.HIGHWAY)
        process_neighbors(neighbors_tunnel, RoadTypes.TUNNEL)
        process_neighbors(neighbors_bridge, RoadTypes.BRIDGE)

    print("No A* path found")
    return None  # Path not found


def get_neighbors(previous, current, *neighbors_masks):
    """
    Get the neighbors of the current cell with the neighbors masks
    The neighbors are within 90 degrees of the current road

    You can pass as many neighbors masks as you want, and it will return the neighbors of all the masks.
    Also, the number of neighbors arrays returned is the same as the number of neighbors masks passed in.

    :param previous: Previous cell
    :param current: Current cell
    :param neighbors_masks: Neighbors masks, eg: neighbors_mask_highway, neighbors_mask_tunnel, neighbors_mask_bridge
    :return: Array(s) of neighbor pixels/vertices
    """
    all_neighbors = []

    for neighbors_mask in neighbors_masks:
        neighbors = []

        for mask in neighbors_mask:
            neighbor = (current[0] + mask[0], current[1] + mask[1])

            # Skip if the neighbor is facing behind the road
            if previous is not None and check_curvature(previous, current, neighbor, 25):
                continue

            neighbors.append(neighbor)

        all_neighbors.append(neighbors)

    return tuple(all_neighbors)


def get_neighbors_masks(n_range):
    """
    Get the neighbor pixels/vertices mask in a circle grid.

    Return masks' orders:
        1. Highway neighbors are the lowest equivalent vector.
        2. Tunnel neighbors are at least MIN_TUNNEL_LEN away.
        3. Bridge neighbors are at least MIN_BRIDGE_LEN away.

    :param n_range: Radius of the neighbor pixels/vertices from current cell
    :return: An array of neighbor pixels/vertices within the circle grid.
    """
    neighbors_highway = set()
    neighbors_tunnel = set()
    neighbors_bridge = set()

    for dx in range(-n_range, n_range + 1):
        for dy in range(-n_range, n_range + 1):
            # Skip the current cell itself
            if dx == 0 and dy == 0:
                continue

            # Neighbor is within the circle range
            if dx ** 2 + dy ** 2 <= n_range ** 2:
                # Calculate the vector and get the lowest equivalent vector
                # Eg: (2, 4) and (4, 8), ignore (4, 8)
                gcd = math.gcd(dx, dy)
                vector_lowest = (dx // gcd, dy // gcd)

                # Highway neighbor add the lowest vector
                if dx ** 2 + dy ** 2 >= MIN_HIGHWAY_LEN ** 2:
                    neighbors_highway.add(vector_lowest)

                # Tunnel neighbor needs to be at least MIN_TUNNEL_LEN away
                if dx ** 2 + dy ** 2 >= MIN_TUNNEL_LEN ** 2:
                    neighbors_tunnel.add((dx, dy))

                # Same as Bridge neighbor
                if dx ** 2 + dy ** 2 >= MIN_BRIDGE_LEN ** 2:
                    neighbors_bridge.add((dx, dy))

    return neighbors_highway, neighbors_tunnel, neighbors_bridge


def generate_a_star_road(path):
    """
    Generate segments between each vertex in the path.

    :param path: An array of vertices
    :return: An array of segments
    """
    segments = []
    if path is None:
        return segments
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
    # edges = get_edges_mst(population_centres)
    edges = get_edges_rng(population_centres)

    for edge in edges:
        node1Idx, node2Idx = edge
        x1, y1, *_ = population_centres[node1Idx]
        x2, y2, *_ = population_centres[node2Idx]

        print("==============================================================")
        print("processing search from ", (x1, y1), " to ", (x2, y2))

        path = generate_a_star_road(a_star_search((x1, y1), (x2, y2)))
        segments.append(path)

    print("==============================================================")
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


def get_edges_rng(nodes):
    """
    Returns a graph consisting of edges based on the population density centre nodes
    using Relative Neighborhood Graph (RNG).

    :param nodes: List of nodes in the form [(x1, y1, w1), (x2, y2, w2), ...]
    :return: A list of edges in the form [(n0, n2), (n1, n3)] where nx is the index of the node
    """
    # Extract the positions of nodes
    positions = [(x, y) for x, y, w in nodes]

    # Compute the distance matrix
    dist_matrix = distance_matrix(positions, positions)

    # Compute the relative neighbor graph
    RNG = returnRNG.returnRNG(dist_matrix)

    def adjacency_matrix_to_edges(adjacency_matrix):
        """
        Convert the adjacency matrix to a list of edges.

        :param adjacency_matrix: Adjacency matrix
        :return: A list of edges in the form [(n0, n2), (n1, n3)] where nx is the index of the node
        """
        edges = []
        for i in range(len(adjacency_matrix)):
            for j in range(i + 1, len(adjacency_matrix)):
                if adjacency_matrix[i][j] != 0:
                    edges.append((i, j))
        return edges

    RNG = adjacency_matrix_to_edges(RNG.values)

    return RNG
