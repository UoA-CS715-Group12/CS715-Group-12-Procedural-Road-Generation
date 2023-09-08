import math
from queue import PriorityQueue
import numpy as np
from src.road_network.vertex import Vertex
from src.road_network.segment import Segment


def heuristic(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def a_star_search(water_map, start, goal):
    # Initialize priority queue and add the start node
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

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
            if x >= 0 and x < np.shape(water_map)[1] and y >= 0 and y < np.shape(water_map)[0] and water_map[y][x] < 200:
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(goal, neighbor)
                    frontier.put((priority, neighbor))
                    came_from[neighbor] = current

    print("No A* path found")
    return None  # Path not found

def get_neighbors(current, range_n):
    neighbors = []
    for dx in range(-range_n, range_n + 1):
        for dy in range(-range_n, range_n + 1):
            if dx == 0 and dy == 0:  # Skip the current cell itself
                continue
            neighbors.append((current[0] + dx, current[1] + dy))
    return neighbors


def generate_a_star_roads(path):
    segments = []
    iteration = 0
    for x1,y1 in path:
        try:
            x2, y2 = path[iteration+1]
            segment = Segment(segment_start=Vertex(np.array([x1, y1])), segment_end=Vertex(np.array([x2, y2])))

            segments.append(segment)

        except IndexError:
            print("End of A* path")

        iteration += 1
    return segments
