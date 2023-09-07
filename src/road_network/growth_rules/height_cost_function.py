import math
import os
from math import inf
from src.utilities import parse_image
from src.utilities import rgb2gray


def get_height_map():
    # path = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
    path = os.getcwd()
    height_map = parse_image(path + "/input/images/height_map.png")
    gray = rgb2gray(height_map)
    return gray


# Replace the major road generation using A* search
def height_cost_function(segment, height_map, threshold):
    height_value = height_map[int(segment.end_vert.position[1])][int(segment.end_vert.position[0])]
    # Get absolute distance between pixel1 and pixel2 as a multiplier to the cost
    distance = math.sqrt((segment.end_vert.position[1] - segment.start_vert.position[1])**2 + (segment.end_vert.position[0] - segment.start_vert.position[0])**2)
    if distance <= 1:
        return 0.0
    cost = float(height_value/distance)
    # If abs(pixel1 - pixel2) > slope_threshold, then cost = inf
    if height_value > threshold:
        print("inf")
        return float(inf)
    else:
        return cost
    # Assume water depth from water maps?
    # Generate a path between pixel1 and pixel100
    # Perform A* search between pixel1 and pixel100
    # Cost = cost function


def check_too_high(segment, height_threshold, height_map):
    try:
        # Get the interpolated points along the segment
        points = linear_interpolate(segment, 10)

        for x, y in points:
            # Round x, y to nearest integer to look them up in the height map
            x, y = int(round(x)), int(round(y))
            height_value = height_map[y][x]

            if height_value > height_threshold:
                return True

    except IndexError:
        print("Check Too High Index Error")
        return True

    return False

def linear_interpolate(segment, num_points=10):
    x1, y1 = segment.start_vert.position
    x2, y2 = segment.end_vert.position

    points = []
    for i in range(num_points + 1):
        t = i / num_points
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        points.append((x, y))

    return points
