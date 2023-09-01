import math
import os
from math import inf
from src.utilities import parse_image
from src.utilities import rgb2gray


def get_height_map():
    # path = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
    path = os.getcwd()
    height_map = parse_image(path + "/input/images/height_map.png")
    return height_map


# Replace the major road generation using A* search
def height_cost_function(segment, map, threshold):
    gray = rgb2gray(map)
    height_value = gray[int(segment.end_vert.position[1])][int(segment.end_vert.position[0])]
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


def check_too_high(segment):
    height_map = get_height_map()
    gray = rgb2gray(height_map)
    try:
        height_value = gray[int(segment.end_vert.position[1])][int(segment.end_vert.position[0])]
    except IndexError:
        return True

    if height_value > 90:
        return True
    else:
        return False