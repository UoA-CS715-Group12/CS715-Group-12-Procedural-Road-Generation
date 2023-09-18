import math
import os
from math import inf
from src.road_network.segment import Segment

from src.utilities import parse_image, get_distance, get_change_in_height, get_angle
from src.utilities import rgb2gray

# Road cost ($?k/m)
HIGHWAY_COST = 0.0264
BRIDGE_COST = 3.33

def check_gradient(point1, point2, height_map, gradient_threshold=7):
    """
    Check if the gradient between two points is too high.

    :param point1: 1st point
    :param point2: 2nd point
    :param height_map: Height_map_gray
    :param gradient_threshold: Threshold to check for the gradient
    :return: True if the gradient is too high, False otherwise
    """
    # Get absolute distance between pixel1 and pixel2 as a multiplier to the cost
    distance = get_distance(point1, point2)
    change_in_height = get_change_in_height(point1, point2, height_map)
    gradient = float(change_in_height) / distance

    if gradient > gradient_threshold:
        return True
    else:
        return False


def check_too_high(segment, height_threshold, height_map):
    """
    Check if the segment or the gradient is too high.

    :param segment: Segment of 2 points
    :param height_threshold: Height threshold to check
    :param height_map: Height_map_gray
    :return: True if the segment or the gradient is too high, False otherwise
    """
    # Get the interpolated points along the segment
    # points = linear_interpolate(segment, 30)
    points = linear_interpolate(segment)

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        height_value1 = height_map[y1][x1]
        height_value2 = height_map[y2][x2]
        point1 = (x1, y1)
        point2 = (x2, y2)

        if (height_value1 > height_threshold
                or height_value2 > height_threshold
                or check_gradient(point1, point2, height_map)):
            return True

    return False


def linear_interpolate(segment):
    num_points = math.floor(get_distance(segment.start_vert.position, segment.end_vert.position))
    x1, y1 = segment.start_vert.position
    x2, y2 = segment.end_vert.position

    points = []
    for i in range(num_points + 1):
        t = i / num_points
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        points.append((round(x), round(y)))

    unique_points = list(set(points))

    return unique_points


def check_water(segment, water_map):
    """
    Check if the segment intersects with water.

    :param segment: A Segment
    :param water_map: Water_map_gray
    :return: True if the segment intersects with water, False otherwise
    """
    # Get the interpolated points along the segment
    points = linear_interpolate(segment)
    for x, y in points:
        water_value1 = water_map[y][x]
        if water_value1 >= 250:
            return True

    return False


def check_bridge(segment, water_map):
    """
    Deprecate soon.

    TODO: Remove this function if it is not used.
    """
    has_water = check_water(segment, water_map)
    if not has_water:
        return False

    x1, y1 = segment.start_vert.position
    x2, y2 = segment.end_vert.position
    if water_map[y1][x1] < 50 and water_map[y2][x2] < 50:
        return True
    else:
        return False


def check_curvature(point0, point1, point2, degree_threshold=90):
    """
    Check if the angle between the 3 points is too high.
    Or in other words, check if the road have sharp turns

    :param point0:
    :param point1:
    :param point2:
    :param degree_threshold: Determine how sharp the turn is
    :return: True if the angle is too high, False otherwise
    """
    return get_angle(point0, point1, point2) > degree_threshold


def cost_function(point1, point2, water_map):
    # TODO: Implement cost function for A Star using the road economic factors, elevation changes
    # TODO: Roads should be able to form bridges over water if the cost is less than taking the long way round
    # TODO: Roads should consider the costs of generating roads through or over elevation changes

    distance = get_distance(point1, point2)
    cost = get_road_cost(point1, point2, water_map)

    return distance * cost


def get_road_cost(point1, point2, water_map):
    """
    Get the cost of the road between 2 points.

    :param point1: 1st point
    :param point2: 2nd point
    :param water_map: Water_map_gray
    :return: Cost of the road
    """
    if check_water(Segment(segment_array=[point1, point2]), water_map):
        return BRIDGE_COST
    else:
        return HIGHWAY_COST
