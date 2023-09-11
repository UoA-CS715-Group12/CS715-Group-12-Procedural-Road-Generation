import math
import os
from math import inf
from src.utilities import parse_image, get_distance
from src.utilities import rgb2gray

height_map = None
water_map = None


def get_height_map():
    global height_map

    if height_map is not None:
        return height_map

    path = os.path.join(os.getcwd(), "input/images/greater_auckland/greater_auckland_height.png")
    height_map = parse_image(path)
    height_map = rgb2gray(height_map)
    return height_map


def get_water_map():
    global water_map

    if water_map is not None:
        return water_map

    path = os.path.join(os.getcwd(), "input/images/greater_auckland/greater_auckland_coast.png")
    water_map = parse_image(path)
    water_map = rgb2gray(water_map)
    return water_map


# Replace the major road generation using A* search
def height_cost_function(point1, point2, height_map):
    # Get absolute distance between pixel1 and pixel2 as a multiplier to the cost
    distance = get_distance(point1, point2)
    change_in_height = abs(height_map[point1[0], point1[1]] - height_map[point2[0], point2[1]])
    cost = float(change_in_height / distance)
    if cost > 7:
        # print(cost)
        return True
    else:
        return False


def check_too_high(segment, height_threshold, height_map):
    try:
        # Get the interpolated points along the segment
        points = linear_interpolate(segment, 30)
        iteration = 0
        for x1, y1 in points:
            height_value1 = height_map[y1][x1]
            if height_value1 > height_threshold:
                # print("Height > Threshold")
                return True

            try:
                x2, y2 = points[iteration + 1]
                x2, y2 = int(round(x2)), int(round(y2))
                height_value2 = height_map[y2][x2]
                if height_value2 > height_threshold:
                    # print("Height > Threshold")
                    return True
                point1 = (x1, y1)
                point2 = (x2, y2)
                if height_cost_function(point1, point2, height_map):
                    # print("Gradient > Threshold")
                    return True
            except IndexError:
                return False

            iteration += 1

    except IndexError:
        # print("Check Too High Index Error")
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
        points.append((round(x), round(y)))

    unique_points = list(set(points))

    return unique_points


def check_water(segment, water_map):
    try:
        # Get the interpolated points along the segment
        points = linear_interpolate(segment, 30)
        iteration = 0
        for x1, y1 in points:
            water_value1 = water_map[y1][x1]
            if water_value1 >= 250:
                return True

            try:
                x2, y2 = points[iteration + 1]
                x2, y2 = int(round(x2)), int(round(y2))
                water_value2 = water_map[y2][x2]
                if water_value2 >= 250:
                    return True

            except IndexError:
                return False

            iteration += 1

    except IndexError:
        return True


def check_bridge(segment, water_map):
    has_water = check_water(segment, water_map)
    if not has_water:
        return False

    x1, y1 = segment.start_vert.position
    x2, y2 = segment.end_vert.position
    if water_map[y1][x1] < 50 and water_map[y2][x2] < 50:
        return True
    else:
        return False


# Apply cost multiplier to the segment distance
def bridge_cost(segment):
    x1, y1 = segment.start_vert.position
    x2, y2 = segment.end_vert.position
    distance = get_distance((x1, y1), (x2, y2))
    return distance * 3.33
