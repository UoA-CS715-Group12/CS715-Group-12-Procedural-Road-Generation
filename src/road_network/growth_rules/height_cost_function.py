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
def height_cost_function(point1, point2, height_map):
    # Get absolute distance between pixel1 and pixel2 as a multiplier to the cost
    distance = math.sqrt((point1[1] - point2[1]) ** 2 + (point1[0] - point2[0]) ** 2)
    change_in_height = abs(height_map[point1[0], point1[1]] - height_map[point2[0], point2[1]])
    cost = float(change_in_height / distance)
    if cost > 7:
        print(cost)
        return True
    else:
        return False


def check_too_high(segment, height_threshold, height_map):
    try:
        # Get the interpolated points along the segment
        points = linear_interpolate(segment, 30)
        iteration = 0
        for x1, y1 in points:
            # Round x, y to nearest integer to look them up in the height map
            x1, y1 = int(round(x1)), int(round(y1))

            height_value1 = height_map[y1][x1]
            if height_value1 > height_threshold:
                print("Height > Threshold")
                return True

            try:
                x2, y2 = points[iteration+1]
                x2, y2 = int(round(x2)), int(round(y2))
                height_value2 = height_map[y2][x2]
                if height_value2 > height_threshold:
                    print("Height > Threshold")
                    return True
                point1 = (x1, y1)
                point2 = (x2, y2)
                if height_cost_function(point1, point2, height_map):
                    print("Gradient > Threshold")
                    return True
            except IndexError:
                return False

            iteration += 1

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
