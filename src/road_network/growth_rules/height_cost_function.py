import os
from src.utilities import parse_image
from src.utilities import rgb2gray


def get_height_map():
    path = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
    height_map = parse_image(path + "/input/images/height_map.png")
    return height_map

# Replace the major road generation using A* search
def height_cost_function(map):
    gray = rgb2gray(map)
    height, width = gray.shape
    for x in range(width):
        for y in range(height):
            # Replace with function for pixel value
            pixel = gray[y][x]
            if x == width - 1:
                continue
            neighbour = gray[y][x+1]
            # Given 2 neighbouring pixels
            # Height Cost = abs(pixel1 - pixel2)
            # Get absolute distance between pixel1 and pixel2 as a multiplier to the cost
            # If abs(pixel1 - pixel2) > slope_threshold, then cost = inf
            # Assume water depth from water maps?
            difference = pixel

    # Generate a path between pixel1 and pixel100
    # Perform A* search between pixel1 and pixel100
    # Cost = cost function


if __name__ == "__main__":
    map = get_height_map()
    height_cost_function(map)