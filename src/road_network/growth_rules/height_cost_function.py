import os
from src.utilities import parse_image
from src.utilities import rgb2gray


def get_height_map():
    path = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
    height_map = parse_image(path + "/input/images/height_map.png")
    return height_map


def height_cost_function(map):
    gray = rgb2gray(map)
    height, width = gray.shape
    for x in range(width):
        for y in range(height):
            # Replace with function for pixel value
            print(gray[y][x])



if __name__ == "__main__":
    map = get_height_map()
    height_cost_function(map)