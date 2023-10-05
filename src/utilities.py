import json
import math
from enum import Enum

import numpy as np
import skimage.morphology
from PIL import Image
from osgeo import gdal


class RoadTypes(Enum):
    BRIDGE = "bridge"
    TUNNEL = "tunnel"
    HIGHWAY = "highway"
    MINOR = "minor"
    NULL = ""


# INPUT:    String
# OUTPUT:   numpy.Array
# We open the supplied image and convert it to a numpy array.
# dtype is inferred as uint8, and the output is an array of
# lists where each parent list corresponds to a row of pixel
# values. All pixels contain [r, g, b]-values.
def parse_image(filename):
    image_array = np.asarray(Image.open(filename))
    # Remove alpha value if image only contains a single color.
    if image_array.shape[2] == 4:
        image_array = np.delete(image_array, 3, 2)
    return image_array


# INPUT:    numpy.Array, numpy.Array
# OUTPUT:   numpy.Array
# Given an array of colour values, and a specific legend color,
# we create and subsequently zip a 2-tuple of arrays containing 
# indices in the first and second dimension corresponding to the 
# pixel values of the colour matching the legend.
def find_legend_color_coordinates(image_array, legend_color):
    indices = np.where(np.all(image_array == legend_color, axis=-1))
    # Convert iterator to list to array before returning.
    return np.array(list(zip(indices[0], indices[1])))


# INPUT:    Segment
# OUTPUT:   numpy.Array (RGB value)
def find_pixel_value(segment, image_array):
    y = int(round(segment.end_vert.position[1]))
    x = int(round(segment.end_vert.position[0]))
    return image_array[y, x]


# INPUT:    numpy.Array
# OUTPUT:   Tuple
def find_coordinates_centroid(coordinates):
    # Given an array of coordinates, we find the centroid
    # of the coordinates and returns them as a tuple.
    coordinate_length = coordinates.shape[0]
    coordinate_x_sum = np.sum(coordinates[:, 0])
    coordinate_y_sum = np.sum(coordinates[:, 1])
    return (coordinate_x_sum / coordinate_length, coordinate_y_sum / coordinate_length)


# INPUT:     numpy.Array, List
# OUTPUT:    numpy.Array
def find_legend_centers(image_array, legend):
    # Find all coordinates matching the specified legend.
    legend_indices = find_legend_color_coordinates(image_array, legend)

    # if the legend is not present in the image, return an empty list, i.e. no centers
    if legend_indices.size == 0:
        return []

    # Create a Boolean matrix of size image_width x image_height and mark every
    # cell as either True or False depending on whether the legend colour is
    # present in that pixel. 
    legend_matches = np.zeros((np.shape(image_array)[0], np.shape(image_array)[1]), dtype=bool)
    legend_matches[legend_indices[:, 0], legend_indices[:, 1]] = True

    # Find clusters of the legend in the array and label them.
    labeled_matches = skimage.morphology.label(legend_matches)

    # Create list of coordinates for each cluster in the array.
    clusters = [list(zip(y, x)) for y, x in
                [(labeled_matches == cluster).nonzero() for cluster in range(1, labeled_matches.max() + 1)]]

    # Find the centroids of each cluster.
    centroids = [find_coordinates_centroid(coords) for coords in np.array(clusters)]

    return np.array(centroids)


# INPUT:    numpy.Array, Float
# OUTPUT:   numpy.Array
def rotate(vector, angle):
    # Often used special case
    if angle == 90:
        return np.array([-vector[1], vector[0]])

    angle = angle * np.pi / 180
    rotation_matrix = np.array([np.cos(angle), np.sin(angle), -np.sin(angle), np.cos(angle)]).reshape(2, 2)

    return np.dot(vector, rotation_matrix)


# INPUT:    Segment, Segment
# OUTPUT:   numpy.Array
# Computes the normalised position of the intersection on segment_one.
def compute_intersection(segment_one, segment_two):
    segment_one_start = segment_one.start_vert.position
    segment_one_sub = segment_one.end_vert.position - segment_one.start_vert.position
    segment_two_start = segment_two.start_vert.position
    segment_two_sub = segment_two.end_vert.position - segment_two.start_vert.position

    try:
        return np.linalg.solve(np.array([segment_one_sub, -segment_two_sub]).T, segment_two_start - segment_one_start)
    except np.linalg.linalg.LinAlgError:
        return np.array([np.inf, np.inf])


# INPUT:    Segment, np.Array
# OUTPUT:   Integer
# Get the population density value for a specific pixel of
# the population density image
def get_population_density_value(segment, population_image_array):
    return population_image_array[int(segment.end_vert.position[1])][int(segment.end_vert.position[0])]


# INPUT:    numpy.Array
# OUTPUT:   numpy.Array
# normalise pixel values to single value in range [0,1]
def normalise_pixel_values(image_array):
    return image_array[:, :, 0] / 255


# INPUT:    String
# OUTPUT:   numpy.Array
def read_tif_file(filename):
    gdo = gdal.Open(filename)
    band = gdo.GetRasterBand(1)
    return np.array(gdo.ReadAsArray(0, 0, band.XSize, band.YSize))


def rgb2gray(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def segment2json(segments):
    edge_list = []
    for segment in segments:
        output = [segment.start_vert.position.tolist(), segment.end_vert.position.tolist()]
        edge_list.append(output)
    return edge_list


def parse_json(filepath):
    f = open(filepath)
    data = json.load(f)
    f.close()
    return data


def get_distance(point1, point2):
    """
    Get the distance between two points.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_change_in_height(point1, point2, height_map, positive_only=False):
    """
    Get the change in height between two points.

    :return: height of point2 - height of point1
    """
    delta_height = get_height(point2, height_map) - get_height(point1, height_map)

    if positive_only:
        return abs(delta_height)
    else:
        return delta_height


def get_height(point, height_map):
    """
    Get the height of a point.
    """
    return height_map[point[1], point[0]]


def get_angle(point0, point1, point2):
    """
    Get the angle between two vectors.
    1st vector is point0 -> point1.
    2nd vector is point1 -> point2.
    Point is in the form (x, y).

    Example angle calculation: https://www.desmos.com/geometry/e1x1yam26h

    :param point0:
    :param point1: Angle at this point
    :param point2:
    :return: Degrees rounded to 2dp in the range of [0, 180]
    """
    # There is no angle
    if point0 is None:
        return 0

    # Define the three points as tuples (x, y)
    x0, y0 = point0
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the vectors representing the two lines
    vector1 = (x1 - x0, y1 - y0)
    vector2 = (x2 - x1, y2 - y1)

    # Calculate the dot product of the two vectors
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate the magnitudes (lengths) of the vectors
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Calculate the cosine of the angle between the two lines using the dot product
    cosine_theta = dot_product / (magnitude1 * magnitude2)

    # Clamp the value between -1 and 1
    cosine_theta = max(-1, min(1, cosine_theta))

    # Calculate the angle in radians
    theta_rad = math.acos(cosine_theta)

    # Convert the angle to degrees
    theta_deg = math.degrees(theta_rad)

    return round(theta_deg, 2)


def get_first_n_population_centres(pop_density_centres, number):
    results = []
    for i in range(min(number, len(pop_density_centres))):
        x = round(pop_density_centres[i]['x'])
        y = round(pop_density_centres[i]['y'])
        w = round(pop_density_centres[i]['weight'])
        results.append((x, y, w))
    return results
