import math
import random
from src.road_network.growth_rules.grid import grid
from src.road_network.growth_rules.height_cost_function import get_height_map, height_cost_function
from src.road_network.growth_rules.organic import organic
from src.utilities import get_population_density_value

def minor_road(config, segment):
    road_organic_probability = config.minor_road_organic_probability
    population_image_array = config.population_density_array
    population_density = get_population_density_value(segment, population_image_array) * config.population_scaling_factor
    height_map = get_height_map()
    threshold = 90  # placeholder threshold
    height_value = height_cost_function(segment, height_map, threshold)

    if random.uniform(0,1) <= road_organic_probability:
        return organic(config, segment, population_density, height_value)
    else:
        return grid(config, segment, population_density, height_value)
