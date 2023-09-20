import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from src.to_json import city_to_json
from src.config_manager import ConfigManager
from src.road_network.segment import Segment
import src.road_network.road_network_generator as rng
import src.city_blocks.polygons as polygons
import src.city_blocks.land_usage as land_usage
from src.stats import compute_orientation_histogram, show_orientation_histogram
from src.stats import compute_orientation_entropy, compute_orientation_order
from src.stats import compute_average_node_degree, compute_intersection_count, compute_total_road_length
from src.stats import compute_proportion_3way_intersections, compute_proportion_4way_intersections, compute_proportion_dead_ends
from src.visualise import Visualiser
from matplotlib import animation
import threading
from src.road_network.a_star import get_all_a_star_roads
from src.road_network.growth_rules.cost_function import *
from src.utilities import *


# INPUT:    String, (Bool, Bool)
# OUTPUT:   Generated city (visualisation)
# Main function used to generate an intermediate representation of a city.
# If show_city is true, the representation is visualised using matplotlib.
# If show_time is true, the process time required to generate the intermediate representation is shown.
# If show_stats is true, the statistics used to evaluate the representation are shown
def run_computations(config, road_network, vertex_dict, visualiser):
    start = time.perf_counter()
    rng.generate_major_roads(config, road_network, vertex_dict, visualiser)
    end = time.perf_counter()
    print(f"major roads completed in {end - start:0.4f} seconds")

    start = time.perf_counter()
    rng.generate_minor_roads(config, road_network, vertex_dict, visualiser)
    end = time.perf_counter()
    print(f"minor roads completed in {end - start:0.4f} seconds")


def generate(config_path, show_city=False, show_time=False, show_stats=False, number_of_centres=35):
    if show_time:
        t = time.process_time()

    # Step 0: Load config.
    start = time.perf_counter()
    config = ConfigManager(config_path)
    end = time.perf_counter()
    print(f"config loader completed in {end - start:0.4f} seconds")

    # Step 1: Grow road network.
    start = time.perf_counter()
    population_centres = get_first_n_population_centres(config.pop_density_centres, number_of_centres)
    end = time.perf_counter()
    print(f"population centres completed in {end - start:0.4f} seconds")

    # debug !!!!!!!!!!!!!!!!!!
    # (428, 537)  to  (398, 455)
    # population_centres = [(336,415,1),(302,235,1)]
    # population_centres = [(428,537,1),(398,455,1)]

    start = time.perf_counter()
    segments = get_all_a_star_roads(population_centres)
    end = time.perf_counter()
    print(f"A* completed in {end - start:0.4f} seconds")

    start = time.perf_counter()
    for path in segments:
        config.axiom.extend(path)
    end = time.perf_counter()
    print(f"config extending completed in {end - start:0.4f} seconds")

    start = time.perf_counter()
    road_network, vertex_dict = rng.initialise(config)
    end = time.perf_counter()
    print(f"Road network generator initialise completed in {end - start:0.4f} seconds")

    # Step 2: Visualise road network.
    visualiser = Visualiser(config.height_map_rgb, road_network)
    # threading.Thread(target=run_computations,
    #                  args=(config, road_network, vertex_dict, visualiser),
    #                  daemon=True).start()


    
    while True:
        # detect if window is closed? 
        # if so, break
        if not plt.fignum_exists(1):
            break
        visualiser.visualise()


    print(123123123)

   

if __name__ == "__main__":
    random.seed(42)
    generate(os.getcwd() + "/input/configs/auckland.json", show_city=True, show_time=False, show_stats=False)
