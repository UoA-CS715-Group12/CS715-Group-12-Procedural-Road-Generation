import json
import os
import random
import threading

import matplotlib.pyplot as plt

import src.road_network.road_network_generator as rng
from src.config_manager import ConfigManager
from src.road_network.a_star import get_all_a_star_roads
from src.stats import (compute_average_node_degree, compute_intersection_count, compute_orientation_entropy,
                       compute_orientation_histogram, compute_orientation_order, compute_proportion_3way_intersections,
                       compute_proportion_4way_intersections, compute_proportion_dead_ends, compute_total_road_length)
from src.timer import Timer
from src.utilities import get_first_n_population_centres
from src.visualise import Visualiser


# INPUT:    String, (Bool, Bool)
# OUTPUT:   Generated city (visualisation)
# Main function used to generate an intermediate representation of a city.
# If show_city is true, the representation is visualised using matplotlib.
# If show_time is true, the process time required to generate the intermediate representation is shown.
# If show_stats is true, the statistics used to evaluate the representation are shown
def run_computations(config, road_network, vertex_dict, total_time_arr):
    timer = Timer("Road network generator")
    # rng.fix_overlapping_segments(config, road_network, vertex_dict)
    total_time_arr[0] += timer.stop()
    timer = Timer("Major roads")
    # rng.generate_major_roads(config, road_network, vertex_dict)
    # rng.fix_overlapping_segments(config, road_network, vertex_dict)
    total_time_arr[0] += timer.stop()
    timer = Timer("Minor roads")
    rng.generate_minor_roads(config, road_network, vertex_dict)
    total_time_arr[0] += timer.stop()


def generate(config_path, show_city=False, show_time=False, show_stats=True, number_of_centres=35):
    total_time_arr = [0]  # use array so thread can update total time

    # Step 0: Load config.
    timer = Timer("Config loader")
    config = ConfigManager(config_path)
    total_time_arr[0] += timer.stop()

    # Step 1: Grow road network.
    timer = Timer("Population centres")
    population_centres = get_first_n_population_centres(
        config.pop_density_centres, number_of_centres)
    total_time_arr[0] += timer.stop()

    # TODO: remove later ----------- Debug
    # population_centres = [(336,415,1),(302,235,1)]
    # population_centres = [(428,537,1),(398,455,1)]

    timer = Timer("A*")
    segments = get_all_a_star_roads(population_centres)
    total_time_arr[0] += timer.stop()

    timer = Timer("Config extending")
    for path in segments:
        config.axiom.extend(path)
    total_time_arr[0] += timer.stop()

    timer = Timer("Road network generator initialise")
    road_network, vertex_dict = rng.initialise(config)
    total_time_arr[0] += timer.stop()

    # Step 2: Visualise road network.
    visualiser = Visualiser(config.visualisation_background, road_network)
    threading.Thread(target=run_computations,
                     args=(config, road_network, vertex_dict, total_time_arr),
                     daemon=True).start()

    while True:
        if not plt.fignum_exists(1):
            visualiser.saveImage()
            break
        visualiser.visualise()

    # Step 3: Compute road stats
    if show_stats:
        compute_and_save_stats(road_network, vertex_dict, total_time_arr[0], config)


def compute_and_save_stats(road_network, vertex_dict, total_time, config=None):
    orientation_histogram = compute_orientation_histogram(road_network)
    stats = {
        'Entropy': compute_orientation_order(compute_orientation_entropy(orientation_histogram)),
        'Average Node Degree': compute_average_node_degree(vertex_dict),
        'Proportion Dead-Ends': compute_proportion_dead_ends(vertex_dict),
        'Proportion 3-way Intersections': compute_proportion_3way_intersections(vertex_dict),
        'Proportion 4-way Intersections': compute_proportion_4way_intersections(vertex_dict),
        'Intersection Count': compute_intersection_count(vertex_dict),
        'Total Road Length': compute_total_road_length(road_network, config=config),
        'Total Time Taken': total_time
    }

    print("==============================================================")
    print("Statistical measurements:")
    print('\n'.join([f'{key}: {value}' for key, value in stats.items()]))
    print("==============================================================")
    print(
        "Full road generation completed in {:.4f} seconds".format(total_time))

    # Dump to json file.
    file_path = os.path.join(os.getcwd(), "output", "stats.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as out:
        json.dump(stats, out, indent=4)

    print(f"Stats saved to {file_path}")


if __name__ == "__main__":
    random.seed(42)
    generate(os.getcwd() + "/input/configs/auckland.json",
             show_city=True, show_time=False, show_stats=True)
