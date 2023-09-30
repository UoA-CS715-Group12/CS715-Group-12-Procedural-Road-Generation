import os
import random
import threading
import time

import matplotlib.pyplot as plt

import src.road_network.road_network_generator as rng
from src.config_manager import ConfigManager
from src.road_network.a_star import get_all_a_star_roads
from src.timer import Timer
from src.utilities import get_first_n_population_centres
from src.visualise import Visualiser


# INPUT:    String, (Bool, Bool)
# OUTPUT:   Generated city (visualisation)
# Main function used to generate an intermediate representation of a city.
# If show_city is true, the representation is visualised using matplotlib.
# If show_time is true, the process time required to generate the intermediate representation is shown.
# If show_stats is true, the statistics used to evaluate the representation are shown
def run_computations(config, road_network, vertex_dict):
    timer = Timer("Road network generator")
    # rng.fix_overlapping_segments(config, road_network, vertex_dict)
    timer.stop()
    timer = Timer("Major roads")
    rng.generate_major_roads(config, road_network, vertex_dict)
    # rng.fix_overlapping_segments(config, road_network, vertex_dict)
    timer.stop()
    timer = Timer("Minor roads")
    rng.generate_minor_roads(config, road_network, vertex_dict)
    timer.stop()


def generate(config_path, show_city=False, show_time=False, show_stats=False, number_of_centres=35):
    if show_time:
        t = time.process_time()

    # Step 0: Load config.
    timer = Timer("Config loader")
    config = ConfigManager(config_path)
    timer.stop()

    # Step 1: Grow road network.
    timer = Timer("Population centres")
    population_centres = get_first_n_population_centres(config.pop_density_centres, number_of_centres)
    timer.stop()

    # TODO: remove later ----------- Debug
    # population_centres = [(336,415,1),(302,235,1)]
    # population_centres = [(428,537,1),(398,455,1)]

    timer = Timer("A*")
    segments = get_all_a_star_roads(population_centres)
    timer.stop()

    timer = Timer("Config extending")
    for path in segments:
        config.axiom.extend(path)
    timer.stop()

    timer = Timer("Road network generator initialise")
    road_network, vertex_dict = rng.initialise(config)
    timer.stop()

    # Step 2: Visualise road network.
    visualiser = Visualiser(config.visualisation_background, road_network)
    threading.Thread(target=run_computations,
                     args=(config, road_network, vertex_dict),
                     daemon=True).start()

    while True:
        if not plt.fignum_exists(1):
            visualiser.saveImage()
            break
        visualiser.visualise()


if __name__ == "__main__":
    random.seed(42)
    generate(os.getcwd() + "/input/configs/auckland.json", show_city=True, show_time=False, show_stats=False)
