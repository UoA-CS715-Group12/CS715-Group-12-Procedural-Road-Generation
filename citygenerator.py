import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from src.to_json import city_to_json
from src.config_loader import ConfigLoader
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
import src.road_network.a_star as astar
from src.road_network.growth_rules.cost_function import *
from src.utilities import *



# INPUT:    String, (Bool, Bool)
# OUTPUT:   Generated city (visualisation)
# Main function used to generate an intermediate representation of a city.
# If show_city is true, the representation is visualised using matplotlib.
# If show_time is true, the process time required to generate the intermediate representation is shown.
# If show_stats is true, the statistics used to evaluate the representation are shown
def run_computations(config, road_network, vertex_dict, visualiser):
    rng.generate_major_roads(config, road_network, vertex_dict, visualiser)
    rng.generate_minor_roads(config, road_network, vertex_dict, visualiser)

def generate(config_path, show_city=False, show_time=False, show_stats=False):

    if show_time:
        t = time.process_time()

    # Step 0: Load config.
    start = time.perf_counter()
    config = ConfigLoader(config_path)
    end = time.perf_counter()
    print(f"config completed in {end - start:0.4f} seconds")
    
    # Step 1: Grow road network.

    # TODO: Integrate Tony's Population center code here, generate A* roads from population centers
    # TODO: Merge generated Main roads
    # TODO: Refine A* cost function

    segments = astar.generate_a_star_roads(astar.a_star_search(get_water_map(), (300, 400), (30, 300)))

    config.axiom.extend(segments)

    road_network, vertex_dict = rng.initialise(config)

    visualiser = Visualiser(config.height_map_array, road_network)

    threading.Thread(target=run_computations, args=(config, road_network, vertex_dict, visualiser), daemon=True).start()

    while True:
        
        visualiser.visualise()
    
    # # Start the visualisation in a separate thread
    # rng.generate_major_roads(config, road_network, vertex_dict, visualiser)
    # rng.generate_minor_roads(config, road_network, vertex_dict, visualiser)


        

    # # Step 2: Compute polygons based on road network.
    # start = time.perf_counter()
    # polys = polygons.get_polygons(vertex_dict)
    # end = time.perf_counter()
    # print(f"compute polygons completed in {end - start:0.4f} seconds")
    # del polys[0] # We delete the first polygon as this corresponds to the outer area.
    
    # # Step 3: Determine land usages.
    # start = time.perf_counter()
    # land_usages = land_usage.get_land_usage(polys, config)
    # end = time.perf_counter()
    # print(f"get land usage completed in {end - start:0.4f} seconds")
    #
    # # Step 4: Dump to .json.
    # start = time.perf_counter()
    # city_to_json(road_network, list(vertex_dict.keys()), land_usages)
    # end = time.perf_counter()
    # print(f"dump to json completed in {end - start:0.4f} seconds")

    if show_time:
        print('Time:', time.process_time() - t)

    if show_stats:
        orientation_histogram = compute_orientation_histogram(road_network)
        entropy = compute_orientation_entropy(orientation_histogram)
        orientation_order = compute_orientation_order(entropy)
        avg_node_degree = compute_average_node_degree(vertex_dict)
        proportion_dead_ends = compute_proportion_dead_ends(vertex_dict)
        proportion_3way_intersections = compute_proportion_3way_intersections(vertex_dict)
        proportion_4way_intersections = compute_proportion_4way_intersections(vertex_dict)
        intersection_count = compute_intersection_count(vertex_dict)
        total_road_length = compute_total_road_length(road_network, config=config)

        print('Entropy:', entropy)
        print('Orientation-Order:', orientation_order)
        print('Average Node Degree:', avg_node_degree)
        print('Proportion Dead-Ends:', proportion_dead_ends)
        print('Proportion 3-way Intersections', proportion_3way_intersections)
        print('Proportion 4-way Intersections', proportion_4way_intersections)
        print('Intersection Count:', intersection_count)
        print('Total Road Length:', total_road_length)

    if show_city:
        # visualise(config.water_map_array, road_network, land_usages=land_usages)
        # visualiser.visualise()
        ## keep plt showing:
        plt.ioff()
        plt.show()

    if show_stats:
        show_orientation_histogram(orientation_histogram)




if __name__ == "__main__":
    random.seed(42)
    generate(os.getcwd() + "/input/configs/auckland.json", show_city=True, show_time=False, show_stats=False)
