import json
import os

import numpy as np

from src.road_network.segment import Segment
from src.utilities import (find_legend_centers, normalise_pixel_values,
                           parse_image, parse_json, read_tif_file)


class SingletonMeta(type):
    """
    Singleton Class
    """
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class ConfigManager(metaclass=SingletonMeta):
    def __init__(self,
                 config_path=None,
                 height_map_path=None,
                 water_map_path=None,
                 population_density_path=None):
        try:
            with open(config_path, "r") as config_file:
                for key, value in json.load(config_file).items():
                    setattr(self, key, value["value"])
        except FileNotFoundError:
            print("Incorrect or missing config file!")
            # break

        # Create starting segments based on config axiom.
        self.axiom = [Segment(segment_array=np.array(segment_coordinates)) for segment_coordinates in self.axiom]

        # Parse road rule map and population density map.
        path = os.getcwd() + "/input/images/"
        json_path = os.getcwd() + "/input/json/"
        self.road_rules_array = parse_image(path + self.rule_image_name)
        self.population_density_array = parse_image(path + self.population_density_image_name)
        self.population_density_array = normalise_pixel_values(self.population_density_array)  # convert to binary array for reading
        # find radial centers. Only relevant if radial road rule is used.
        self.radial_centers = find_legend_centers(self.road_rules_array, self.radial_legend)
        # Parse water map.
        self.water_map_array = parse_image(path + self.water_map_image_name)
        # Parse land usage map.
        self.land_use_array = read_tif_file(path + self.land_use_image_name)
        self.height_map_array = parse_image(path + self.height_map_image_name)
        self.pop_density_centres = parse_json(json_path + self.pop_density_centres_name)
