import os
import json


# INPUT:    List, Dict, Dict
# OUTPUT:   -
def city_to_json(road_network, vertices, land_usages):
    output = {}

    # Save all vertices in road network.
    output['roadSegments'] = []
    output['roadVertices'] = []
    for vertex in vertices:
        output["roadVertices"].append({
            "position": {
                'x' : float(vertex.position[0]),
                'y' : float(vertex.position[1])
            }
        })

    # Save all segments in road network.
    for segment in road_network:
        output["roadSegments"].append({
            "startVertIndex" : vertices.index(segment.start_vert),
            "endVertIndex" : vertices.index(segment.end_vert)
        })

    # Save all polygons w/ land usage, population density, and population.
    output["land_usages"] = land_usages

    # Dump to json file.
    file_path = os.path.join(os.getcwd(), "output", "roadnetwork.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as out:
        json.dump(output, out)
