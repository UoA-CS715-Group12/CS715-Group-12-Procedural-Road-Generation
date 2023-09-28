# Procedural road generation in Python

This is the road generation module that generates road networks based on various input maps and creates a json file for
visualisation.

For the visualisation module, see: ... (TODO: insert repo link)

## Acknowledgements:

This repository is built based on an existing implementation of the extended L-system for road generation:
https://github.com/x775/citygenerator

All changes after the first commit is our own implementation.

## Getting started:

### K-means clustering

This script generates a list of population density centres and their positions given an input image. The results are
saved in the file `pop_density_centres.json`

```
python src/generate_pop_density_centres.py <image-path>
```

For example, the following script runs it on the Greater Auckland area (Windows 10 filepath):

```
python src/generate_pop_density_centres.py input/images/greater_auckland/greater_auckland_pop_density.png
```

### Road Generation

```
pip install -r requirements.txt
```

Run the `citygenerator.py` file either using an IDE or the command line

# Acknowledgements:
- Input images are source from [Koordinates.com](https://koordinates.com/) and [Mapbox](https://www.mapbox.com/)