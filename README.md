<h1 align="center">Procedural Road Generation</h1>

<div align="center">

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
</div>

<div align="center">
This repository contains the source code for a novel procedural road generation algorithm that leverages terrain, height, and population density data.

<br/>

The algorithms utilised include [A Star](https://en.wikipedia.org/wiki/A*_search_algorithm), [L-System](https://en.wikipedia.org/wiki/L-system), and [relative neighbourhood graph](https://en.wikipedia.org/wiki/Relative_neighborhood_graph).

The Unity 3D visualisation module can be found [here](https://github.com/UoA-CS715-Group12/Unity-Visualisation)

<img src="https://github.com/UoA-CS715-Group12/CS715-Group-12-Procedural-Road-Generation/assets/69305722/7f6861d9-56bb-44b5-9f4a-29178c562fc3" alt="image" width="600"/>

<br/>

</div>

## Contents

- [Contents](#contents)
- [📋 Requirements](#-requirements)
- [👩‍🏫 Getting started:](#-getting-started)
  - [K-means clustering](#k-means-clustering)
  - [Road Generation](#road-generation)
- [⚙️ Input Configurations](#️-input-configurations)
  - [Images](#images)
  - [Population density centres and other parameters](#population-density-centres-and-other-parameters)
  - [A\* and (Legacy) Minimum Spanning Tree (MST)](#a-and-legacy-minimum-spanning-tree-mst)
- [📖 Acknowledgements:](#-acknowledgements)
- [💻 Credits](#-credits)

## 📋 Requirements
This project was run on Python 3.11 using a Windows machine. For the complete list of dependencies, check `requirements.txt`.

## 👩‍🏫 Getting started:
1. Clone the repository using `git clone`.

2. It is recommended to use a [virtual environment](https://docs.python.org/3/library/venv.html) to install the python packages to avoid conflicts.

2. Run `pip install -r requirements.txt` in the **root directory** of the package.

3. We included the Python 3.11 version of the GDAL package for a windows 64-bit machine in the repository. For other versions and operating systems, instructions are found [here](https://github.com/cgohlke/geospatial-wheels/).

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

Run the following command in the command line:
```
python citygenerator.py
```

## ⚙️ Input Configurations

### Images
Image configurations are located in the file `input/configs/auckland.json`. Modify the file names for the input files as needed.

Currently, we are using a population density, coastline, and water depth map obtained from [Koordinates.com](https://koordinates.com/).

### Population density centres and other parameters
The centers JSON file can be found in `input/configs/json`. Adjust the number of population density centers in the `generate` function within the file `citygenerator.py`.


### A* and (Legacy) Minimum Spanning Tree (MST)
Parameters used for A* and the MST can be found as a list of constants at the top of the file `a_star.py`.

Note: Increasing these parameters will significantly increase the runtime.

## 📖 Acknowledgements:
This repository is built based on an existing implementation of the extended L-system for road generation:
https://github.com/x775/citygenerator

All changes after the first commit is our own implementation

- Input images are source from [Koordinates.com](https://koordinates.com/) and [Mapbox](https://www.mapbox.com/)

## 💻 Credits
This project was created by the following people as part of their CS715 university project.

- Alex Liang
- Nick Huang
- Bowen Xiang
- Tony Cui
- Benjamin Goh

