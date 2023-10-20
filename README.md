<h1 align="center">Procedural Road Generation</h1>

<div align="center">

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
</div>

<div align="center">
This repository contains the source code for a novel procedural road generation algorithm utilising terrain, height, and population density data.

<br/>

The algorithms used are [A Star](https://en.wikipedia.org/wiki/A*_search_algorithm) and the [L-System](https://en.wikipedia.org/wiki/L-system#:~:text=An%20L%2Dsystem%20consists%20of,generated%20strings%20into%20geometric%20structures.).

The Unity 3D visualisation module can be found [here](https://github.com/UoA-CS715-Group12/Unity-Visualisation)

![example_cover](https://github.com/UoA-CS715-Group12/Python-Lsystem/assets/61865484/9f3d9e8a-befb-4b11-b744-9c3b952598ec)

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
This project was run on Python 3.11 using a Windows machine. The full list of dependencies can be found in `requirements.txt`

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
Image configurations can be found in the file `input/configs/auckland.json`

You can change the names of the input files to be used accordingly.

Currently, we are using a population density, coastline, and water depth map sourced from [Koordinates.com](https://koordinates.com/).

### Population density centres and other parameters
The centres JSON file could be found in `input/configs/json`

Number of population density centres can be changed in the `generate` function within the file `citygenerator.py`


### A* and (Legacy) Minimum Spanning Tree (MST)
Parameters used for A* and the MST can be found as a list of constants at the top of the file `a_star.py`.

Note: Increasing the value of these parameters will significantly increase the runtime.

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

