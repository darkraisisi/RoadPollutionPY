# RoadPollutionPY
RoadPollutionPY is a simulation tool that helps visualize the effect that different maximum speeds have on pollution air concentration created by wheeled vehicles


# Installation
### Install OpenStreetMap Overpass API libary:
`pip install overpass`

### Installing plotting libary:
`python -m pip install -U matplotlib`

### Install pandas:
[Official installation guide](https://pandas.pydata.org/getting_started.html)


# Using the simulator

## Overview
All the important things are located under the folder `roadpollutionpy/`.

`roadpollutionpy/config.py`: this folder has a bunch of user editable variables that work with other set variables.

`roadpollutionpy/roadpollution.py`: is the user oriented file to run, this will promt you for the action you want to perform based on the parameters set in the `config.py`.

`roadpollutionpy/api.py`: this is the file that interacts with the `overpass` api and download map data, this file will also handle normalisation(memory oriented).

`roadpollutionpy/draw.py`: this file will handle all things regarding drawing, this is done with `matplotlib`.

`roadpollutionpy/poll.py`: saving the best for last, this file will handle all the pollution related calculations.

## Running
To run a simulation i suggest you to run it from a relative position like: `C:/Python383/python.exe "C:/path/to/repo/RoadPollutionPY/roadpollutionpy/roadpollution.py"`

To run a test: `C:/Python383/python.exe -m pytest .\test_algorithm.py`


# Troubleshooting
Problems with saving or reading maps?<br>
Check if the folders `maps/norm/` are present, check the config for path variables