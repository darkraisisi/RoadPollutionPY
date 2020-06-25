import matplotlib.pyplot as plt
import pandas as pd
# from math import sin, cos, sqrt, atan2, radians, pi, asin
import math
import time 
import numpy as np
import concurrent.futures
import poll as pl
import sys
import config as conf

path = "maps/norm/"
extension = ".json"

mapName, name = ["baarnWays_way","baarn"]
# mapName, name = ["utrechtSurWays_way","utrechtSur"]
# mapName, name = ["utrechtProvWays_way","utrechtProv"]

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

def readFromFile(name:str) -> pd.DataFrame:
    print(path+name+extension)
    df = pd.json_normalize(pd.read_json(path+name+extension,typ='series', dtype=object))
    df.set_index('id')
    return df


def drawPlot(data:pd.DataFrame, roads:list=None) -> None:
    # data.plot(kind='scatter',x='lon',y='lat',color='red')

    startTime = time.time()
    data = data.drop_duplicates('id',keep='first')
    
    i=0
    if(roads):
        for node in data.loc[(data['tags.highway'].isin(roads))].itertuples():
            out = data.iloc[(pd.Index(data['id']).get_indexer(node.nodes))]
            plt.plot(out.lon,out.lat,color='green')
            i+=1
    else:
        # Go through all ways
        for node in data.loc[(data['type'] == 'way')].itertuples():
            # Get the current way and look at the nodes that mare the road and get a list returned based on the id's
            out = data.iloc[(pd.Index(data['id']).get_indexer(node.nodes))]
            # plot the two lists of longitude and latitude lines
            plt.plot(out.lon,out.lat,color='green')
            i+=1

    print(f'Looping through all ways took {time.time() - startTime}s, {i} amount of ways')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.show()


def drawImagePlot(matrix, name, bboxSize, radius):
    plt.imshow(matrix, cmap='hot', interpolation='quadric')
    plt.colorbar()
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title(f'{name} bbox{bboxSize}_R{radius}')
    plt.show()


def drawConcentrationAndRoads(concentrationMatrix, data:pd.DataFrame):
    fig = plt.figure()
    concentrationFig = fig.add_subplot(111,label='concentration')
    wayFig = fig.add_subplot(111,label='roads', frame_on=False)
    concentrationFig.imshow(concentrationMatrix, cmap='hot', interpolation='quadric')
    concentrationFig.set_xlabel('boundCol', color='red')
    concentrationFig.set_ylabel('boundRow', color='red')
    concentrationFig.tick_params(axis='x', colors="red")
    concentrationFig.tick_params(axis='y', colors="red")

    if(rds):
        for node in data.loc[(data['highway'].isin(rds))].itertuples():
            out = data.iloc[(pd.Index(data['id']).get_indexer(node.nodes))]
            wayFig.plot(out.lon,out.lat,color='green', alpha=0.4)
    else:
        # Go through all ways
        for node in data.loc[(data['type'] == 'way')].itertuples():
            # Get the current way and look at the nodes that mare the road and get a list returned based on the id's
            out = data.iloc[(pd.Index(data['id']).get_indexer(node.nodes))]
            # plot the two lists of longitude and latitude lines
            wayFig.plot(out.lon,out.lat,color='green', alpha=0.4)

    wayFig.set_xlabel('longitude', color='green')
    wayFig.set_ylabel('latitude', color='green')
    wayFig.xaxis.tick_top()
    wayFig.yaxis.tick_right()
    wayFig.xaxis.set_label_position('top') 
    wayFig.yaxis.set_label_position('right') 
    plt.show()


data = readFromFile(mapName)
rds = conf.draw['roads']
# drawPlot(data,rds)
# drawPlot(data)

# matrix = pl.boundBasedConcentration(data,rds)
# drawImagePlot(matrix,name,bboxSize,radius)
bboxSize = conf.sim['bbox_size']
radius = conf.sim['radius']
windSpeed = conf.sim['wind_speed']
windAngle = conf.sim['wind_angle']
concentrationMatrix = pl.receptorpointBasedConcentration(data,windSpeed,windAngle,radius,bboxSize,rds)
# drawImagePlot(matrix,name,bboxSize,radius)
# drawImagePlot(matrix,name,concentrationFig,bboxSize,radius)
# drawPlot(data,wayFig)

drawConcentrationAndRoads(concentrationMatrix,data)