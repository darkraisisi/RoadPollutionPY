import concurrent.futures
import math
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
if (__name__ == 'draw'):
    import config as conf
else:
    from roadpollutionpy import config as conf


def readFromFile(name) -> pd.DataFrame:
    """
    Get a Dataframe form json file.

    Parameters:
        name (str): A string containing the name of the file.
        
    Returns:
        response (Pandas.DataFrame): A dataframe with all the nodes and ways.
    """
    if(conf.draw['verbose']):
        print(conf.draw['path'])
    df = pd.json_normalize(pd.read_json(conf.draw['path'],typ='series', dtype=object))
    df.set_index('id')
    return df


def plot(data:pd.DataFrame, roads:list=None) -> None:
    """
    Draw and show a figure with all the roads in the given dataframe.

    Parameters:
        data (Pandas.DataFrame): A dataframe containing a map with nodes and ways
        
    Returns:
        None: Shows a figure
    """
    if(conf.draw['verbose']):
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
    if(conf.draw['verbose']):
        print(f'Looping through all ways took {time.time() - startTime}s, {i} amount of ways')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.show()


def imagePlot(matrix, name, bboxSize, radius):
    """
    Draw and show an figure with the values of a concentration matrix

    Parameters:
        matrix (list[list[]]): a 2 dimensional matrix with numerical values in the cells.
        name (str): The name of the plot.
        bboxSize (int): The size of the bounding boxes used.
        radius (int): The radius around the receptor points used.
        
    Returns:
        None: Shows a figure
    """
    plt.imshow(matrix, cmap='hot', interpolation='quadric')
    plt.colorbar()
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title(f'{name} bbox{bboxSize}_R{radius}')
    plt.show()


def concentrationAndRoads(concentrationMatrix, data:pd.DataFrame, roads:list=None):
    """
    Draw and show an figure with the values of a concentration matrix and all roads overlayed.

    Parameters:
        matrix (list[list[]]): a 2 dimensional matrix with numerical values in the cells.
        data (Pandas.DataFrame): A dataframe containing a map with nodes and ways.
        roads (list[str]): a list of roads that you want to draw, none suggests you want to draw all is default.
        
    Returns:
        None: Shows a figure
    """
    # This function is an attempt to add both plots in one figure
    fig = plt.figure()
    concentrationFig = fig.add_subplot(111,label='concentration')
    wayFig = fig.add_subplot(111,label='roads', frame_on=False)
    concentrationFig.imshow(concentrationMatrix, cmap='hot', interpolation='quadric')
    concentrationFig.set_xlabel('boundCol', color='red')
    concentrationFig.set_ylabel('boundRow', color='red')
    concentrationFig.tick_params(axis='x', colors="red")
    concentrationFig.tick_params(axis='y', colors="red")

    if(roads):
        for node in data.loc[(data['highway'].isin(roads))].itertuples():
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

