import config as conf
import json
import overpass as op
# import multiprocessing as mp
import concurrent.futures
import time 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def init():
    """
    Initializes the overpass api.

    Returns:
        op (API): An overpass api method.
    """
    return op.API(timeout=180,debug=False)


def getWayFromCoordinates(coordinates:str) -> str:
    """
    Retrieves a json file of the map data in specified the bounding box.

    Parameters:
        coordinates (str): A string of coordinates split by a space, order from bottom,left,top,right .
        
    Returns:
        response (JSON): A json with the nodes and ways requesten in the bounds.
    """
    api = init()
    response = api.get(f'way({coordinates})["highway"];(._;>;);', responseformat="json")
    print(list(response.keys()))
    return response


def writeToCsv(path:str,data:str) -> None:
    """
    Writes text to a file at a given path.

    Parameters:
        fullPath (str): A string containing the full path where the file needs to be written including the filename and extension.
        data (str): Data in the shape of json.
        
    Returns:
        response (JSON): A json with the nodes and ways requesten in the bounds.
    """
    data = pd.DataFrame(data['elements'])

    nodes = data[data['type'] == 'node'][['id','lat','lon','tags']]

    ways = data[data['type'] == 'way'][['id','tags','nodes']]
    print(ways.head())
    print(ways.dtypes)
    # ways['nodes'] = ways['nodes'].apply(np.array)

    nodes.to_csv(path+'_node.csv', index = False, header=True)
    ways.to_csv(path+'_way.csv', index = False, header=True)


def readFromFile(name:str) -> pd.DataFrame:
    """
    Get a Dataframe form json file.

    Parameters:
        name (str): A string containing the name of the file.
        
    Returns:
        response (Pandas.DataFrame): A dataframe with all the nodes and ways.
    """
    return pd.read_csv(conf.osm['path']+name+conf.osm['extension'])


def getDataframeTotalSize(df:pd.DataFrame) -> int:
    """
    Returns the size of a dataframe in memory

    Parameters:
        df (Pandas.DataFrame): a dataframe to be measured
        
    Returns:
        response (int): returns the size in bytes
    """
    size=0
    for i in df.memory_usage(index=True,deep=True): 
        size+=i
    print(size)

