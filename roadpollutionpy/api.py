import config as conf
import json
import overpass as op
# import multiprocessing as mp
import concurrent.futures
import time 

import matplotlib.pyplot as plt
import pandas as pd


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


def writeToFile(fullPath:str,data:str) -> None:
    """
    Writes text to a file at a given path.

    Parameters:
        fullPath (str): A string containing the full path where the file needs to be written including the filename and extension.
        data (str): Data in the shape of json.
        
    Returns:
        response (JSON): A json with the nodes and ways requesten in the bounds.
    """
    with open(fullPath,'w') as outfile:
        json.dump(data['elements'],outfile)


def readFromFile(name:str) -> pd.DataFrame:
    """
    Get a Dataframe form json file.

    Parameters:
        name (str): A string containing the name of the file.
        
    Returns:
        response (Pandas.DataFrame): A dataframe with all the nodes and ways.
    """
    return pd.json_normalize(pd.read_json(conf.osm['path']+name+conf.osm['extension'],typ='series', dtype=object))


def normalizeDataframe(_df:pd.DataFrame,cols:list,verbose=False):
    """
    Normalizes a Dataframe, deletes unnececary columns and strips prexfixes.

    Parameters:
        _df (Pandas.DataFrame): a dataframe
        cols (list[(str)]): A list of column names that need to be kept.
        verbose (boolean): Flag for turning ong verbosity.
        
    Returns:
        response (JSON): A json with the nodes and ways requesten in the bounds
    """
    toRm = []
    for dfCol in _df.columns: 
        if(dfCol not in cols):
            toRm.append(dfCol)

    if verbose:
        getDataframeTotalSize(_df)
        startTime = time.time()

    _df.drop(toRm, axis=1, inplace=True)
    _df.set_index('id')
    _df = _df.drop_duplicates('id',keep='first')

    toRename = {}
    for dfCol in _df.columns: 
        if 'tags.' in dfCol:
            toRename.update({dfCol:dfCol.replace('tags.','')})
    _df.rename(columns=toRename,inplace=True)


    if verbose:
        getDataframeTotalSize(_df)
        print(f'normalizing took {time.time() - startTime} seconds')

    return _df


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


def dataframeToFile(df:pd.DataFrame,fullPath:str) -> None:
    """
    writes a dataframe to a file.

    Parameters:
        df (Pandas.DataFrame): a dataframe that you want to write away.
        fullPath (str): A string containing the full path where the file needs to be written including the filename and extension.
        
    """
    df.to_json(fullPath,orient='records')

