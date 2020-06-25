import config as conf
import json
import overpass as op
# import multiprocessing as mp
import concurrent.futures
import time 

import matplotlib.pyplot as plt
import pandas as pd


def init():
    return op.API(timeout=180,debug=False)


def getWayFromCoordinates(coordinates:str) -> str:
    api = init()
    response = api.get(f'way({coordinates})["highway"];(._;>;);', responseformat="json")
    print(list(response.keys()))
    return response


def writeToFile(name:str,data:str) -> None:
    with open(conf.osm['path']+name+conf.osm['extension'],'w') as outfile:
        json.dump(data['elements'],outfile)


def readFromFile(name:str) -> pd.DataFrame:
    return pd.json_normalize(pd.read_json(conf.osm['path']+name+conf.osm['extension'],typ='series', dtype=object))


def normalizeDataframe(_df,cols,verbose=False):
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


def getDataframeTotalSize(df):
    size=0
    for i in df.memory_usage(index=True,deep=True): 
        size+=i
    print(size)


def dataframeToFile(df:pd.DataFrame,name):
    df.to_json(conf.osm['normalisation']['path']+name+conf.osm['extension'],orient='records')

name = conf.sim['current']

# res = getWayFromCoordinates(conf.osm["coordinates"][name])
# writeToFile(name,res)

# normalize
df = readFromFile(name)
normDf = normalizeDataframe(df,conf.osm['normalisation']['cols'],True)
dataframeToFile(normDf,name)

