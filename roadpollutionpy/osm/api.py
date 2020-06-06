import config as con
import json
import overpass as op
# import multiprocessing as mp
import concurrent.futures
import time 

import matplotlib.pyplot as plt
import pandas as pd

path = "maps/"
extension = ".json"
cols = ['type', 'id', 'lat', 'lon', 'nodes', 'tags.name', 'tags.highway', 'tags.maxspeed', 'tags.surface']


def init():
    return op.API(timeout=180,debug=False)


def getWayFromCoordinates(coordinates:str) -> str:
    api = init()
    response = api.get(f'way({coordinates})["highway"];(._;>;);', responseformat="json")
    print(list(response.keys()))
    return response


def writeToFile(name:str,data:str) -> None:
    with open(path+name+extension,'w') as outfile:
        json.dump(data['elements'],outfile)


def readFromFile(name:str) -> pd.DataFrame:
    return pd.json_normalize(pd.read_json(path+name+extension,typ='series', dtype=object))


def normalizeDataframe(df,cols,verbose=False):
    toRm = []
    for dfCol in df.columns: 
        if(dfCol not in cols):
            toRm.append(dfCol)

    if verbose:
        getDataframeTotalSize(df)
        startTime = time.time()

    df.drop(toRm, axis=1, inplace=True)
    df.set_index('id')
    df = df.drop_duplicates('id',keep='first')

    if verbose:
        getDataframeTotalSize(df)
        print(f'normalizing took {time.time() - startTime} seconds')


def dropCol(df,colName):
    df.drop(colName, axis=1, inplace=True)


def getDataframeTotalSize(df):
    size=0
    for i in df.memory_usage(index=True,deep=True): 
        size+=i
    print(size)


def dataframeToFile(df:pd.DataFrame,name):
    df.to_json(path+'norm/'+name+extension,orient='records')

# mapName, name = ["baarnWays_way","baarn"]
mapName, name = ["utrechtSurWays_way","utrechtSur"]
# mapName, name = ["utrechtProvWays_way","utrechtProv"]

print(mapName,name)
# res = getWayFromCoordinates(con.osm["coordinates"][name])
# writeToFile(mapName,res)

# normalize
cols = ['type', 'id', 'lat', 'lon', 'nodes', 'tags.name', 'tags.highway', 'tags.maxspeed', 'tags.surface']
df = readFromFile(mapName)
normalizeDataframe(df,cols,True)
dataframeToFile(df,mapName)

data = readFromFile('norm/'+mapName)
data.plot(kind='scatter',x='lon',y='lat',color='red')
plt.show()
