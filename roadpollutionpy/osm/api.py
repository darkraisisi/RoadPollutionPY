import config as con
import json
import overpass as op
# import multiprocessing as mp
import concurrent.futures

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
    if verbose:
        getDataframeTotalSize(df)

    # processes = []
    for item in df.iteritems():
        print(item[0] not in cols)
        if(item[0] not in cols):
            df.drop(item[0], axis=1, inplace=True)
            # p = mp.Process(target=dropCol,args=(item[0]))
            # p.start()
            # processes.append(p)

    # for p in processes:
    #     p.join()
        
    if verbose:
        getDataframeTotalSize(df)


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
res = getWayFromCoordinates(con.osm["coordinates"][name])
writeToFile(mapName,res)


# test draw nodes
# data = readFromFile(mapName)
# print(data.head)
# data.plot(kind='scatter',x='lon',y='lat',color='red')
# plt.show()


# normalize
cols = ['type', 'id', 'lat', 'lon', 'nodes', 'tags.name', 'tags.highway', 'tags.maxspeed', 'tags.surface']
df = readFromFile(mapName)
normalizeDataframe(df,cols,True)
dataframeToFile(df,mapName)

data = readFromFile('norm/'+mapName)
data.plot(kind='scatter',x='lon',y='lat',color='red')
plt.show()
