import config as con
import json
import overpass as op

import matplotlib.pyplot as plt
import pandas as pd

path = "maps/"
extension = ".json"

def init():
    return op.API(timeout=180,debug=False)


def getWayFromCoordinates(coordinates:str) -> str:
    api = init()
    response = api.get(f'node({coordinates})', responseformat="json")
    print(list(response.keys()))
    return response


def writeToFile(name:str,data:str) -> None:
    with open(path+name+extension,'w') as outfile:
        json.dump(data['elements'],outfile)


def readFromFile(name:str) -> pd.DataFrame:
    # with open(name+extension,'r') as outfile:
    #     data = json.load(outfile)
    #     return data
    
    return pd.json_normalize(pd.read_json(path+name+extension,typ='series', dtype=object))


mapName, name = ["baarnWays_map","baarn"]
# mapName, name = ["utrechtProvWays_map","utrecht"]

print(mapName,name)
res = getWayFromCoordinates(con.osm["coordinates"][name])
writeToFile(mapName,res)

# test draw nodes
data = readFromFile(mapName)
data.plot(kind='scatter',x='lon',y='lat',color='red')
plt.show()