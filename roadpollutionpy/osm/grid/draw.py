import matplotlib.pyplot as plt
import pandas as pd

extension = ".json"
mapName, name = ["baarnWays_map","baarn"]

def readFromFile(name:str) -> pd.DataFrame:
    # with open(name+extension,'r') as outfile:
    #     data = json.load(outfile)
    #     return data
    
    return pd.json_normalize(pd.read_json(name+extension,typ='series', dtype=object))


def drawScatterPlotFrom(data:pd.DataFrame) -> None:
    data = readFromFile(mapName)
    data.plot(kind='scatter',x='lon',y='lat',color='red')
    plt.show()