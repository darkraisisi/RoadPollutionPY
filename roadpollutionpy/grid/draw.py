import matplotlib.pyplot as plt
import pandas as pd
# from math import sin, cos, sqrt, atan2, radians, pi, asin
import math
import time 
import numpy as np
import concurrent.futures

path = "maps/norm/"
extension = ".json"

mapName, name = ["baarnWays_way","baarn"]
# mapName, name = ["utrechtSurWays_way","utrechtSur"]
# mapName, name = ["utrechtProvWays_way","utrechtProv"]

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

def readFromFile(name:str) -> pd.DataFrame:
    print(path+name+extension)
    return pd.json_normalize(pd.read_json(path+name+extension,typ='series', dtype=object))


def calculateDistance(lat1, lat2, lon1, lon2):
    p = math.pi/180
    a = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p) * math.cos(lat2*p) * (1-math.cos((lon2-lon1)*p))/2
    return 12742 * math.asin(math.sqrt(a))


def drawScatterPlotFrom(data:pd.DataFrame) -> None:
    # data.plot(kind='scatter',x='lon',y='lat',color='red')
    lon_min = min(data['lon'])
    lon_max = max(data['lon'])
    lat_min = min(data['lat'])
    lat_max = max(data['lat'])

    lonInKm = calculateDistance(lat_min,lat_min,lon_min,lon_max)
    latInKm = calculateDistance(lat_min,lat_max,lon_min,lon_min)

    print(math.gcd(int(lonInKm*1000),int(latInKm*1000)))
    print(((lon_max - lon_min)*1000),((lat_max - lat_min)*1000))
    print(((lon_max - lon_min)*1000)/lonInKm,((lat_max - lat_min)*1000)/latInKm)
    print(lon_min,lon_max,lat_min,lat_max)
    print(lonInKm)
    print(latInKm)

    startTime = time.time()
    for node in data.itertuples():
        if(node[1] =='way'):

            lon = []
            lat = []
            for roadPiece in node.nodes:
                # print(node)
                # print(data.loc[(data['id'] == roadPiece) & (data['type'] == 'node')])
                # _,_,_,lat,lon,_,_,_,_,_ = data.loc[(data['id'] == roadPiece) & (data['type'] == 'node')]
                out = data.loc[(data['id'] == roadPiece) & (data['type'] == 'node')]
                lon.append(float(out['lon']))
                lat.append(float(out['lat']))

            plt.plot(lon,lat,color='green')
    print(f'elapsed time to looop through connected way nodes{time.time() - startTime}')
    plt.show()


data = readFromFile(mapName)
drawScatterPlotFrom(data)