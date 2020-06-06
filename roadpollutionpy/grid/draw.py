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
    df = pd.json_normalize(pd.read_json(path+name+extension,typ='series', dtype=object))
    df.set_index('id')
    return df


def calculateDistance(lat1, lat2, lon1, lon2):
    p = math.pi/180
    a = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p) * math.cos(lat2*p) * (1-math.cos((lon2-lon1)*p))/2
    return 12742 * math.asin(math.sqrt(a))


def drawPlot(data:pd.DataFrame,roads=None) -> None:
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

    matrix = np.zeros((math.ceil(((lat_max - lat_min)*1000)/latInKm),math.ceil(((lon_max - lon_min)*1000)/lonInKm)))
    print(matrix.shape)
    print(matrix)
    data = data.drop_duplicates('id',keep='first')
    startTime = time.time()
    i=0
    if(roads):
        for node in data.loc[(data['tags.highway'].isin(roads))].itertuples():
            out = data.iloc[(pd.Index(data['id']).get_indexer(node.nodes))]
            plt.plot(out.lon,out.lat,color='green')
            i+=1
    else:
        for node in data.loc[(data['type'] == 'way')].itertuples():
            out = data.iloc[(pd.Index(data['id']).get_indexer(node.nodes))]
            plt.plot(out.lon,out.lat,color='green')
            i+=1

    print(f'Looping through all ways took {time.time() - startTime}s, {i} amount of ways')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.show()


data = readFromFile(mapName)
# rds = ['motorway_link','primary','secondary','tertiary']
# rds = ['motorway_link','primary','secondary','tertiary','residential','service']
# drawPlot(data,rds)
drawPlot(data)
