import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

"""
Distance between longitude and latitude lines
Source: https://stackoverflow.com/a/21623206/7271827
"""

"""
Source: https://scipython.com/book/chapter-7-matplotlib/examples/the-two-dimensional-diffusion-equation/
A diffusion equation to minamilly disperse the produced NOx

Diffusion flux of NOx generally is ~1.5 nmol mol^-1 or 
https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2003JD004326
"""

"""
Source : https://www.sciencedirect.com/science/article/abs/pii/S1352231000000741
C is ht concentration
Q1 is the source strength per unit length (mass/(time-length))
u is the average wind speed
theta is the angle between the wind direction and the road
IMPORTANT NOTE the angle is a value between -89 and 90 , as the outer most values produce an infintely large sigmaZ/Y
Yr/Xr are the receptor points
X1/Y1 & X2/Y2 are the start and end of the line source respectively
erf is the error function
sigZ and sigY are the vertival and lateral dispersion parameters,
these are calculated by a formula with the downwind distance as the parameter
"""

def calcSigZ(x):
    return 0.14*x*math.pow(1+0.0003*x,-0.5)


def calcSigY(x):
    return 0.16*x*math.pow(1+0.0004*x,-0.5)


def limit(Xr,Yr,Xi,Yi,theta):
    # calculates the respective distance between the receptor and the given point to create a limit
    return (((Yr - Yi)*math.cos(theta)-(Xr - Xi)*math.sin(theta))/
    (math.sqrt(2*sigY)))


def concentration(Q,U,Xr,Yr,X1,Y1,X2,Y2,theta,downWindDistance):
    u1 = limit(Xr,Yr,X1,Y1,theta)
    u2 = limit(Xr,Yr,X2,Y2,theta)
    # x = downwind distance from the source to the receptor
    sigZ = calcSigZ(downWindDistance)
    sigY = calcSigY(downWindDistance)
    return ((Q / math.sqrt(2*math.pi)) * 
    (1 / (U*math.cos(theta)*sigZ)) *
    (math.erf(u1)-math.erf(u2)))

def showWindAngleImpact(Q,U,Xr,Yr,X1,Y1,X2,Y2):
    x = []
    y = []
    for i in range(-89,90,1):
        # print(i)
        # print(concentration(Q,U,Xr,Yr,X1,Y1,X2,Y2,math.radians(i)))
        x.append(i)
        y.append(concentration(Q,U,Xr,Yr,X1,Y1,X2,Y2,math.radians(i),0))

    plt.plot(x,y)
    plt.show()

def calculateDistance(lat1, lat2, lon1, lon2):
    p = math.pi/180
    a = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p) * math.cos(lat2*p) * (1-math.cos((lon2-lon1)*p))/2
    return 12742 * math.asin(math.sqrt(a))


def getBounds(matrixShape,lon_min,lon_max,lat_min,lat_max):
    print(matrixShape)
    print(lon_min,lon_max,lat_min,lat_max)
    print(matrixShape)
    bounds = np.zeros((matrixShape[0],matrixShape[1],2,2))
    hightFrac = (lat_max - lat_min) / matrixShape[0]
    widthFrac = (lon_max - lon_min) / matrixShape[1]
    print(hightFrac,widthFrac)

    currHight = lat_max
    currWidth = lon_min

    for rowIndex in range(0, matrixShape[0]):
        newHight = currHight - hightFrac
        for columIndex in range(0, matrixShape[1]):
            newWidth = currWidth + widthFrac
            bounds[rowIndex][columIndex] = [[newHight,currHight],[currWidth,newWidth]]
            currWidth = newWidth

        currWidth = lon_min
        currHight = newHight
  
    return bounds


def calc(df:pd.DataFrame,roads:list = None) -> np.array:
    startTime = time.time()
    lon_min = min(df['lon'])
    lon_max = max(df['lon'])
    lat_min = min(df['lat'])
    lat_max = max(df['lat'])

    print(lat_min,lat_min,lon_min,lon_max)

    lonInKm = calculateDistance(lat_min,lat_min,lon_min,lon_max)
    latInKm = calculateDistance(lat_min,lat_max,lon_min,lon_min)

    matrix = np.zeros(
    (math.ceil( (((lat_max - lat_min)*1000)/latInKm)*3),
    math.ceil( (((lon_max - lon_min)*1000)/lonInKm)*3 )))

    print(matrix)

    bounds = getBounds(matrix.shape,lon_min,lon_max,lat_min,lat_max)
    print(bounds.shape)
    # print(getNodesInBounds(df,bounds[4][7]))
    waysDf = df.loc[(df['type'] == 'way')]
    i = 0
    # go through the different bounds
    for rowIndex in range(0, len(bounds)):
        for columIndex in range(0, len(bounds[rowIndex])):
            centerLat, centerLon = ((bounds[rowIndex][columIndex][0][0] + bounds[rowIndex][columIndex][0][1]) / 2) , ((bounds[rowIndex][columIndex][1][0] + bounds[rowIndex][columIndex][1][1]) / 2)
            # get all the nodes that are in the current bound
            nodesInBound = getNodesInBounds(df,bounds[rowIndex][columIndex])
            # get all the wayNodes that have one more nodes that are in the current bound
            _set = set(nodesInBound['id'])
            ways = waysDf.loc[ ~waysDf['nodes'].map(_set.isdisjoint)]
            # go through all the individual ways in in this bound and intersect the way that is in this bound so no connection is calculated multiple times
            for way in ways.nodes:
                intersections = list(_set.intersection(way))
                for wayIndex in range(0,len(intersections)-1):
                    currNode, nextNode = nodesInBound.loc[nodesInBound.id == intersections[wayIndex]] , nodesInBound.loc[nodesInBound.id == intersections[wayIndex+1]]
                    lineLength = calculateDistance(float(currNode.lat),float(nextNode.lat),float(currNode.lon),float(nextNode.lon))
                    ret = concentration(Q,U,centerLon,centerLat,currNode.lon,currNode.lat,nextNode.lon,nextNode.lat,theta,(lineLength/3))
                    matrix[rowIndex][columIndex] += ret

            # print(f'Amount of bounds calculated:{i}')
            i+=1

    # print(f'1 cycle of calculations took {time.time() - startTime} seconds.')
    return matrix


def getNodesInBounds(df:pd.DataFrame,bounds:list):
    # bounds[0] = [start latitude, end latitude line]
    # bounds[1] = [start longitude, end longitude]
    # lat 52.193919  lon 5.303260
    return df.loc[(df['lat'] >= bounds[0][0]) & (df['lat'] < bounds[0][1]) &
    (df['lon'] >= bounds[1][0]) & (df['lon'] < bounds[1][1])
    ]


Q = 10
U = 3 # 3 m/s average wind speed
theta = 40 # 
# Xr, Yr = 1800, 9200
Xr, Yr = 1200, 9800
X1,Y1, X2,Y2 = 1000, 9000, 2000, 10000

sigZ = 4.5 # a value based on the fact that a third of every road is downwind, based on 100m between nodes
sigY = 5   # a value based on the fact that a third of every road is downwind, based on 100m between nodes

# showWindAngleImpact(Q,U,Xr,Yr,X1,Y1,X2,Y2)