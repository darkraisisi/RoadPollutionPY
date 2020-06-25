import math
import time
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as conf
import draw


"""
Distance between longitude and latitude lines
Source: https://stackoverflow.com/a/21623206/7271827
"""

"""
Creating a realistic circle out of squares
https://en.wikipedia.org/wiki/Midpoint_circle_algorithm
Source for code function generateCirleCoordsList:
https://www.geeksforgeeks.org/mid-point-circle-drawing-algorithm/ by Nabaneet Roy & Tuhina Singh Improved and alterd by David Demmers
"""

"""
Using the emission factor to calculate actual emission
https://www.sciencedirect.com/topics/engineering/emission-factor 5.7
E = A * EF 
E: Emissions in units, A: Activity rate in unit distance, EF: Emission Factor in unit per pollutant
"""

"""
European source of emission factors for NOx and generalisation for car dispersion.
https://www.eea.europa.eu/publications/EMEPCORINAIR5/page016.html
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


def limit(Xr,Yr,Xi,Yi,theta,sigY):
    # calculates the respective distance between the receptor and the given point to create a limit
    return (((Yr - Yi)*math.cos(theta)-(Xr - Xi)*math.sin(theta))/
    (math.sqrt(2*sigY)))


def concentration(Q,U,Xr,Yr,X1,Y1,X2,Y2,theta,downWindDistance):
    # x = downwind distance from the source to the receptor
    sigZ = calcSigZ(downWindDistance)
    sigY = calcSigY(downWindDistance)
    # print(downWindDistance)
    # print(sigZ,sigY)

    u1 = limit(Xr,Yr,X1,Y1,theta,sigY)
    u2 = limit(Xr,Yr,X2,Y2,theta,sigY)
    
    return ((Q / math.sqrt(2*math.pi)) * 
    (1 / (U*math.cos(theta)*sigZ)) *
    (math.erf(u1)-math.erf(u2)))


def emissionfactorToEmission(distance, EF):
    return distance * EF


def emissionfactorToReducedEmission(distance, EF, ER):
    return emissionfactorToEmission(distance,EF)*(1-ER/100)


def showWindAngleImpact(Q,U,Xr,Yr,X1,Y1,X2,Y2):
    """
    Q = 10
    U = 3 # 3 m/s average wind speed
    theta = 40 # 
    # Xr, Yr = 1800, 9200
    Xr, Yr = 1200, 9800
    X1,Y1, X2,Y2 = 1000, 9000, 2000, 10000

    sigZ = 4.5 # a value based on the fact that a third of every road is downwind, based on 100m between nodes
    sigY = 5   # a value based on the fact that a third of every road is downwind, based on 100m between nodes
    """
    x = []
    y = []
    for i in range(-89,90,1):
        # print(i)
        # print(concentration(Q,U,Xr,Yr,X1,Y1,X2,Y2,math.radians(i)))
        x.append(i)
        y.append(concentration(Q,U,Xr,Yr,X1,Y1,X2,Y2,math.radians(i),0))

    plt.plot(x,y)
    plt.show()


def calculateDistanceKm(lat1, lat2, lon1, lon2):
    # In meters
    p = math.pi/180
    a = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p) * math.cos(lat2*p) * (1-math.cos((lon2-lon1)*p))/2
    return 12742 * math.asin(math.sqrt(a))


def calculateDistanceM(lat1, lat2, lon1, lon2):
    # In meters
    p = math.pi/180
    a = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p) * math.cos(lat2*p) * (1-math.cos((lon2-lon1)*p))/2
    if a == 0:
        return 1
    return 12742000 * math.asin(math.sqrt(a))


def getMinMaxDataframe(df:pd.DataFrame) -> tuple:
    return (min(df['lon']),
    max(df['lon']),
    min(df['lat']),
    max(df['lat']))
    

def getBoundsRanges(matrixShape,lon_min,lon_max,lat_min,lat_max):
    bounds = np.zeros((matrixShape[0],matrixShape[1],2,2))
    hightFrac = (lat_max - lat_min) / matrixShape[0]
    widthFrac = (lon_max - lon_min) / matrixShape[1]

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


def getBoundsNodelist(df,matrixShape,lon_min,lon_max,lat_min,lat_max):
    boundsRange = getBoundsRanges(matrixShape,lon_min,lon_max,lat_min,lat_max)
    nodesInBounds = []
    for rowIndex in range(0, matrixShape[0]):
        nodesInBounds.append([]) # Add a new row
        for columIndex in range(0, matrixShape[1]):
            nodesInBounds[rowIndex].append([]) # Add a column to the row
            nodes = getNodesInBoundsFromDf(df,boundsRange[rowIndex][columIndex])
            nodes.set_index('id')
            nodesInBounds[rowIndex][columIndex] = list(nodes.to_dict('records'))
    return nodesInBounds


def boundBasedConcentration(df:pd.DataFrame,roads:list = None) -> np.array:
    startTime = time.time()
    lon_min, lon_max, lat_min ,lat_max = getMinMaxDataframe(df)

    print(lat_min,lat_min,lon_min,lon_max)

    lonInKm = calculateDistanceKm(lat_min,lat_min,lon_min,lon_max)
    latInKm = calculateDistanceKm(lat_min,lat_max,lon_min,lon_min)

    matrix = np.zeros(
    (math.ceil( (((lat_max - lat_min)*1000)/latInKm)*6),
    math.ceil( (((lon_max - lon_min)*1000)/lonInKm)*6 )))

    bounds = getBoundsRanges(matrix.shape,lon_min,lon_max,lat_min,lat_max)
    # print(getNodesInBoundsFromDf(df,bounds[4][7]))
    nodeToWays = generateLookup(df)
    nodesInBounds = getBoundsNodelist(df,matrix.shape,lon_min,lon_max,lat_min,lat_max)
    i = 0
    j=0
    # go through the different bounds
    for rowIndex in range(0, len(bounds)):
        for columIndex in range(0, len(bounds[rowIndex])):
            centerLat, centerLon = ((bounds[rowIndex][columIndex][0][0] + bounds[rowIndex][columIndex][0][1]) / 2) , ((bounds[rowIndex][columIndex][1][0] + bounds[rowIndex][columIndex][1][1]) / 2)
            # get all the nodes that are in the current bound
            nodesInBound = nodesInBounds[rowIndex][columIndex]
            # get all the wayNodes that have one more nodes that are in the current bound
            currentWaysId = {}
            if len(nodesInBound) > 0:
                for nodeInBound in nodesInBound:
                    for wayId in nodeToWays[nodeInBound['id']]:
                        if wayId in currentWaysId:
                            nodeInBound.update({'order':nodeToWays[nodeInBound['id']][wayId]})
                            currentWaysId[wayId].insert(nodeInBound['order'],nodeInBound)
                        else:
                            nodeInBound.update({'order':nodeToWays[nodeInBound['id']][wayId]})
                            currentWaysId[wayId] = [nodeInBound]

                for currWayId in currentWaysId:
                    for wayIndex in range(0,len(currentWaysId[currWayId])-1):
                        j+=1
                        currNode, nextNode = currentWaysId[currWayId][wayIndex] ,  currentWaysId[currWayId][wayIndex+1]
                        lineLength = calculateDistanceKm(float(currNode['lat']),float(nextNode['lat']),float(currNode['lon']),float(nextNode['lon']))
                        ret = concentration(10,U,centerLon,centerLat,currNode['lon'],currNode['lat'],nextNode['lon'],nextNode['lat'],theta,(lineLength/4))
                        matrix[rowIndex][columIndex] += ret

                # print(f'Amount of bounds calculated:{i}')
                i+=1
    print('J:',j)
    print(f'1 cycle of calculations took {time.time() - startTime} seconds.')
    return matrix


def getNodesInBoundsFromDf(df:pd.DataFrame,bounds:list):
    # bounds[0] = [start latitude, end latitude line]
    # bounds[1] = [start longitude, end longitude]
    # lat 52.193919  lon 5.303260
    return df.loc[(df['lat'] >= bounds[0][0]) & (df['lat'] < bounds[0][1]) &
    (df['lon'] >= bounds[1][0]) & (df['lon'] < bounds[1][1])][['id','lat','lon','highway','maxspeed','surface']]


def generateLookup(df:pd.DataFrame):
    # Key: Node id
    # Value: List of dicts {Key: way id's, Value: order } 
    nodeLookup = {}
    for way in df.loc[df['type'] == 'way'].itertuples():
        for node in way.nodes:
            if node in nodeLookup:
                nodeLookup[node].update({way.id:way.nodes.index(node)})
            else:
                nodeLookup[node] = {way.id:way.nodes.index(node)}
    return nodeLookup


def waytypeToSpeed(roadTypeName:str):
    if roadTypeName in conf.sim['roads']['speeds']:
        return conf.sim['roads']['speeds'][roadTypeName]
    else:
        return 30


def effMoped(speed):
    # Table 8-30: Euro 3
    return 0.26


def effPassenger(speed):
    # Table 8-5, ECE 15-04, CC > 2.0 l, gasoline 
    return 2.427 - 0.014 * speed + 0.000266*math.pow(speed,2)


def effLightduty(speed):
    # Table 8-24, conventional, gasoline vehicles <3.5 t
    return 0.0179 * speed + 1.9547


def effHeavyduty(speed):
    # Table 8-28, gasoline vehicles >3.5 t
    if(speed < 60):
        return 4.5 # urban
    else:
        return 7.5 # Highway


def generateWayEF(df:pd.DataFrame):
    unhandled = set()
    wayLookup = {}
    
    for way in df.loc[df['type'] == 'way'].itertuples():
        _type = str(way.highway)
        if conf.sim['overwriting']['enabled'] and _type in conf.sim['overwriting']['roads_speeds']:
            speed = conf.sim['overwriting']['roads_speeds'][_type]
        elif(way.maxspeed):
            speed = int(way.maxspeed)
        else:
            speed = waytypeToSpeed(_type)

        if _type == 'service' or _type == 'services':
            eff, busy = (effPassenger(speed)*0.9 + effLightduty(speed)*0.1), 1
        elif _type == 'cycleway':
            eff, busy = effMoped(speed), 1
        elif _type == 'pedestrian' or _type == 'footway' or _type == 'path':
            eff, busy = effMoped(speed) * 0.1 , 1
        elif _type == 'motorway_link' or _type == 'motorway' or _type == 'highway' or _type == 'speedway':
            eff, busy = (effPassenger(speed)*0.55 + effLightduty(speed)*0.2 + effHeavyduty(speed)*0.25), 10
        elif _type == 'primary' or  _type == 'primary_link':
            eff, busy = (effPassenger(speed)*0.6 + effLightduty(speed)*0.2 + effHeavyduty(speed)*0.2), 8
        elif _type == 'secondary' or  _type == 'secondary_link':
            eff, busy = (effPassenger(speed)*0.7 + effLightduty(speed)*0.15 + effHeavyduty(speed)*0.15), 5
        elif _type == 'tertiary' or  _type == 'tertiary_link':
            eff, busy = (effPassenger(speed)*0.8 + effLightduty(speed)*0.1 + effHeavyduty(speed)*0.1), 3
        elif _type == 'residential' or _type == 'living_street':
            eff, busy = (effPassenger(speed)*0.9 + effLightduty(speed)*0.05 + effMoped(speed) * 0.05), 1
        else:
            unhandled.add(_type) # way unknown
            eff, busy = (effPassenger(speed)*0.8 + effLightduty(speed)*0.1 + effHeavyduty(speed)*0.1), 2

        wayLookup[way.id] = {'eff':eff,'busy':busy}
    print(f'Warning, unhandled ways:{unhandled}')
    return wayLookup


def generateIndexListFromCircumference(coords:list):
    startRow = min(coords,key=itemgetter(0))[0]
    endRow = max(coords,key=itemgetter(0))[0]
    startCol = min(coords,key=itemgetter(1))[1]
    endCol = max(coords,key=itemgetter(1))[1]

    outList = []

    inside = False
    for rowIndex in range(startRow,endRow+1):
        inside = False
        for colIndex in range(startCol,endCol+1):

            if((rowIndex,colIndex) in coords):
                inside = not inside
                outList.append((rowIndex,colIndex))
                continue

            if(inside):
                outList.append((rowIndex,colIndex))
            
    return outList


def getNodesInBoundsByIndex(nodesInBounds,coords):
    retNodes = []
    for coord in coords:
        if coord[0] in range(-len(nodesInBounds), len(nodesInBounds)):
            if coord[1] in range(-len(nodesInBounds[coord[0]]), len(nodesInBounds[coord[0]])):
                retNodes.append(nodesInBounds[coord[0]][coord[1]])

    return retNodes


def generateCirleCoordsList(r,start:tuple):
    # r, radius is the boundingboxsize multiplier. bbox 100x100m & 1r = r = 100m OR bbox 100x100m & 2r = r = 200m
    # start, starting position of the circle
    # 5,8
    y_centre, x_centre = start
    x = r 
    y = 0
    # pointList = set() 
    pointList = []
      
    # When radius is zero only a single  
    # point be printed  
    if (r > 0) : 
        pointList.append((-y + y_centre,x + x_centre))
        pointList.append((x + y_centre,y + x_centre)) 
        pointList.append((y_centre,-x + x_centre))
      
    # Initialising the value of P  
    P = 1 - r  
  
    while x > y: 
      
        y += 1
          
        # Mid-point inside or on the perimeter 
        if P <= 0:  
            P = P + 2 * y + 1
              
        # Mid-point outside the perimeter  
        else:          
            x -= 1
            P = P + 2 * y - 2 * x + 1
          
        # All the perimeter points have  
        # already been printed  
        if (x < y): 
            break
          
        # Append the generated point its reflection  
        # in the other octants after translation  
        pointList.append((y + y_centre,x + x_centre))
        pointList.append(( y + y_centre,-x + x_centre)) 
        pointList.append((-y + y_centre,x + x_centre))
        pointList.append((-y + y_centre,-x + x_centre))
          
        # If the generated point on the line x = y then  
        # the perimeter points have already been appended  
        if x != y: 
            pointList.append((x + y_centre, y + x_centre))
            pointList.append((x + y_centre,-y + x_centre)) 
            pointList.append((-x + y_centre, y + x_centre))
            pointList.append((-x + y_centre, -y + x_centre))

    return list(pointList)


def receptorpointBasedConcentration(df:pd.DataFrame,windSpeed:int,windAngle:int,radius:int,bboxSize:int = 100) -> np.array:
    """
    a not working! /maybe/ better average downwind distance
    averageDownwind = calculateDistanceM(
        centerLat, (float(currNode['lat'])+float(nextNode['lat']))/2,
        centerLon,(float(currNode['lon'])+float(nextNode['lon']))/2)
    """

    startTime = time.time()
    total = 0
    downWindFrac = conf.sim['downwind']

    lon_min, lon_max, lat_min ,lat_max = getMinMaxDataframe(df)

    lonInKm = calculateDistanceM(lat_min,lat_min,lon_min,lon_max)
    latInKm = calculateDistanceM(lat_min,lat_max,lon_min,lon_min)
    
    if(conf.sim['verbose']):
        print(lon_min, lon_max, lat_min ,lat_max)
        print('lonInKm',lonInKm,'latInKm',latInKm)
        print('lon/100 ceil',math.ceil(lonInKm/bboxSize),'lat/100 ceil',math.ceil(latInKm/bboxSize))

    concentrationMatrix = np.zeros(
        (math.ceil(latInKm/bboxSize),math.ceil(lonInKm/bboxSize))
    )

    nodeToWays = generateLookup(df)
    wayIdToInfo = generateWayEF(df)
    nodesInBounds = getBoundsNodelist(df,concentrationMatrix.shape,lon_min,lon_max,lat_min,lat_max)
    latFreq = ((lat_max - lat_min) / concentrationMatrix.shape[0])
    lonFreq = ((lon_max - lon_min) / concentrationMatrix.shape[1])
    if(conf.sim['verbose']):
        print(f'Prep took: {time.time() - startTime}s')
    # depending on the size of the bounds you make the radius has a different impact as multiplier
    for rowIndex in range(0, len(concentrationMatrix)):
        centerLat = lat_max - (latFreq * rowIndex)
        for colIndex in range(0, len(concentrationMatrix[rowIndex])):
            total +=1
            centerLon = lon_min + (lonFreq * colIndex)
            circumference = generateCirleCoordsList(int(radius/bboxSize)-1,(rowIndex,colIndex))
            areaList = generateIndexListFromCircumference(circumference)
            nodesList = getNodesInBoundsByIndex(nodesInBounds,areaList)
            currentWaysId = {}
            for nodes in nodesList:
                for nodeInBound in nodes:
                    if(nodeInBound['id'] in nodeToWays):
                        for wayId in nodeToWays[nodeInBound['id']]:
                            if wayId in currentWaysId:
                                nodeInBound.update({'order':nodeToWays[nodeInBound['id']][wayId]})
                                currentWaysId[wayId].insert(nodeInBound['order'],nodeInBound)
                            else:
                                nodeInBound.update({'order':nodeToWays[nodeInBound['id']][wayId]})
                                currentWaysId[wayId] = [nodeInBound]

            for currWayId in currentWaysId:
                EF = wayIdToInfo[currWayId]['eff']
                busyness = wayIdToInfo[currWayId]['busy']
                for wayIndex in range(0,len(currentWaysId[currWayId])-1):
                    currNode, nextNode = currentWaysId[currWayId][wayIndex] ,  currentWaysId[currWayId][wayIndex+1]
                    lineLength = calculateDistanceM(float(currNode['lat']),float(nextNode['lat']),float(currNode['lon']),float(nextNode['lon']))
                    emission = emissionfactorToEmission(lineLength/1000,EF*busyness)
                    ret = concentration(emission,windSpeed,centerLon,centerLat,currNode['lon'],currNode['lat'],nextNode['lon'],nextNode['lat'],windAngle,(lineLength*downWindFrac))
                    concentrationMatrix[rowIndex][colIndex] += ret

            # print(f'Receptor {total}/{concentrationMatrix.size}')
    if(conf.sim['verbose']):
        print(f'Full setup and 1 cycle of concentration took: {time.time() - startTime}s')
    return concentrationMatrix

