import math
import time
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if (__name__ == 'poll'):
    import config as conf
    import draw
    import algorithm as algo
else:
    from roadpollutionpy import config as conf
    from roadpollutionpy import draw
    from roadpollutionpy import algorithm as algo
    from roadpollutionpy import api


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


def boundBasedConcentration(df:pd.DataFrame) -> np.array:
    """
    Calculate the concentration based on a boundingbox system, with a receptor point in the center of each boundingbox.
    WARNING: the function has a static concentration of 10.

    Parameters:
        df (Pandas.DataFrame): Map data with nodes and ways.
        
    Returns:
        concentrationMatrix (list): a matrix with at each cell a particular concentration.
    """
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
                        ret = algo.concentration(10,conf.sim['wind_speed'],centerLon,centerLat,currNode['lon'],currNode['lat'],nextNode['lon'],nextNode['lat'],conf.sim['relative_wind_angle'],(lineLength/4))
                        matrix[rowIndex][columIndex] += ret

                # print(f'Amount of bounds calculated:{i}')
                i+=1
    print('J:',j)
    print(f'1 cycle of calculations took {time.time() - startTime} seconds.')
    return matrix


def emissionfactorToEmission(distance, EF):
    """
    Calculates the emmissions per unit length

    Parameters:
        distance (int): A string of coordinates split by a space, order from bottom,left,top,right.
        EF (int): emission factor, a factor to determine the emissions this time represented as grams per distance (kilometer)
        
    Returns:
        emissions (int): The emissions per unit length.
    """
    return distance * EF


def emissionfactorToReducedEmission(distance, EF, ER):
    """
    Calculates the emmissions per unit length

    Parameters:
        distance (int): A string of coordinates split by a space, order from bottom,left,top,right.
        EF (int): emission factor, a factor to determine the emissions this time represented as grams per distance (kilometer).
        ER (int): emission reduction factor.
        
    Returns:
        emissions (int): The emissions per unit length after emission reduction.
    """
    return emissionfactorToEmission(distance,EF)*(1-ER/100)


def showWindAngleImpact(Q,U,Xr,Yr,X1,Y1,X2,Y2):
    """
    Shows the imact wind angles have

    Parameters:
        Q (int): The amount of mass per unit length in time
        U (int): Windspeed in unit length per time
        Xr (int): Receptor point X coordinate
        Yr (int): Receptor point Y coordinate
        X1 (int): Start of the line source X coordinate
        Y1 (int): Start of the line source Y coordinate
        X2 (int): End of the line source X coordinate
        Y2 (int): End of the line source Y coordinate
        
    Returns:
        None: Shows a figure of the impact of the wind angle.
    """

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
        # print(algo.concentration(Q,U,Xr,Yr,X1,Y1,X2,Y2,math.radians(i)))
        x.append(i)
        y.append(algo.concentration(Q,U,Xr,Yr,X1,Y1,X2,Y2,math.radians(i),0))

    plt.plot(x,y)
    plt.show()


def calculateDistanceKm(lat1, lat2, lon1, lon2):
    """
    Calculate the distance between two longitude and latitude points in kilometers.

    Parameters:
        lat1 (float): latitude line of the fist point.
        lat2 (float): latitude line of the second point.
        lon1 (float): longitude line of the fist point.
        lon2 (float): longitude line of the second point.
        
    Returns:
        distance (int): The distance between the two points in kilometers.
    """
    p = math.pi/180
    a = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p) * math.cos(lat2*p) * (1-math.cos((lon2-lon1)*p))/2
    return 12742 * math.asin(math.sqrt(a))


def calculateDistanceM(lat1, lat2, lon1, lon2):
    """
    Calculate the distance between two longitude and latitude points in meters.

    Parameters:
        lat1 (float): latitude line of the fist point.
        lat2 (float): latitude line of the second point.
        lon1 (float): longitude line of the fist point.
        lon2 (float): longitude line of the second point.
        
    Returns:
        distance (int): The distance between the two points in meters.
    """
    p = math.pi/180
    a = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p) * math.cos(lat2*p) * (1-math.cos((lon2-lon1)*p))/2
    if a == 0:
        return 1
    return 12742000 * math.asin(math.sqrt(a))


def getMinMaxDataframe(df:pd.DataFrame) -> tuple:
    """
    Get the outer values of longitude and latitude form a dataframe.

    Parameters:
        df (Pandas.DataFrame): Map data with nodes and ways.
        
    Returns:
        outer (tuple): min longitude, max longitude, min latitude, max latitude.
    """
    return (min(df['lon']),
    max(df['lon']),
    min(df['lat']),
    max(df['lat']))
    

def getBoundsRanges(matrixShape,lon_min,lon_max,lat_min,lat_max):
    """
    Create a 2D frame of a predetermined size with longitude & latitude bounds as values.

    Parameters:
        matrixShape (tuple): a 2 long tuple with row and column count.
        lon_min (float): start of the boundingbox as longitude
        lon_max (float): end of the boundingbox as longitude
        lat_min (float): start of the boundingbox as latitude
        lat_max (float): end of the boundingbox as latitude
        
    Returns:
        bounds (3D list[x,x,2,2]): a matrix with for each cell 2 rows of min and max values of that particulair bound.
    """
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
    """
    Create a 3D frame of a predetermined size with all the nodes in those bounds from a dataframe, with a predetermined shape.

    Parameters:
        df (Pandas.DataFrame): Map data with nodes and ways.
        matrixShape (tuple): a 2 long tuple with row and column count.
        lon_min (float): start of the boundingbox as longitude
        lon_max (float): end of the boundingbox as longitude
        lat_min (float): start of the boundingbox as latitude
        lat_max (float): end of the boundingbox as latitude
        
    Returns:
        bounds (3D list[x,y,z]): a matrix with a particular size with per cell a list of nodes that belong there.
    """
    boundsRange = getBoundsRanges(matrixShape,lon_min,lon_max,lat_min,lat_max)
    print(boundsRange, boundsRange.shape)
    nodesInBounds = []
    df.sort_values(by=['lat','lon'],inplace=True) # WIP
    df.reset_index(inplace=True) # WIP

    for rowIndex in range(0, matrixShape[0]):
        nodesInBounds.append([]) # Add a new row
        row = boundsRange[rowIndex][0][0]
        minLat = df.index[df['lat'] >= row[0]].min()
        maxLat = df.index[df['lat'] < row[1]].max()
        dfHorRow = df.iloc[minLat-1:maxLat+1]
        # df.drop(list(range(minLat, maxLat+1))) makes it slower

        for columIndex in range(0, matrixShape[1]):
            nodesInBounds[rowIndex].append([]) # Add a column to the row
            nodes = getNodesInBoundsFromDf(dfHorRow,boundsRange[rowIndex][columIndex])
            x = list(nodes.to_dict('records'))
            # print(minLat,maxLat,boundsRange[rowIndex][columIndex][1])
            # print(x)
            nodesInBounds[rowIndex][columIndex] = x
            
    return np.array(nodesInBounds,dtype=object)


def getNodesInBoundsFromDf(_df:pd.DataFrame,bounds:list):
    """
    Search for all the nodes that belong between 4 bounding lines.

    Parameters:
        df (Pandas.DataFrame): Map data with nodes and ways.
        bounds (list): 2 rows with each 2 values, row1 latitudes(min-max), row2 longitudes(min-max).
        
    Returns:
        nodesInBound(Series): A series object with all the nodes an their atributes.
    """
    # bounds[0] = [start latitude, end latitude line]
    # bounds[1] = [start longitude, end longitude]

    return _df[(_df['lon'] >= bounds[1][0]) & (_df['lon'] < bounds[1][1])][['id','lat','lon']]


def generateLookup(dfWays:pd.DataFrame):
    """
    Creates a lookup table where the key is a nodeId's and as value another dict with the corrosponding waysId's and order number.

    Parameters:
        dfWays (Pandas.DataFrame): Map data with ways.
        
    Returns:
        nodeLookup(dict): A dictionary with all nodeId's as keys and as value another dict with the corrosponding waysId's and order number.
    """

    nodeLookup = {}
    for way in dfWays.itertuples():
        # input(f'{way.nodes}, {type(way.nodes)}')
        for node in way.nodes:
            if node in nodeLookup:
                nodeLookup[node].update({way.id:way.nodes.index(node)})
            else:
                nodeLookup[node] = {way.id:way.nodes.index(node)}
    return nodeLookup


def waytypeToSpeed(roadTypeName:str):
    """
    A fallback function to determine typical speeds for roads that have no meta-data.

    Parameters:
        roadTypeName (str): The type of road, usually specified under the tags.highway/highway column.
        
    Returns:
        speed (int): returns a set or fallback speed for a particular road type.
    """
    if roadTypeName in conf.sim['roads']['speeds']:
        return conf.sim['roads']['speeds'][roadTypeName]
    else:
        return 30


def effMoped():
    """
    Get the effectiveness factor for mopeds in grams per kilometer.
        
    Returns:
        effectiveness factor (int): the effectiveness factor determined by the EU.
    """
    # Table 8-30: Euro 3
    return 0.26


def effPassenger(speed):
    """
    Calculate the effectiveness factor for passenger cars in grams per kilometer depending on the speed.

    Parameter:
        speed (int): the speed of the vehicle
        
    Returns:
        effectiveness factor (int): the effectiveness factor determined by the EU.
    """
    # Table 8-5, ECE 15-04, CC > 2.0 l, gasoline 
    return 2.427 - 0.014 * speed + 0.000266*math.pow(speed,2)


def effLightduty(speed):
    """
    Calculate the effectiveness factor for light duty vehicles <3.5 t in grams per kilometer depending on the speed.

    Parameter:
        speed (int): the speed of the vehicle
        
    Returns:
        effectiveness factor (int): the effectiveness factor determined by the EU.
    """
    # Table 8-24, conventional, gasoline vehicles <3.5 t
    return 0.0179 * speed + 1.9547


def effHeavyduty(speed):
    """
    Calculate the effectiveness factor for heavy duty vehicles <3.5 t in grams per kilometer depending on the speed.

    Parameter:
        speed (int): the speed of the vehicle
        
    Returns:
        effectiveness factor (int): the effectiveness factor determined by the EU.
    """
    # Table 8-28, gasoline vehicles >3.5 t
    if(speed < 60):
        return 4.5 # urban
    else:
        return 7.5 # Highway


def generateWayEF(df:pd.DataFrame):
    """
    Compose the emmisionfactor depending on the speed and composition of cars and bussyness of the particular road.

    Parameter:
        df (Pandas.DataFrame): Map data with nodes and ways.
        
    Returns:
        waysToFactor (dict): Returns a dictionary with the emission factor and bussyness offset.
    """
    unhandled = set()
    wayLookup = {}

    df['tags'] = df['tags'].apply(eval)
    
    for way in df.itertuples():
        _type = str(way.tags['highway'])
        if conf.sim['overwriting']['enabled'] and _type in conf.sim['overwriting']['roads_speeds']:
            speed = conf.sim['overwriting']['roads_speeds'][_type]
        elif(way.tags.get('maxspeed')):
            speed = int(way.tags['maxspeed'])
        else:
            speed = waytypeToSpeed(_type)

        if _type == 'service' or _type == 'services':
            eff, busy = (effPassenger(speed)*0.9 + effLightduty(speed)*0.1), 1
        elif _type == 'cycleway':
            eff, busy = effMoped(), 1
        elif _type == 'pedestrian' or _type == 'footway' or _type == 'path':
            eff, busy = effMoped() * 0.1 , 1
        elif _type == 'motorway_link' or _type == 'motorway' or _type == 'highway' or _type == 'speedway':
            eff, busy = (effPassenger(speed)*0.55 + effLightduty(speed)*0.2 + effHeavyduty(speed)*0.25), 10
        elif _type == 'primary' or  _type == 'primary_link':
            eff, busy = (effPassenger(speed)*0.6 + effLightduty(speed)*0.2 + effHeavyduty(speed)*0.2), 8
        elif _type == 'secondary' or  _type == 'secondary_link':
            eff, busy = (effPassenger(speed)*0.7 + effLightduty(speed)*0.15 + effHeavyduty(speed)*0.15), 5
        elif _type == 'tertiary' or  _type == 'tertiary_link':
            eff, busy = (effPassenger(speed)*0.8 + effLightduty(speed)*0.1 + effHeavyduty(speed)*0.1), 3
        elif _type == 'residential' or _type == 'living_street':
            eff, busy = (effPassenger(speed)*0.9 + effLightduty(speed)*0.05 + effMoped() * 0.05), 1
        else:
            unhandled.add(_type) # way unknown
            eff, busy = (effPassenger(speed)*0.8 + effLightduty(speed)*0.1 + effHeavyduty(speed)*0.1), 2

        wayLookup[way.id] = {'eff':eff,'busy':busy}
    print(f'Warning, unhandled ways:{unhandled}')
    return wayLookup


def generateIndexListFromCircumference(coords:list,shape:tuple) -> np.ndarray: 
    """
    Create a list of indexes from a list of coordinates form a shape with a start and en end every row.

    Parameter:
        coords (list): A list of indexes with a shape recognizable per row.
        
    Returns:
        outList (list): Returns a list with the complete surface.
    """
    rows = shape[0]
    cols = shape[1]

    startRow = min(coords,key=itemgetter(0))[0]
    endRow = max(coords,key=itemgetter(0))[0]
    startCol = min(coords,key=itemgetter(1))[1]
    endCol = max(coords,key=itemgetter(1))[1]

    if startRow < 0:
        startRow = 0
    if startCol < 0:
        startCol = 0
    if endRow >= rows:
        endRow = rows-1
    if endCol >= cols:
        endCol = cols-1

    outList = []
    inside = False
    for rowIndex in range(startRow,endRow+1):
        inside = False
        for colIndex in range(startCol,endCol+1):

            if((rowIndex,colIndex) in coords):
                inside = not inside
                outList.append([rowIndex,colIndex])
                continue

            if(inside):
                outList.append([rowIndex,colIndex])

    return np.asarray(outList)


def getNodesInBoundsByIndex(nodesInBounds,coords):
    """
    Create a list of lists with nodes.

    Parameter:
        nodesInBounds (list): A 3D matix with in each cell a list of nodes
        coords (list): A list of list consiting of 2 values (row, col) the matrix coordinates.
        
    Returns:
        outList (list): Returns a list with the requested surface.
    """
    rows = coords[:,0]
    cols = coords[:,1]
    return nodesInBounds[[rows,cols]] # REFACTOR


def generateCirleCoordsList(r,start:tuple):
    """
    Create a list of perimiter coordinates for a circle given a radius and a starting point. 

    Parameter:
        r (int): The radius in coordinates.
        start (tuple): an x & y starting value.
        
    Returns:
        pointlist (list): Returns a list with perimiter of a circle.
    """
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


def compareAngles(roadAngle):
    if (conf.sim["actual_wind_angle"] >= 0 and conf.sim["actual_wind_angle"] < 180):
        # return conf.sim["actual_wind_angle"] + roadAngle
        return (roadAngle - conf.sim["actual_wind_angle"])
    elif (conf.sim["actual_wind_angle"] >= 180 and conf.sim["actual_wind_angle"] < 360):
        Exception
        return conf.sim["actual_wind_angle"] + roadAngle


def nodesToAngle(node1, node2):
    opp = node1['lat'] - node2['lat']
    adj = node1['lon'] - node2['lon']
    if (adj == 0):
        adj = 0.000001
    diff = opp/adj
    return math.degrees(math.atan(diff))


def receptorpointBasedConcentration(dfNodes:pd.DataFrame,dfWays:pd.DataFrame,windSpeed:int,windAngle:int,radius:int,bboxSize:int = 100) -> np.array:
    """
    Calculate the concentration based on the receptorpoints, taking in account different regions based on the radius.

    Parameters:
        df (Pandas.DataFrame): Map data with nodes and ways.
        windSpeed (int): The windspeed in units per time. (meters/second)
        windAngle (int): The windangle form -89 to 90.
        radius (int): The radius aroung each receptor point to take in account. (meters)
        bboxSize (int): The size of the bounding boxes in height and width. Standard 100 (meters)
        
    Returns:
        concentrationMatrix (list): a matrix with at each cell a particular concentration.
    """

    """
    a not working! /maybe/ better average downwind distance
    averageDownwind = calculateDistanceM(
        centerLat, (float(currNode['lat'])+float(nextNode['lat']))/2,
        centerLon,(float(currNode['lon'])+float(nextNode['lon']))/2)
    """

    startTime = time.time()
    total = 0
    downWindFrac = conf.sim['downwind']

    dfWays['nodes'] = dfWays['nodes'].apply(eval)

    lon_min, lon_max, lat_min ,lat_max = getMinMaxDataframe(dfNodes)

    lonInKm = calculateDistanceM(lat_min,lat_min,lon_min,lon_max)
    latInKm = calculateDistanceM(lat_min,lat_max,lon_min,lon_min)
    
    if(conf.sim['verbose']):
        print(lon_min, lon_max, lat_min ,lat_max)
        print('lonInKm',lonInKm,'latInKm',latInKm)
        print('lon/100 ceil',math.ceil(lonInKm/bboxSize),'lat/100 ceil',math.ceil(latInKm/bboxSize))

    concentrationMatrix = np.zeros(
        (math.ceil(latInKm/bboxSize),math.ceil(lonInKm/bboxSize))
    )
    nodeToWays = generateLookup(dfWays) # REFACTOR (still fast)
    wayIdToInfo = generateWayEF(dfWays) # REFACTOR (still fast)

    # REFACTOR this slow mess vv
    print(concentrationMatrix.shape)
    nodesInBounds = getBoundsNodelist(dfNodes,concentrationMatrix.shape,lon_min,lon_max,lat_min,lat_max)
    latFreq = ((lat_max - lat_min) / concentrationMatrix.shape[0])
    lonFreq = ((lon_max - lon_min) / concentrationMatrix.shape[1])

    if(conf.sim['verbose']):
        print(f'Prep took: {time.time() - startTime}s')
    # return None
    
    # depending on the size of the bounds you make the radius has a different impact as multiplier
    for rowIndex in range(0, len(concentrationMatrix)):
        centerLat = lat_max - (latFreq * rowIndex)
        for colIndex in range(0, len(concentrationMatrix[rowIndex])):
            total +=1
            centerLon = lon_min + (lonFreq * colIndex)
            circumference = generateCirleCoordsList(int(radius/bboxSize)-1,(rowIndex,colIndex))
            areaList = generateIndexListFromCircumference(circumference,nodesInBounds.shape)
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
                    ret = algo.concentration(emission,windSpeed,centerLon,centerLat,currNode['lon'],currNode['lat'],nextNode['lon'],nextNode['lat'],windAngle,(lineLength*downWindFrac))
                    concentrationMatrix[rowIndex][colIndex] += ret

            # print(f'Receptor {total}/{concentrationMatrix.size}')
    if(conf.sim['verbose']):
        print(f'Full setup and 1 cycle of concentration took: {time.time() - startTime}s')
    return concentrationMatrix

if __name__ == "__main__":
    name = conf.sim["current"]
    
    windSpeed = conf.sim["wind_speed"]
    windAngle = conf.sim["relative_wind_angle"]
    bboxSize = conf.sim["bbox_size"]
    radius = conf.sim["radius"]

    nodes = api.readFromFile(name+'_node')
    ways = api.readFromFile(name+'_way')
    concentrationMatrix = receptorpointBasedConcentration(nodes,ways,windSpeed,windAngle,radius,bboxSize)
    draw.imagePlot(concentrationMatrix,name,bboxSize,radius)