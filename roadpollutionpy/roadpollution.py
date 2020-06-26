import api
import config as conf
import draw
import poll


def simulateCurrentConcentration():
    """
    Prepare and run all the steps requird to do a simulation, dictated by the set config variables.
    Shows just the concentration.

    Parameters:
        name (str): conf.sim["current"], name of your file.
        windSpeed (int): conf.sim["wind_speed"], windspeed in unit distance.
        windAngle (int): conf.sim["wind_angle"], relative windangle from -89 to 90.
        bboxSize (int): conf.sim["bbox_size"], Bounding box size in unit length.
        radius (int): conf.sim["radius"], radius aroung a receptor in unit length.
        
    Returns:
        None: Shows the user a figure of concentration.
    """
    name = conf.sim["current"]
    
    windSpeed = conf.sim["wind_speed"]
    windAngle = conf.sim["wind_angle"]
    bboxSize = conf.sim["bbox_size"]
    radius = conf.sim["radius"]

    mapData = draw.readFromFile(name)
    concentrationMatrix = poll.receptorpointBasedConcentration(mapData,windSpeed,windAngle,radius,bboxSize)
    draw.imagePlot(concentrationMatrix,name,bboxSize,radius)


def simulateCurrentConcentrationAndRoads():
    """
    Prepare and run all the steps requird to do a simulation, dictated by the set config variables.
    Shows the concentration and an overlay of roads.

    Parameters:
        name (str): conf.sim["current"], name of your file.
        windSpeed (int): conf.sim["wind_speed"], windspeed in unit distance.
        windAngle (int): conf.sim["wind_angle"], relative windangle from -89 to 90.
        bboxSize (int): conf.sim["bbox_size"], Bounding box size in unit length.
        radius (int): conf.sim["radius"], radius aroung a receptor in unit length.
        
    Returns:
        None: Shows the user a figure of concentration and an overlay of roads.
    """
    name = conf.sim["current"]
    
    windSpeed = conf.sim["wind_speed"]
    windAngle = conf.sim["wind_angle"]
    bboxSize = conf.sim["bbox_size"]
    radius = conf.sim["radius"]
    roads = conf.draw["roads"]

    mapData = draw.readFromFile(name)
    concentrationMatrix = poll.receptorpointBasedConcentration(mapData,windSpeed,windAngle,radius,bboxSize)
    draw.concentrationAndRoads(concentrationMatrix,mapData,roads)


def downloadNormalizeNew():
    downloadNew()
    normalizeCurrent()



def downloadNew():
    """
    Download a new map and save it locally.

    Parameters:
        name (str): conf.sim["current"], name of your file.
        path (str): conf.osm['path']+name+conf.osm['extension']
        coordinates (int): conf.osm["coordinates"][name]
        
    Returns:
        None: Downloads and saves a map locally.
    """
    name = conf.sim["current"]
    fullPath = conf.osm['path']+name+conf.osm['extension']

    res = api.getWayFromCoordinates(conf.osm["coordinates"][name])
    api.writeToFile(fullPath,res)


def normalizeCurrent():
    """
    Normalize a currently saved map.

    Parameters:
        name (str): conf.sim["current"], name of your file.
        path (str): conf.osm['normalisation']['path']+name+conf.osm['extension'], path where you want to save your normalized file.
        collums (int): conf.osm['normalisation']['cols'], Columns to keep.
        
    Returns:
        None: Downloads and saves a map locally.
    """
    name = conf.sim["current"]
    fullPath = conf.osm['normalisation']['path']+name+conf.osm['extension']

    mapData = api.readFromFile(name)
    normDf = api.normalizeDataframe(mapData,conf.osm['normalisation']['cols'],True)
    api.dataframeToFile(normDf,fullPath)



inputToFunction = {
    "1": simulateCurrentConcentration,
    "2": simulateCurrentConcentrationAndRoads,
    "3": downloadNormalizeNew,
    "4": downloadNew,
    "5": normalizeCurrent,
    "quit": quit
}

if __name__ == "__main__":
    while True:
        try:
            selection = str(
            input("\nwhich function would you like to perform?\n\n"+
            "1: Simulate current, radial bounds, concentration map\n"+
            "2: Simulate current, radial bounds, concentration map & roads\n"+
            "3: Download & normalize new map by current name \n"+
            "4: Download new map by current name \n"+
            "5: Normalize map by current name \n"+
            "quit\n"))
            
            inputToFunction[selection]()

        except Exception as x:
            print(f"An error occured.\n{x} ")

