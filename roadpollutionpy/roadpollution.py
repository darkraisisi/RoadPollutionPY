import api
import config as conf
import draw
import poll


def simulateCurrentConcentration():
    name = conf.sim["current"]
    
    windSpeed = conf.sim["wind_speed"]
    windAngle = conf.sim["wind_angle"]
    bboxSize = conf.sim["bbox_size"]
    radius = conf.sim["radius"]

    mapData = draw.readFromFile(name)
    concentrationMatrix = poll.receptorpointBasedConcentration(mapData,windSpeed,windAngle,radius,bboxSize)
    draw.imagePlot(concentrationMatrix,name,bboxSize,radius)


def simulateCurrentConcentrationAndRoads():
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
    name = conf.sim["current"]
    fullPath = conf.osm['path']+name+conf.osm['extension']

    res = api.getWayFromCoordinates(conf.osm["coordinates"][name])
    api.writeToFile(fullPath,res)


def normalizeCurrent():
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


    