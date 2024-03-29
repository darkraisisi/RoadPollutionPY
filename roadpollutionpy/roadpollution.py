import api
import config as conf
import draw
import poll


def simulateCurrentConcentration():
    """
    Will show just the concentration.

    Prepare and run all the steps requird to do a simulation, dictated by the set config variables.

    Parameters:
        name (str): conf.sim["current"], name of your file.
        windSpeed (int): conf.sim["wind_speed"], windspeed in unit distance.
        windAngle (int): conf.sim["relative_wind_angle"], relative windangle from -89 to 90.
        bboxSize (int): conf.sim["bbox_size"], Bounding box size in unit length.
        radius (int): conf.sim["radius"], radius aroung a receptor in unit length.

    Returns:
        None: Shows the user a figure of concentration.
    """
    name = conf.sim["current"]

    windSpeed = conf.sim["wind_speed"]
    windAngle = conf.sim["relative_wind_angle"]
    bboxSize = conf.sim["bbox_size"]
    radius = conf.sim["radius"]

    nodes = api.readFromFile(name+'_node')
    ways = api.readFromFile(name+'_way')
    concentrationMatrix = poll.receptorpointBasedConcentration(nodes, ways, windSpeed, windAngle, radius, bboxSize)
    draw.imagePlot(concentrationMatrix, name, bboxSize, radius)


def simulateCurrentConcentrationAndRoads():
    """
    Will show the concentration and an overlay of roads.

    Prepare and run all the steps requird to do a simulation, dictated by the set config variables.

    Parameters:
        name (str): conf.sim["current"], name of your file.
        windSpeed (int): conf.sim["wind_speed"], windspeed in unit distance.
        windAngle (int): conf.sim["relative_wind_angle"], relative windangle from -89 to 90.
        bboxSize (int): conf.sim["bbox_size"], Bounding box size in unit length.
        radius (int): conf.sim["radius"], radius aroung a receptor in unit length.

    Returns:
        None: Shows the user a figure of concentration and an overlay of roads.
    """
    name = conf.sim["current"]

    windSpeed = conf.sim["wind_speed"]
    windAngle = conf.sim["relative_wind_angle"]
    bboxSize = conf.sim["bbox_size"]
    radius = conf.sim["radius"]
    roads = conf.draw["roads"]

    nodes = api.readFromFile(name+'_node')
    ways = api.readFromFile(name+'_way')
    concentrationMatrix = poll.receptorpointBasedConcentration(nodes,ways,windSpeed,windAngle,radius,bboxSize)
    draw.concentrationAndRoads(concentrationMatrix,nodes,ways,roads)

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
    path = conf.osm['path']+name

    res = api.getWayFromCoordinates(conf.osm["coordinates"][name])
    api.writeToCsv(path,res)


inputToFunction = {
    "1": simulateCurrentConcentration,
    "2": simulateCurrentConcentrationAndRoads,
    "3": downloadNew,
    "quit": quit
}

if __name__ == "__main__":
    while True:
        try:
            selection = str(
            input("\nwhich function would you like to perform?\n\n"+
            "1: Simulate current, radial bounds, concentration map\n"+
            "2: Simulate current, radial bounds, concentration map & roads\n"+
            "3: Download new map by current name \n"+
            "quit\n"))
            
            inputToFunction[selection]()

        except Exception as x:
            print(f"An error occured.\n{x} ")
