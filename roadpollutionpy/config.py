# Openstreetmap settings
osm = {
    "url": "https://overpass-api.de/",
    "api": "api/",
    "coordinates" : { # Boundingbox region
        # bottom,left,top,right
        "baarn": "52.1983,5.2588,52.2221,5.3098",
        "baarnPlus": "52.1983,5.2488,52.2271,5.3098",
        "utrechtProv": "51.9383,4.7969,52.3076,5.6332",
        "utrechtSur": "52.0415,4.9871,52.1554,5.1958"
    },
    "path": "maps/",
    "extension": ".json",
    "normalisation":{
        "cols":["type", "id", "lat", "lon", "nodes", "tags.name", "tags.highway", "tags.maxspeed", "tags.surface"],
        "path": "maps/norm/"
    }
}

# Simmulation settings
sim = {
    "current": "baarn",
    "verbose": False,
    "bbox_size": 50, # Meters.
    "radius": 300, # Meters.
    "wind_angle": 40, # Keep this parameter between -89 and 89 as it is relative to the road.
    "wind_speed": 3, # Meters/second.
    "downwind": 0.25, # Hardcoded fraction, part of the road that is down wind from the receptor point.
    "roads": {
        "speeds": {
            "service": 30,
            "cycleway": 50,
            "pedestrian": 1,
            "footway": 1,
            "path": 1,
            "motorway_link": 80,
            "primary": 80,
            "primary_link": 80,
            "secondary": 50,
            "secondary_link": 50,
            "tertiary": 30,
            "tertiary_link": 30,
            "residential": 30,
            "living_street": 30,
            "unclassified": 30
        }
    },
    "overwriting": { # Options for overwriting metadata to see the effect of changing the speed on pollution 
        "enabled": False,
        "roads_speeds":{
            "motorway": 130,
            "residential": 15
        }
    }
}

# Drawing related settings.
draw = { 
    "verbose": False,
    "roads": None, # List of roadTypeNames to draw.
    "path":osm["normalisation"]["path"]+sim["current"]+osm["extension"]
}