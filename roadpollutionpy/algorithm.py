import math

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
    """
    Calculate the z sigmoid

    Parameters:
    x (int): the downwind length of a line source

    Returns:
    int: zsigmoid 
    """
    return 0.14*x*math.pow(1+0.0003*x,-0.5)


def calcSigY(x):
    """
    Calculate the y sigmoid

    Parameters:
        x (int): the downwind length of a line source

    Returns:
        int: y sigmoid
    """
    return 0.16*x*math.pow(1+0.0004*x,-0.5)


def limit(Xr,Yr,Xi,Yi,theta,sigY):
    """
    Calculate the limits for the error functions

    Parameters:
        Xr (int): X coordinate for the receptor
        Yr (int): Y coordinate for the receptor
        Xi (int): X coordinate for a position in the linesource
        Yi (int): Y coordinate for a position in the linesource
        theta  (radian): A radian relative to the angle of the wind between -89 and 90
        sigY  (int): the vertical dispersion parameter

    Returns:
        int: the limit to the error function
    """
    return (((Yr - Yi)*math.cos(theta)-(Xr - Xi)*math.sin(theta))/
    (math.sqrt(2*sigY)))


def concentration(Q,U,Xr,Yr,X1,Y1,X2,Y2,theta,downWindDistance):
    """
    Calculate the concentration of pollutants in the air based on a line source

    Parameter:
        Q (int): The amount of mass per unit length in time
        U (int): Windspeed in unit length per time
        Xr (int): Receptor point X coordinate
        Yr (int): Receptor point Y coordinate
        X1 (int): Start of the line source X coordinate
        Y1 (int): Start of the line source Y coordinate
        X2 (int): End of the line source X coordinate
        Y2 (int): End of the line source Y coordinate
        theta (int): A radian relative to the angle of the wind between -89 and 90
        downWindDistance (int): The distance in unit length of the linesource that is downwind

    Returns:
        int: the concentration of a pollutant in the air
    """
    sigZ = calcSigZ(downWindDistance)
    sigY = calcSigY(downWindDistance)

    u1 = limit(Xr,Yr,X1,Y1,theta,sigY)
    u2 = limit(Xr,Yr,X2,Y2,theta,sigY)
    
    return ((Q / math.sqrt(2*math.pi)) * 
    (1 / (U*math.cos(theta)*sigZ)) *
    (math.erf(u1)-math.erf(u2)))