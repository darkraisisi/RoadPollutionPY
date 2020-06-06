import math
import numpy as np

"""
Source : https://www.sciencedirect.com/science/article/abs/pii/S1352231000000741
C is ht concentration
Q1 is the source strength per unit length
u is the average wind speed
theta is the angle between the wind direction and the road
x,y,z are the coordinates(road?)
H is the effective source height,
p is the half-length of the line source
erf is the error function
sigZ and sigY are the vertival and lateral dispersion parameters
"""


def err(p,y,theta,x,sigY):
    return errMin(p,y,theta,x,sigY) + errPls(p,y,theta,x,sigY)


def errMin(p,y,theta,x,sigY):
    return ((math.sin(theta)*(p-y) - x*math.cos(theta))
    /math.sqrt(2*sigY))


def errPls(p,y,theta,x,sigY):
    return ((math.sin(theta)*(p+y)+x*math.cos(theta))
    /math.sqrt(2*sigY))


def exp(z,H,sigZ):
    return expMin(z,H,sigZ) + expPls(z,H,sigZ)

def expMin(z,H,sigZ):
    return -(math.pow(z-H,2)/2*math.pow(sigZ,2))

def expPls(z,H,sigZ):
    return -(math.pow(z+H,2)/2*math.pow(sigZ,2))


def concentration(Q1,sigZ,u,theta):
    return (Q1
    /2*math.sqrt(2*math.pi)*sigZ*u*math.sin(theta))

Q1 = 120
u = 2
theta = 180
x,y,z = 10, 10, 1
H = 2
p = ((x+y)/2)
sigZ = 10
sigY = 8


print(concentration(Q1,sigZ,u,theta),exp(z,H,sigZ),err(p,y,theta,x,sigY))
print(concentration(Q1,sigZ,u,theta)*exp(z,H,sigZ)*err(p,y,theta,x,sigY))