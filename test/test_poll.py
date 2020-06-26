import os
import sys
import math
import pandas as pd

import pytest

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)).replace('\\test',''))
from roadpollutionpy import poll
from roadpollutionpy import draw


def test_EFtoEM():
    distance = 12
    EF = 0.125
    assert poll.emissionfactorToEmission(distance,EF) == 1.5


def test_calculateDistanceKm():
    lat1, lat2 = 52.0867, 52.0867
    lon1, lon2 = 5.1122, 5.14145
    assert (poll.calculateDistanceKm(lat1, lat2, lon1, lon2) - 2) < 0.001


def test_calculateDistanceM():
    lat1, lat2 = 52.0867, 52.0867
    lon1, lon2 = 5.1122, 5.14145
    assert (poll.calculateDistanceM(lat1, lat2, lon1, lon2) - 2000) < 0.001

"""
def test_getBoundsNodelist():
    lat1, lat2 = 52.0867, 52.0840
    lon1, lon2 = 5.1122, 5.14145
    matrixShape = 1,1
    nodeList = [{'id':1,'lon':52.0855,'lat':5.1125},{'id':2,'lon':52.0858,'lat':5.1128}]
    nodesInBounds = [[nodeList]]
    df = draw.readFromFile('')
    assert poll.getBoundsNodelist(df,(1,1),lat2,lat1,lon1,lon2) == nodesInBounds
"""
def test_waytypeToSpeed():
    assert poll.waytypeToSpeed('none') == 30
    assert poll.waytypeToSpeed('residential') == 30
    assert poll.waytypeToSpeed('motorway_link') >= 80


def test_generateWayEf():
    d = [{'id':1,'type':'way','maxspeed':30,'highway':'cycleway'},{'id':2,'type':'way','maxspeed':False,'highway':'residential'}]
    df = pd.DataFrame(d)
    # Generate a way eff for a way with speed and one without speed, lookup speed and apply
    assert poll.generateWayEF(df) == {1: {'busy': 1, 'eff': 0.26}, 2: {'busy': 1, 'eff': 2.159345}}


def test_generateCirleCoordsList():
    start = (5,5)
    assert poll.generateCirleCoordsList(1,start) == [(5, 6), (6, 5), (5, 4), (6, 6), (6, 4), (4, 6), (4, 4)]
    assert poll.generateCirleCoordsList(2,start) == [(5, 7), (7, 5), (5, 3), (6, 7), (6, 3), (4, 7), (4, 3), (7, 6), (7, 4), (3, 6), (3, 4)]


def test_generateIndexListFromCircumference():
    lst = [(1,5),(1,8)]
    assert poll.generateIndexListFromCircumference(lst) == [(1,5),(1,6),(1,7),(1,8)]
    lst = [ (1,5),(1,8),
            (3,6),(3,9)]
    assert poll.generateIndexListFromCircumference(lst) == [
        (1,5),(1,6),(1,7),(1,8),
        (3,6),(3,7),(3,8),(3,9)]