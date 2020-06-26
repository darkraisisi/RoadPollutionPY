import os
import sys
import math

import pytest

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)).replace('\\test',''))
from roadpollutionpy import algorithm as algo


def test_concentration():
    # This test shows the behavior explained in the sources abount this model.
    Q = 10
    U = 3 # 3 m/s average wind speed
    # Xr, Yr = 1800, 9200
    Xr, Yr = 1200, 9800
    X1,Y1, X2,Y2 = 1000, 9000, 2000, 10000

    assert algo.concentration(Q,U,Xr,Yr,X1,Y1,X2,Y2,math.radians(-90),1) > 1000000
    assert algo.concentration(Q,U,Xr,Yr,X1,Y1,X2,Y2,math.radians(-89),1) > 1000
    assert algo.concentration(Q,U,Xr,Yr,X1,Y1,X2,Y2,math.radians(40),1) < 1000