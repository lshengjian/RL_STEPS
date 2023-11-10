import  sys
import numpy as np
from os import path
dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)

from  mygrid.config import *
from  mygrid.world import MyWorld

def test_rule1():
    desc=MAPS['2x2']
    desc = np.asarray(desc, dtype="c")
    grid=MyWorld(False,desc)
    ps=grid.P[0]

    #  0,1     S  H
    #  2,3     F  G
    assert ps[LEFT]==[(1.0, 0, -1, False)] # 概率，状态，奖励，结束
    assert ps[DOWN]==[(1.0, 2, 0, False)]
    assert ps[RIGHT]==[(1.0, 1, -1, False)]
    assert ps[UP]==[(1.0, 0, -1, False)]

def test_rule2():
    desc=MAPS['2x2']
    desc = np.asarray(desc, dtype="c")
    grid=MyWorld(True,desc)
    ps=grid.P[0]
    assert ps[STAY]==[(1.0, 0, 0, False)]
    assert ps[RIGHT]==[(0.1, 2, 0, False), (0.8, 1, -1, False), (0.1, 0, -1, False)]
    ps=grid.P[3]
    assert ps[STAY]==[(1.0, 3, 1, True)]
    assert ps[LEFT]==[(1.0, 2, 0, False)]
    assert ps[DOWN]==[(1.0, 3, -1, True)]
    assert ps[RIGHT]==[(1.0, 3, -1, True)]
    assert ps[UP]==[(1.0, 1, -1, False)]


