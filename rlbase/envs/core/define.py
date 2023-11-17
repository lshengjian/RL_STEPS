from __future__ import annotations
import numpy as np
from dataclasses import dataclass as component

from typing import TypeVar
from enum import IntEnum

TState = TypeVar('TState',int,np.ndarray)

@component
class Transition:
    s1: int = 0
    action: Action = 0
    s2: int = 0
    reward:float=0.0
    terminated:bool=False

'''
top
o-------->  x
|
|
↓ y 
bottom
'''

class Action(IntEnum):
    STAY = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4