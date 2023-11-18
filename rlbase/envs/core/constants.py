from __future__ import annotations

import numpy as np

from .define import *

NUM_ACTIONS = Action.LEFT+1
ACTION_FLAGS = ["o", "↑", "→", "↓", "←"]

DIR_VECTOR = [
    np.array((0., 0.)),
    np.array((0., -1.)),
    np.array((1., 0.)),
    np.array((0., 1.)),
    np.array((-1., 0.)),
]
COLORS = {
    '-': (255, 255, 255),
    'S': (184, 203, 184),
    'X': (207, 199, 248),
    'G': (0, 255, 255, 66),
}

AGENT_COLOR = (207, 99, 48)
VISITED_COLOR = (47, 39, 68)
POLICY_COLOR = (47, 239, 38)
TEXT_COLOR = (0, 0, 0)

IN_GOAL = 1
IN_FORBIDDEN = -1
OUT_BOUND = -1

MAPS = {
    "1x3": ["-SG"],
    "2x2": ["SX", "-G"],
    "4x4": ["S---", "-X-X", "---X", "X--G"],
    "8x8": [
        "S-------",
        "--------",
        "---X----",
        "-----X--",
        "---X----",
        "--X---X-",
        "-X--X-X-",
        "---X---G"
    ]
}
