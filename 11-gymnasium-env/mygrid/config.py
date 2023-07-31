STAY =  0
UP =    1
RIGHT = 2
DOWN =  3
LEFT =  4

ACT_DIRS={
   LEFT:  (-1,0),
   DOWN:  (0,1),
   RIGHT: (1,0),
   UP:    (0,-1)
}
ACT_NAMES={
   STAY:'STAY',
   LEFT:'LEFT',
   DOWN:'DOWN',
   RIGHT:'RIGHT',
   UP:'UP'
}
COLORS={
   '-': (0,20,230,128),
   'S': (55,202,43,128),
   'X': (11,22,33),
   'G': (255,123,23,128),
}
TILE_SIZE=(64,48)
IN_GOAL=1
IN_OBSTACLE=-1
OUT_BOUND=-1

MAPS = {
    "4x4": ["S---", "-X-X", "---X", "X--G"],
    "8x8": [
        "S------S",
        "--------",
        "---X----",
        "-----X--",
        "--SX----",
        "--X---X-",
        "-X--X-X-",
        "---X---G"
    ]
}

FPS = 4