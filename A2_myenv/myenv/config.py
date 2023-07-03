LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAY = 4

GO_TO_GOAL=1
GO_TO_HOLE=-1
OUT_BOUND=-1

MAPS = {
    "2x2": ["SH", "FG"],
    "4x4": ["SFFS", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFS",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFSHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ]
}

FPS = 120