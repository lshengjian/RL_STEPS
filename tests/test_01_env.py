
import sys
import pytest

from os import path
dir = path.abspath(path.dirname(__file__) + './..')
sys.path.append(str(dir))
from rlbase.grid.data import Tile, G, Action

def test_data():
    t = Tile(1, 1)
    w, h = G.TILE_SIZE
    assert t.position == (w+w//2, h+h//2)
    assert t.side(Action.STAY) ==  (72, 72)
    assert t.side(Action.UP) == (72, 72-h//3)
    assert t.side(Action.DOWN) == (72, 72+h//3)
    assert t.side(Action.LEFT) == (72-h//3, 72)
    assert t.side(Action.RIGHT) == (72+h//3, 72)



if __name__ == '__main__':
    pytest.main(["-s", f"{__file__}"])
