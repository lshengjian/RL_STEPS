import numpy as np
from gymnasium.utils import seeding
from rlbase.grid.data import MAPS
from rlbase import World
def main():
    desc = np.asarray(MAPS['2x2'], dtype="c")
    world=World('human',desc,False,True)
    rand_generator=seeding.np_random(1234)[0]
    world.reset(rand_generator)
    for _ in range(1000):
        world.update()

if __name__ == "__main__":
    main()
    