import numpy as np
from gymnasium.utils import seeding
from rlbase.grid.data import MAPS,CACHE,StatInfo
from rlbase.grid.utils import greedy_select
from rlbase import MiniGrid
import esper
import time
def main():
    env=MiniGrid('human','4x4',True)
    
    s,_=env.reset(seed=time.time_ns())
    for _ in range(24):
        
        ent=CACHE[env.world.state2idx(s)]
        stat=esper.component_for_entity(ent,StatInfo)
        a=greedy_select(stat.Qs,env.np_random)
        s, _, terminated, _, info=env.step(a)
        if terminated:
            break

if __name__ == "__main__":
    main()
    