from ..data import G, NUM_ACTIONS, Tile,Agent,StatInfo,CACHE
import numpy as np
import esper
import copy


class RewardSystem(esper.Processor):
    def __init__(self):
        super().__init__()
        self.index=0

    def process(self):
        for _, agent in esper.get_component(Agent):
            data = agent.visited
            if len(data)<=self.index:
                return

            t1,a,t2,r=agent.visited[self.index]
            e1=CACHE[t1.row,t1.col]
            e2=CACHE[t2.row,t2.col]
            info1=esper.component_for_entity(e1,StatInfo)
            info2=esper.component_for_entity(e2,StatInfo)
            info1.Qs[a]=r+G.GAMMA*info2.V
            if info1.V<info1.Qs[a]:
                info1.V=info1.Qs[a] #todo
            self.index+=1

