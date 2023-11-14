#from typing import List, Optional
import numpy as np
import esper
from ..utils.sample import categorical_sample
from .data import *
from .processors import RenderSystem


class World:

    def __init__(self, render_mode, desc: np.ndarray,show_stat_info=False):#,autoPolicy=True, isManualControl=False
        
        self.desc = desc
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.nA = nA = NUM_ACTIONS
        self.nS = nS = nrow * ncol
        G.GRID_SIZE = (nrow, ncol)
        self.state = 0
        # self.lastaction = None
        self.initial_state_distrib = np.array(
            desc == b'S').astype("float32").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}  # 动作转移矩阵
        for row in range(nrow):
            for col in range(ncol):
                ent = esper.create_entity()
                CACHE[(row, col)] = ent
                flag = desc[row][col].decode()
                tile = Tile(row, col, flag)
                esper.add_component(ent, tile)

                s = tile.state
                info = StatInfo()
                esper.add_component(ent, info)
                for a in range(nA):
                    li = self.P[s][a]
                    li.append((1.0, *self._try_move(row, col, a)))


        self.rederer = RenderSystem(render_mode,show_stat_info)
        esper.add_processor(self.rederer)



    def update(self):
        esper.process()

    def reset(self, np_random):
        self.np_random = np_random
        data=esper.get_component(Agent)
        if len(data)>0:
            e,_=data[0]
            esper.remove_component(e,Agent)
        self.state = categorical_sample(self.initial_state_distrib, np_random)
        r, c = self.state2idx(self.state)
        ent = CACHE[(r, c)]
        t = esper.component_for_entity(ent, Tile)
        esper.add_component(ent,Agent(t))
        #todo reset processors
        self.update()

    def move(self, action: Action):
        s1 = self.state
        transitions = self.P[s1][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, terminated = transitions[i]
        self.state = s
        self.lastaction = action
        s2 = s
        r1, c1 = self.state2idx(s1)
        r2, c2 = self.state2idx(s2)
        # t1=esper.component_for_entity(CACHE[r1,c1],Tile)
        t2 = esper.component_for_entity(CACHE[r2, c2], Tile)
         
        agent = esper.component_for_entity(CACHE[r1, c1], Agent)
        esper.remove_component(CACHE[r1, c1], Agent)
        agent.move(action, t2, r)
        esper.add_component(CACHE[r2, c2], agent)
        self.update()
        # esper.dispatch_event('agent_moved',CACHE[s],action,CACHE[self.state2idx(s2)],r)
        return s, r, terminated



    def idx2state(self, row, col):
        return row*self.ncol+col

    def state2idx(self, state):
        return state//self.ncol, state % self.ncol

    def _try_move(self, row: int, col: int, action: int):
        ok, newrow, newcol = self._check_next_pos(row, col, action)
        newstate = self.idx2state(newrow, newcol)
        newletter = self.desc[newrow, newcol]
        terminated = bytes(newletter) in b"G"
        reward = 0
        if not ok:
            reward = OUT_BOUND
        elif newletter == b"G":
            reward = IN_GOAL
        elif newletter == b"X":
            reward = IN_FORBIDDEN
        return newstate, reward, terminated

    def _check_next_pos(self, row, col, a):
        canPass = True
        if a == Action.LEFT:
            if col - 1 < 0:
                canPass = False
            col = max(col - 1, 0)
        elif a == Action.RIGHT:
            if col + 1 > self.ncol - 1:
                canPass = False
            col = min(col + 1, self.ncol - 1)
        elif a == Action.DOWN:
            if row + 1 > self.nrow - 1:
                canPass = False
            row = min(row + 1, self.nrow - 1)
        elif a == Action.UP:
            if row - 1 < 0:
                canPass = False
            row = max(row - 1, 0)
        return canPass, row, col
