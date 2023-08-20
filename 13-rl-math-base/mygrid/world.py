from typing import List, Optional
import numpy as np
from gymnasium.utils import seeding
from .utils import categorical_sample
from .config import *

class World:
    def __init__(self,desc:np.ndarray):
        
        self.desc=desc
        
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.nA = nA = 5
        self.nS = nS = nrow * ncol
        self.state=0
        self.lastaction=None

        self.initial_state_distrib = np.array(desc == b'S').astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}#动作转移概率
        for row in range(nrow):
            for col in range(ncol):
                s =self.idx2state(row, col)
                for a in range(nA):
                    li = self.P[s][a]
                    li.append((1.0, *self.try_move(row, col, a)))

    def reset(self,np_random=None):
        self.state =  categorical_sample(self.initial_state_distrib, np_random)

    def move(self, action:int,np_random=None):
        transitions = self.P[self.state][action]
        i = categorical_sample([t[0] for t in transitions], np_random)
        p, s, r,terminated = transitions[i]
        self.state=s
        self.lastaction=action
        return s, r,terminated



    def try_move(self,row:int, col:int, action:int):
        ok,newrow, newcol = self.next(row, col, action)
        newstate =self.idx2state(newrow, newcol)
        newletter = self.desc[newrow, newcol]
        terminated = bytes(newletter) in b"G"
        reward=0
        if not ok:
            reward=OUT_BOUND
        elif newletter == b"G":
            reward = IN_GOAL
        elif newletter == b"X":
            reward = IN_FORBIDDEN
        return newstate, reward,terminated

    def get_vec(self):
        r,c=self.state2idx(self.state)
        r+=1
        c+=1
        x,y=c,r
        return 1,x,y,x**2,y**2,x*y
    def get_vec_state(self,state):
        r,c=self.state2idx(state)
        r+=1
        c+=1
        x,y=c/self.ncol,r/self.nrow
        return 1,x,y,x**2,y**2,x*y

    def idx2state(self,row, col):
        return row*self.ncol+col
    
    def state2idx(self,state):
        return state//self.ncol,state%self.ncol
    
    def next(self,row, col, a):
        canPass=True
        if a == LEFT:
            if col - 1< 0:canPass=False
            col = max(col - 1, 0)
        elif a == DOWN:
            if row + 1> self.nrow - 1:
                canPass=False
            row = min(row + 1, self.nrow - 1)
        elif a == RIGHT:
            if col + 1> self.ncol - 1:
                canPass=False
            col = min(col + 1, self.ncol - 1)
        elif a == UP:
            if row - 1< 0:canPass=False
            row = max(row - 1, 0)
        return canPass,row, col