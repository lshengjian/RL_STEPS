import numpy as np
from .define import TState
from .constants import Action,OUT_BOUND,IN_GOAL,IN_FORBIDDEN,COLORS,NUM_ACTIONS
from .utils import categorical_sample

class Model:

    def __init__(self,
                 desc: np.ndarray): # colors:Dict[str,Tuple[int,int,int]]

        self.desc = desc
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.nA = nA = NUM_ACTIONS
        self.nS = nS = nrow * ncol
        self.state:TState = 0

        self.ext_data={}#插件产生的扩展数据

        self.initial_state_distrib = np.array(
            desc == b'S').astype("float32").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}  # 动作转移矩阵
        for row in range(nrow):
            for col in range(ncol):
                s=self.idx2state(row,col)
                for a in range(nA):
                    li = self.P[s][a]
                    li.append((1.0, *self.move(row, col, a)))
    
    def set_random_generator(self,np_random:np.random.Generator):
        self.np_random = np_random
       
            
    def action_mask(self, state: TState):
        mask = np.ones(self.nA, dtype=np.int8)
        nrow = self.nrow
        ncol = self.ncol
        row, col = self.state2idx(state)
        if col == 0:
            mask[Action.LEFT] = 0
        elif col == ncol-1:
            mask[Action.RIGHT] = 0
        if row == 0:
            mask[Action.UP] = 0
        elif row == nrow-1:
            mask[Action.DOWN] = 0
        return mask

    def reset(self ):
        self.state = categorical_sample(self.initial_state_distrib, self.np_random)

    def get_color(self,s:TState):
        r,c=self.state2idx(s)
        flag = self.desc[r][c].decode()
        return COLORS[flag]
      


    def idx2state(self, row:int, col:int)->TState:
        return row*self.ncol+col

    def state2idx(self, state:TState):
        return state//self.ncol, state % self.ncol

    def step(self, action: Action):
        r,c=self.state2idx(self.state)
        newstate, reward, terminated=self.move(r,c,action)
        self.state=newstate
        return newstate, reward, terminated

    def move(self, row: int, col: int, action: Action):
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

    def _check_next_pos(self,row: int, col: int, action: Action):
        canPass = True
        if action == Action.LEFT:
            if col - 1 < 0:
                canPass = False
            col = max(col - 1, 0)
        elif action == Action.RIGHT:
            if col + 1 > self.ncol - 1:
                canPass = False
            col = min(col + 1, self.ncol - 1)
        elif action == Action.DOWN:
            if row + 1 > self.nrow - 1:
                canPass = False
            row = min(row + 1, self.nrow - 1)
        elif action == Action.UP:
            if row - 1 < 0:
                canPass = False
            row = max(row - 1, 0)
        return canPass, row, col
