from typing import List, Optional
import numpy as np
from gymnasium.utils import seeding
from .config import *



class MyWorld:
    def __init__(self,is_slippery:bool,desc:np.ndarray):
        
        self.desc=desc
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.nA = nA = 5
        self.nS = nS = nrow * ncol
        self.s=0
        self.lastaction=None

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}#动作转移概率
        for row in range(nrow):
            for col in range(ncol):
                s = self.to_s(row, col)
                for a in range(nA):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if is_slippery and a!=STAY and letter not in b"GH" :
                        acts=[(a - 1) % 4, a, (a + 1) % 4] #左，前，右
                        ps=[0.1,0.8,0.1] #概率
                        for b,p in zip(acts,ps):
                            li.append(
                                (p, *self.update_probability_matrix(row, col, b))
                            )
                    else:
                        li.append((1.0, *self.update_probability_matrix(row, col, a)))

    def update_probability_matrix(self,row, col, action):
        ok,newrow, newcol = self.inc(row, col, action)
        newstate = self.to_s(newrow, newcol)
        newletter = self.desc[newrow, newcol]
        terminated = bytes(newletter) in b"G"
        reward=0
        if not ok:
            reward=OUT_BOUND
        elif newletter == b"G":
            reward = GO_TO_GOAL
        elif newletter == b"H":
            reward = GO_TO_HOLE
        return newstate, reward, terminated


    @staticmethod
    def generate_random_map(
        size: int = 8, p: float = 0.8, seed: Optional[int] = None
    ) -> List[str]:
        """Generates a random valid map (one that has a path from start to goal)

        Args:
            size: size of each side of the grid
            p: probability that a tile is frozen
            seed: optional seed to ensure the generation of reproducible maps

        Returns:
            A random valid map
        """
        valid = False
        board = []  # initialize to make pyright happy

        np_random, _ = seeding.np_random(seed)

        while not valid:
            p = min(1, p)
            board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])
            board[0][0] = "S"
            board[-1][-1] = "G"
            valid = MyWorld.is_valid(board, size)
        return ["".join(x) for x in board]

    @staticmethod
    def is_valid(board: List[List[str]], max_size: int) -> bool:
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                        continue
                    if board[r_new][c_new] == "G":
                        return True
                    if board[r_new][c_new] != "H":
                        frontier.append((r_new, c_new))
        return False

    
    def to_s(self,row, col):
        return row * self.ncol + col

    
    def inc(self,row, col, a):
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