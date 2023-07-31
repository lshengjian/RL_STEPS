from typing import List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.utils import seeding

from .utils import categorical_sample
from .config import *
from .world import World
from .renderer import Renderer
class MiniGrid(Env):
    """
    ## Action Space
    The action shape is `(1,)` in the range `{0, 4}` indicating
    which direction to move the player.

    - 0: Stay
    - 1: Move left
    - 2: Move down
    - 3: Move right
    - 4: Move up

    ## Observation Space
    The observation is a value representing the player's current position as
    current_row * nrows + current_col (where both the row and col start at 0).

    ## Rewards

    Reward schedule:
    - Reach goal: +1
    - Reach firbidden: -1
    - Reach frozen: 0

    ## Episode End
    The episode ends when The player reaches the goal 

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        map_name="4x4"
    ):
        desc = np.asarray(MAPS[map_name], dtype="c")
        self.world=World(desc)
        self.renderer=Renderer(self.world,self.metadata['render_fps'])
        nA=self.world.nA
        nS=self.world.nS
        self.visits=np.zeros((nS,nA))
        self.V=np.zeros(nS)
        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)
        self.render_mode = render_mode

    def update_value(self, state: int,val:float):
        self.V[state]=val

    def action_mask(self, state: int):
        mask = np.ones(self.world.nA, dtype=np.int8)
        nrow=self.world.nrow
        ncol=self.world.ncol
        row,col=self.world.state2idx(state)
        if col==0:
            mask[LEFT]=0
        elif col==ncol-1:
            mask[RIGHT]=0
        if row==0:
            mask[UP]=0
        elif row==nrow-1:
            mask[DOWN]=0
        return mask
    
    def step(self, a):
        s=self.world.state
        self.visits[s,a]+=1
        transitions = self.world.P[s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.world.state = s
        self.world.lastaction = a

        if self.render_mode == "human":
            self.render()
        return (int(s), r, t, False, {"prob": p,"action_mask": self.action_mask(s)})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.world.state = s= categorical_sample(self.world.initial_state_distrib, self.np_random)
        self.visits*=0
        self.lastaction = None
        if self.render_mode == "human":
            self.render()
        return int(s), {"prob": 1,"action_mask": self.action_mask(s)}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
            )
            return
        
        return self.renderer.render(self.render_mode,self.visits,self.V)

 