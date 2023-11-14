from typing import  Optional
import numpy as np
from gymnasium import Env, spaces

from .data import *
from .world import World

class MiniGrid(Env):
    """
    ## Action Space
    The action shape is `(1,)` in the range `{0, 4}` indicating

    Reward schedule:
    - Reach goal: +1
    - Reach forbidden: -1
    - Reach frozen: 0

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        map_name="4x4",
        show_stat_info=True
        #is_terminate_reach_goal=True,
        # isAutoPolicy=True,
        # isDemo=False,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        #self.is_terminate_reach_goal=is_terminate_reach_goal
        self.desc = desc = np.asarray(MAPS[map_name], dtype="c")
        self.world=World(render_mode,desc,show_stat_info)
        self.metadata['render_fps']=G.FPS
        self.observation_space = spaces.Discrete(self.world.nS)
        self.action_space = spaces.Discrete(self.world.nA)

    def close(self):
        self.world.rederer.close()
    def render(self):
        return self.world.rederer.render()
    
    def action_mask(self, state: int):
        mask = np.ones(self.world.nA, dtype=np.int8)
        nrow=self.world.nrow
        ncol=self.world.ncol
        row,col=self.world.state2idx(state)
        if col==0:
            mask[Action.LEFT]=0
        elif col==ncol-1:
            mask[Action.RIGHT]=0
        if row==0:
            mask[Action.UP]=0
        elif row==nrow-1:
            mask[Action.DOWN]=0
        return mask
    
    def step(self, a):
        s=self.world.state
        s, r, terminated=self.world.move(a)
        #terminated = terminated if  self.is_terminate_reach_goal else False
        return (s, r, terminated, False, {"prob": 1,"action_mask": self.action_mask(s)})

    def reset(  self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset(self.np_random)
        s=self.world.state
        self.world.update()
        return s, {"prob": 1,"action_mask": self.action_mask(s)}



 