from typing import List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.utils import seeding


from .data import *
from .world import World

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
    - Reach forbidden: -1
    - Reach frozen: 0

    ## Episode End
    The episode ends when The player reaches the goal 

    """

    metadata = {
        "render_modes": ["human", "none"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        map_name="4x4",
        is_terminate_reach_goal=True,
        isAutoPolicy=True,
        isDemo=False,
    ):
        self.is_terminate_reach_goal=is_terminate_reach_goal
        self.desc = desc = np.asarray(MAPS[map_name], dtype="c")
        self.world=World(render_mode,desc,isAutoPolicy,isDemo)
        self.metadata['render_fps']=G.FPS
        self.observation_space = spaces.Discrete(self.world.nS)
        self.action_space = spaces.Discrete(self.world.nA)


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
        terminated = terminated if  self.is_terminate_reach_goal else False
        return (int(s), r, terminated, False, {"prob": 1,"action_mask": self.action_mask(s)})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.world.reset(self.np_random)
        s=self.world.state
        if self.render_mode == "human":
            self.world.update()
        return int(s), {"prob": 1,"action_mask": self.action_mask(s)}

    def render(self):
        pass

 