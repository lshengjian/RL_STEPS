from typing import List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.utils import seeding

from .myutils import categorical_sample
from .config import *
from .myworld import MyWorld
from .myrender import MyRender
class MyLakeEnv(Env):
    """
    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.

    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up

    ## Observation Space
    The observation is a value representing the player's current position as
    current_row * nrows + current_col (where both the row and col start at 0).

    For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
    The number of possible observations is dependent on the size of the map.

    The observation is returned as an `int()`.

    ## Starting State
    The episode starts with the player in state `[0]` (location [0, 0]).

    ## Rewards

    Reward schedule:
    - Reach goal: +1
    - Reach hole: 0
    - Reach frozen: 0

    ## Episode End
    The episode ends if the following happens:

    - Termination:
        1. The player moves into a hole.
        2. The player reaches the goal at `max(nrow) * max(ncol) - 1` (location `[max(nrow)-1, max(ncol)-1]`).

    - Truncation (when using the time_limit wrapper):
        1. The length of the episode is 100 for 4x4 environment, 200 for FrozenLake8x8-v1 environment.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=True,
    ):
        if desc is None and map_name is None:
            desc = MyWorld.generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        desc = np.asarray(desc, dtype="c")
        self.world=MyWorld(is_slippery,desc)
        self.renderer=MyRender(self.world,self.metadata['render_fps'])
        
        
        nA=self.world.nA
        nS=self.world.nS
        nrow=self.world.nrow
        ncol=self.world.ncol
        self.reward_range = (0, 1)

    
        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.render_mode = render_mode
    def action_mask(self, state: int):
        mask = np.ones(self.world.nA, dtype=np.int8)
        nrow=self.world.nrow
        ncol=self.world.ncol
        row=state//ncol
        col=state%ncol
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
        s=self.world.s
        transitions = self.world.P[s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.world.s = s
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
        self.world.s = s= categorical_sample(self.world.initial_state_distrib, self.np_random)
      
        self.lastaction = None
        if self.render_mode == "human":
            self.render()
        return int(s), {"prob": 1,"action_mask": self.action_mask(s)}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
               
            )
            return
        
        return self.renderer.render_gui(self.render_mode)

 