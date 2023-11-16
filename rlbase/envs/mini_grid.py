from typing import Optional
import numpy as np
from gymnasium import Env, spaces

from .data import *
from .game import Game
from .state import State
#from .event_center import EventCenter
from .plugins.renderer import Renderer
from .plugins.renderer_agent import AgentRenderer
from .plugins.renderer_last import LastRenderer
from .plugins.renderer_visited import VisitedRenderer

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
    ):

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.metadata['render_fps'] = G.FPS
        self.desc = np.asarray(MAPS[map_name], dtype="c")
        self.render_mode = render_mode

    def close(self):
        self.renderer.close()

    def render(self):
        return self.renderer.render()

    def step(self, a):

        s, r, terminated = self.game.step(a)
        # terminated = terminated if  self.is_terminate_reach_goal else False
        return (s, r, terminated, False, {"prob": 1, "action_mask": self.game.state.action_mask(s)})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = State(self.desc, self.np_random)
        
        self.game = Game(state)
        self.renderer = Renderer(state, 100,self.render_mode)
        self.game.add_plugin(self.renderer)
        self.game.add_plugin(AgentRenderer(state,101,self.renderer))
        self.game.add_plugin(VisitedRenderer(state,102,self.renderer))
        self.game.add_plugin(LastRenderer(state,199,self.renderer))

        self.observation_space = spaces.Discrete(self.game.state.nS)
        self.action_space = spaces.Discrete(self.game.state.nA)
        # self.game.update()
        s = state.current
        return s, {"prob": 1, "action_mask": state.action_mask(s)}
