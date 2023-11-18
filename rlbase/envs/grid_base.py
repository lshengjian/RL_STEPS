import numpy as np
from gymnasium import Env, spaces

from .game import Game
from .core import Model,MAPS

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
        render_mode: str = None,
        map_name="4x4",
        fps=24,
        win_size=(1024,768)
    ):

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.metadata['render_fps'] = fps
        self.desc = np.asarray(MAPS[map_name], dtype="c")
        self.render_mode = render_mode
        self.init_game()
        
        

    def init_game(self):
        self.model = Model(self.desc)
        self.game = Game(self.model)
        self.renderer = Renderer(self.model, 100,self.render_mode,self.metadata['render_fps'] )
        self.game.add_plugin(self.renderer)
        self.game.add_plugin(AgentRenderer(self.model,101,self.renderer))
        self.game.add_plugin(VisitedRenderer(self.model,102,self.renderer))
        self.game.add_plugin(LastRenderer(self.model,199,self.renderer))


    def close(self):
        self.renderer.close()

    def render(self):
        return self.renderer.render()

    def step(self, a):
        s, r, terminated = self.game.step(a)
        return (s, r, terminated, False, {"prob": 1, "action_mask": self.game.model.action_mask(s)})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        model=self.game.model
        model.set_random_generator(self.np_random)
        self.game.reset()
        
        
        self.observation_space = spaces.Discrete(model.nS)
        self.action_space = spaces.Discrete(model.nA)
        s = model.state
        return s, {"prob": 1, "action_mask": model.action_mask(s)}
