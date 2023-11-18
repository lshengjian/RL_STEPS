import numpy as np
from gymnasium import Env, spaces

from .game import Game
from .core import Model,MAPS

from .plugins.renderer import Renderer
from .plugins.renderer_agent import AgentRenderer
from .plugins.renderer_last import LastRenderer
from .plugins.renderer_policy import PolicyRenderer
from .grid_base import MiniGrid
class PolicyGrid(MiniGrid):

    def init_game(self):
        self.model = Model(self.desc)
        self.game = Game(self.model)
        self.renderer = Renderer(self.model, 100,self.render_mode,self.metadata['render_fps'] )
        self.game.add_plugin(self.renderer)
        self.game.add_plugin(PolicyRenderer(self.model,101,self.renderer))
        self.game.add_plugin(AgentRenderer(self.model,102,self.renderer))
        
        self.game.add_plugin(LastRenderer(self.model,199,self.renderer))

