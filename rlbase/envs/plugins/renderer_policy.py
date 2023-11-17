import numpy as np
from .plugin import Plugin
from ..core import Model, Transition,Action,VISITED_COLOR,ACTION_FLAGS
from .renderer import Renderer


class PolicyRenderer(Plugin):
    def __init__(self,
                 model: Model,
                 delay: int,
                 renderer: Renderer):
        super().__init__(model, delay)
        self.renderer = renderer
        self.model.ext_data['pi'] = np.ones((model.nS,model.nA))/model.nA

    def update(self, t: Transition):
        r = min(*self.renderer.tile_size)
        for s in range(self.model.nS):
            for a in range(self.model.nA):
                # if a == Action.STAY:
                #     x, y = self.renderer.side(s,a)
                #     self.renderer.circle(x, y, r//8, VISITED_COLOR)
                #     continue
                x, y = self.renderer.side(s,a)
                pi=self.model.ext_data['pi']
                self.renderer.draw_text(x-10, y, f'{pi[s,a]}', VISITED_COLOR)

