from random import randrange
from .plugin import Plugin
from ..core import Model, Action,Transition,VISITED_COLOR
from .renderer import Renderer


class VisitedRenderer(Plugin):
    MAX_SIZE = 1024

    def __init__(self,
                 model: Model,
                 delay: int,
                 renderer: Renderer):
        super().__init__(model, delay)
        self.renderer = renderer
        self.model.ext_data['offsets'] = []
        self.model.ext_data['visited'] = []
    def reset(self ):
        self.model.ext_data['offsets'].clear()
        self.model.ext_data['visited'].clear()
        
    def update(self, t: Transition):
        self.model.ext_data['visited'].append(t)
        self.model.ext_data['offsets'].append(
            (randrange(-8, 8), randrange(-8, 8)))

        if len(self.model.ext_data['visited']) > VisitedRenderer.MAX_SIZE:
            self.model.ext_data['offsets'].pop(0)
            self.model.ext_data['visited'].pop(0)

        w, h = self.renderer.tile_size
        r = min(w, h)
        for i, t0 in enumerate(self.model.ext_data['visited']):
            t: Transition = t0
            dx, dy = self.model.ext_data['offsets'][i]
            if t.action == Action.STAY:
                x, y = self.renderer.side(t.s1, t.action)
                self.renderer.circle(x, y, r//8+dx, VISITED_COLOR)
                continue
            x, y = self.renderer.side(t.s1, t.action)
            x += dx
            y += dy
            x2, y2 = self.renderer.side(t.s2, 0)
            self.renderer.line(x, y, x2, y2, VISITED_COLOR)
