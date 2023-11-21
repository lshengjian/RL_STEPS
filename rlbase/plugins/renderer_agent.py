from ..core import Model,Plugin, AGENT_COLOR,Transition
from .renderer import Renderer


class AgentRenderer(Plugin):
    def __init__(self,
                 model: Model,
                 delay: int,
                 renderer: Renderer):
        super().__init__(model, delay)
        self.renderer = renderer

    def update(self, t: Transition):
        x, y = self.renderer.agent_position
        r = min(*self.renderer.tile_size)
        self.renderer.filled_circle(x, y, r//4, AGENT_COLOR)
