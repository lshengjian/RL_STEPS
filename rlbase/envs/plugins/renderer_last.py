from .plugin import Plugin
from ..core import Model, Transition
from .renderer import Renderer


class LastRenderer(Plugin):
    def __init__(self,
                 model: Model,
                 delay: int,
                 renderer: Renderer):
        super().__init__(model, delay)
        self.renderer = renderer

    def update(self, t: Transition):
        self.renderer.flip_wait()
