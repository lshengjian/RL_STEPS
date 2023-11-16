from ..data import AGENT_COLOR, Transition
from .plugin import Plugin
# from ..event_center import  EventCenter
from ..state import State
from .renderer import Renderer


class LastRenderer(Plugin):
    def __init__(self,
                 state: State,
                 delay: int,
                 renderer: Renderer):
        super().__init__(state, delay)
        self.renderer = renderer

    def update(self, t: Transition):
        self.renderer.flip_wait()
