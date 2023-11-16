from ..data import  Action,Transition
from .plugin import Plugin
from ..event_center import  EventCenter
from ..state import State
from .renderer import Renderer

class ManualControl(Plugin):
    def __init__(
        self,
        state: State,
        delay: int,
        renderer: Renderer,
        hub:EventCenter):
        super().__init__(state, delay)
        self.hub=hub
        self._pygame = renderer._pygame
        



    def update(self, t: Transition):
        pygame=self._pygame

        if  pygame is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.hub.dispatch_event("APP_QUIT")
                
                return
            elif event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)
                self.key_handler(key)

    def key_handler(self, key):
        # print(key)
        if key == "escape":
            self.hub.dispatch_event("APP_QUIT")
            return

        action_map = {
            "space": Action.STAY,
            "left": Action.LEFT,
            "right": Action.RIGHT,
            "up": Action.UP,
            "down": Action.DOWN,
        }


        if key not in action_map.keys():
            return
        self.hub.dispatch_event('cmd_move_agent', action_map[key])

