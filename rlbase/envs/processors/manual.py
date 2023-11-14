from ..data import  Action
import esper


class ManualControl(esper.Processor):
    def __init__(
        self,
        pygame
    ) -> None:
        self._pygame = pygame
        self._cache = None


    def process(self):
        pygame = self._pygame
        if pygame is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                esper.dispatch_event("APP_QUIT")
                
                return
            elif event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)
                self.key_handler(key)

    def key_handler(self, key):
        # print(key)
        if key == "escape":
            esper.dispatch_event("APP_QUIT")
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
        esper.dispatch_event('cmd_move_agent', action_map[key])

