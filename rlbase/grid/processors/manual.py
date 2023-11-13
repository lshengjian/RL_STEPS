from ..data import G, NUM_ACTIONS, Tile, Agent, DIR_TO_VEC, Action, CACHE
import esper


class ManualControl(esper.Processor):
    def __init__(
        self,
        pygame
    ) -> None:
        self._pygame = pygame
        self._cache = None
        # self.running = True

    def process(self):
        pygame = self._pygame
        if pygame is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                esper.dispatch_event("APP_QUIT")
                # pygame.quit()
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

        # data = esper.get_components(Tile,Agent)
        # e,(t,a) = data[0]
        # r, c = t.row, t.col
        if key not in action_map.keys():
            return
        esper.dispatch_event('cmd_move_agent', action_map[key])
        print(key)

        # dir =DIR_TO_VEC[action_map[key]]
        # r += dir[1]
        # c += dir[0]
        # if c < 0:
        #     c = 0
        # elif c > G.GRID_SIZE[1]-1:
        #     c = G.GRID_SIZE[1]-1
        # if r < 0:
        #     r = 0
        # elif r > G.GRID_SIZE[0]-1:
        #     r = G.GRID_SIZE[0]-1
        # esper.remove_component(e, Agent)
        # e2=CACHE[(r,c)]
        # esper.add_component(e2, a)
        # esper.dispatch_event('agent_moved',e,action_map[key],e2)
