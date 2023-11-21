
from rlbase.core import Action,TState
from gymnasium import Env
#from rlbase.envs.event_center import EventCenter
from .random import RandomPolicy
class ManualPolicy(RandomPolicy):
    def __init__(self,env:Env):
        super().__init__(env)
        self.action=Action.STAY
        self.running=True
        #self.hub=hub



    def decition(self,state:TState):
        pygame=self.env.renderer._pygame
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                #self.hub.dispatch_event("APP_QUIT")
                self.running=False
                return
            elif event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)
                self.key_handler(key)
        rt=self.action
        self.action=Action.STAY
        return rt 
    def key_handler(self, key):
        # print(key)
        if key == "escape":
            self.running=False
            #self.hub.dispatch_event("APP_QUIT")
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
        self.action=action_map[key]


