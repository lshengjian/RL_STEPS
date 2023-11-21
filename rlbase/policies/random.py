from gymnasium import Env
from rlbase.core import TState
class RandomPolicy:
    def __init__(self,env:Env):
        self.env:Env=env
    def decition(self,state:TState):
        return self.env.action_space.sample() 
