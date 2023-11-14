from gymnasium import Env

class RandomPolicy:
    def __init__(self,env:Env):
        self.env:Env=env
    def decition(self,state):
        return self.env.action_space.sample() 
