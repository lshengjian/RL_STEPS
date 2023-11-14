import esper
from rlbase import Action
from gymnasium import Env
from rlbase.envs.processors import ManualControl,UpdateQs
from .random import RandomPolicy
class ManualPolicy(RandomPolicy):
    def __init__(self,env:Env):
        super().__init__(env)
        self.action=Action.STAY
        esper.set_handler('cmd_move_agent', self.move_agent)
        esper.add_processor(ManualControl(env.world.rederer._pygame))
        esper.add_processor(UpdateQs())

    def move_agent(self,action:Action):
        self.action=action

    def decition(self,state):
        rt=self.action
        self.action=Action.STAY
        return rt 
