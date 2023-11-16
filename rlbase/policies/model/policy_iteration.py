from rlbase.envs.processors import PolicyIterationSystem
from rlbase import Action, CACHE, StatInfo
from rlbase import MiniGrid
from ..random import RandomPolicy
import esper
from rlbase.utils.sample import greedy_select


class IterationPolicy(RandomPolicy):
    def __init__(self, env: MiniGrid):
        super().__init__(env)
        self.action = Action.STAY
        esper.add_processor(PolicyIterationSystem(env.game.P))

    def decition(self, state):
        ent = CACHE[self.env.world.state2idx(state)]
        stat = esper.component_for_entity(ent, StatInfo)
        return greedy_select(stat.Qs, self.env.np_random)
