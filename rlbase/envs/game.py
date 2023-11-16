from .data import *
from .state import State
#from .event_center import EventCenter
from .plugins.plugin import Plugin


class Game:
    def __init__(self, state: State):  # ,center:EventCenter
        self.state = state
        self.plugins = []

    def add_plugin(self, plugin: Plugin):
        self.plugins.append(plugin)
        self.plugins.sort(key=lambda x: x.delay)

    def update(self,t: Transition):
        for p in self.plugins:
            p.update(t)

    # def reset(self):
    #     self.state.reset()
    #     for p in self.plugins:
    #         p.reset()

    def step(self, action: Action):
        s1=self.state.current
        s2, reward, terminated = self.state.step(action)
        self.update(Transition(s1,action,s2,reward, terminated))
        return s2, reward, terminated
