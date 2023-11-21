from .model import Model
from .define import  Transition,Action
from .plugin import Plugin


class Game:
    def __init__(self, model: Model):  # ,center:EventCenter
        self.model = model
        self.plugins = []

    def add_plugin(self, plugin: Plugin):
        self.plugins.append(plugin)
        self.plugins.sort(key=lambda x: x.delay)

    def update(self,t: Transition):
        for p in self.plugins:
            if p.enable:
                p.update(t)

    def reset(self):
        self.model.reset()
        for p in self.plugins:
            p.reset()

    def step(self, action: Action):
        s1=self.model.state
        s2, reward, terminated = self.model.step(action)
        self.update(Transition(s1,action,s2,reward, terminated))
        return s2, reward, terminated
