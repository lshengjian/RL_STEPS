from ..core import Model,Plugin, AGENT_COLOR,Transition
from .renderer import Renderer
import numpy as np

class ReturnRenderer(Plugin):
    def __init__(self,
                 model: Model,
                 delay: int,
                 renderer: Renderer,
                 gamma=0.9
                 ):
        super().__init__(model, delay)
        self.renderer = renderer
        self.gamma=gamma
        self._V=np.zeros(model.nS)
        self._Qs:np.ndarray=np.zeros((model.nS,model.nA))
    def reset(self):
        super().reset()
        self._V*=0
        self._Qs*=0


    def update(self, t: Transition):
        x, y = self.renderer.agent_position
        r = min(*self.renderer.tile_size)/6.2
        self._Qs[t.s1,t.action]=t.reward+self.gamma*self._V[t.s2]
        self._V[t.s1]=self._Qs[t.s1].max()
        for s in range(self._model.nS):
            for a in range(self._model.nA):
                x, y = self.renderer.side(s, a)
                self.renderer.draw_text(x-r,y-r/3,f'{self._Qs[s,a]:.2f}',True)
