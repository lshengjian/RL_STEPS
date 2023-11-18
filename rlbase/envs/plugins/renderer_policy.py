import numpy as np
from .plugin import Plugin
from ..core import Model, Transition, Action, POLICY_COLOR
from .renderer import Renderer


class PolicyRenderer(Plugin):
    def __init__(self,
                 model: Model,
                 delay: int,
                 renderer: Renderer):
        super().__init__(model, delay)
        self.renderer = renderer
        self.model.ext_data['pi'] = np.ones((model.nS, model.nA))/model.nA

    def make_arrorw(self, action: Action, p: int):
        # print(action,p)

        if action == Action.UP:
            return [(-0.1, 0), (-0.1, -0.618), (-0.18, -0.618), (0, -0.8),
                    (0.18, -0.618), (0.1, -0.618), (0.1, 0)]
        elif action == Action.DOWN:
            return [(-0.1, 0), (-0.1, 0.618), (-0.18, 0.618), (0, 0.8),
                    (0.18, 0.618), (0.1, 0.618), (0.1, 0)]
        elif action == Action.LEFT:
            return [(0, -0.1), (-0.618, -0.1), (-0.618, -0.18), (-0.8, 0),
                    (-0.618, 0.18), (-0.618, 0.1), (0, 0.1)]
        elif action == Action.RIGHT:
            return [(0, -0.1), (0.618, -0.1), (0.618, -0.18), (0.8, 0),
                    (0.618, 0.18), (0.618, 0.1), (0, 0.1)]
        return []

    def update(self, t: Transition):
        r = min(*self.renderer.tile_size)
        for s in range(self.model.nS):
            for a in range(self.model.nA):
                x, y = self.renderer.side(s, 0)
                pi = self.model.ext_data['pi']
                p = pi[s][a]

                if p < 0.01:
                    continue
                if a == Action.STAY:
                    self.renderer.circle(x, y, int(r//3*p+0.5), POLICY_COLOR)
                else:
                    data = self.make_arrorw(a, p)
                    data = np.array(data)
                    data *= p*r
                    data = np.array(data, dtype=int) + \
                        np.array([[x, y]], dtype=int)
                    self.renderer.filled_polygon(data, POLICY_COLOR)
        s = t.s1
        a = t.action
        pi[s][a] += 0.05
        pi[s, :] /= sum(pi[s])
