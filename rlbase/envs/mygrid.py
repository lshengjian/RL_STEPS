from __future__ import annotations
from gymnasium import  spaces

from .minigrid import MiniGrid


class MyGrid(MiniGrid):

    def __init__(
        self,
        render_mode: str = None,
        map=["SX", "-G"],
        fps=24,
        win_size=(1024,768)
    ):
        super().__init__(render_mode, map,fps,win_size)
        self.observation_space = spaces.Box(0,1,(2,))

    def step(self, a):
        s, rew, terminated = self.game.step(a)
        r,c=self.model.state2idx(s)
        x=c/self.model.ncol
        y=r/self.model.nrow
        return (x,y), rew, terminated, False, {"prob": 1, "action_mask": self.game.model.action_mask(s)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s = self.model.state
        r,c=self.model.state2idx(s)
        x=c/self.model.ncol
        y=r/self.model.nrow
        return (x,y), {"prob": 1, "action_mask": self.model.action_mask(s)}