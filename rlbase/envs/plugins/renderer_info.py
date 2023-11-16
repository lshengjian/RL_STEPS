import numpy as np
from gymnasium.error import DependencyNotInstalled
from ..data import *
from .plugin import Plugin
#from ..event_center import  EventCenter
from ..state import  State
from .renderer import Renderer
class RendererAgent(Plugin):
    def __init__(self, 
                 state:State,
                 delay:int,
                 renderer:Renderer):
        super().__init__(state,delay)
        self.renderer=renderer
     


    def update(self,t:Transition):
        if self._surface is None:
            return
        
        x,y=self.state.position
        r = min(*self.renderer.tile_size)
        self.renderer._gfxdraw.filled_circle(self._surface, x, y, r//4, AGENT_COLOR)
    


    def draw_info(self, r):
        k=r//96
        for i in range(self.model.nS):
            if i==self.model.state:
                x, y = self.model.position
                self._gfxdraw.filled_circle(self._surface, x, y, r//4, AGENT_COLOR)
                #self.draw_text(x-10*k,y-5*k,f"{info.V:.1f}",True)
                #self.draw_text(x-12,y+16,f"{info.Qs[0]:.1f}",False,(255,255,255))

            # for i in range(1,NUM_ACTIONS):
            #     x, y = tile.side(i)
            #     self.draw_text(x-10*k,y+5*k,f"{info.Qs[i]:.1f}",False)

    def draw_history(self, r):
        pass
        # agent:Agent=esper.get_component(Agent)[0][1]
        # for t1,_,t2,_,dx,dy in agent.visited:
        #     x1, y1 = t1.position
        #     x2, y2 = t2.position
        #     if t1==t2:
        #         self._gfxdraw.circle(self._surface, x1, y1, r//3, (*HISTORY_COLOR,30))
        #     else:
        #         self._gfxdraw.line(self._surface, x1+dx, y1+dy, x2,y2, (*HISTORY_COLOR,30))

    def draw_tiles(self, tile_w, tile_h):
        for i in range(self.model.nS):
            x, y = self.model.side(i,Action.STAY)
            color=self.model.get_color(i)
            self._gfxdraw.box(self._surface, [x-tile_w//2, y-tile_h//2, tile_w, tile_h], color)
        r,c=G.GRID_SIZE
        for i in range(1,r):
            self._gfxdraw.hline(self._surface,0,tile_w*c,i*tile_h,(0,0,0))
        for i in range(1,c):
            self._gfxdraw.vline(self._surface,i*tile_w,0,tile_h*r,(0,0,0))

    def close(self):
        if self._surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
    def render(self):
        # ,visits:np.ndarray,V:np.ndarray=None
        mode = self.render_mode
        if mode not in ['human', 'rgb_array']:
            return
        try:
            import pygame
            from pygame import gfxdraw
            self._pygame = pygame
            self._gfxdraw = gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium`"
            ) from e

        if self._surface is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("使用方向键移动焦点")
                w, h = G.TILE_SIZE
                r = min(w, h)
                #print(r)
                self._font = pygame.font.Font(None, 12*r//86)
                self._font2 = pygame.font.Font(None, 14*r//86)
                self._surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self._surface = pygame.Surface(self.window_size)

        assert (
            self._surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self._clock is None:
            self._clock = pygame.time.Clock()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._surface)), axes=(1, 0, 2)
            )
