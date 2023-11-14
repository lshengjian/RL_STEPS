import numpy as np
from gymnasium.error import DependencyNotInstalled
from ..data import *

import esper


class RenderSystem(esper.Processor):
    def __init__(self, render_mode,show_stat_info=False):
        self.render_mode = render_mode
        self.show_stat_info=show_stat_info

        self.fps = G.FPS
        self._surface = None
        self._clock = None
        self.window_size = G.WIN_SIZE
        w, h = G.WIN_SIZE
        row, col = G.GRID_SIZE
        
        G.TILE_SIZE = (w//col, h//row)

        self.render()
        self.offsets={}

    def draw_text(self,x,y,msg,isBig=False):
        textSerface =self._font2.render(msg, True,TEXT_COLOR ) if isBig else self._font.render(msg, True,TEXT_COLOR )
        self._surface.blit(textSerface, (x, y))

    def process(self):
        if self._surface is None:
            return
        

        w, h = G.TILE_SIZE
        r = min(w, h)
        self._surface.fill((255,255,255))
        self.draw_tiles(w, h)
        if self.show_stat_info:
            self.draw_history(r)
            self.draw_info(r)
        else:
            for e, (tile,agent) in esper.get_components(Tile,Agent):
                x, y = tile.position
                self._gfxdraw.filled_circle(self._surface, x, y, r//4, AGENT_COLOR)
    

        self._pygame.display.flip()
        if self.render_mode == "human":
            self._pygame.event.pump()
            self._pygame.display.update()
            if self.fps > 0:
                self._clock.tick(self.fps)


    def draw_info(self, r):
        k=r//96
        for e, (tile,info) in esper.get_components(Tile,StatInfo):
            start=0
            if esper.has_component(e,Agent):
                x, y = tile.position
                self._gfxdraw.filled_circle(self._surface, x, y, r//4, AGENT_COLOR)
                self.draw_text(x-10*k,y-5*k,f"{info.V:.1f}",True)
                #self.draw_text(x-12,y+16,f"{info.Qs[0]:.1f}",False,(255,255,255))
                start=1

            for i in range(start,NUM_ACTIONS):
                x, y = tile.side(i)
                self.draw_text(x-10*k,y+5*k,f"{info.Qs[i]:.1f}",False)

    def draw_history(self, r):
        agent:Agent=esper.get_component(Agent)[0][1]
        for t1,_,t2,_,dx,dy in agent.visited:
            x1, y1 = t1.position
            x2, y2 = t2.position
            if t1==t2:
                self._gfxdraw.circle(self._surface, x1, y1, r//3, (*HISTORY_COLOR,30))
            else:
                self._gfxdraw.line(self._surface, x1+dx, y1+dy, x2,y2, (*HISTORY_COLOR,30))

    def draw_tiles(self, tile_w, tile_h):
        for e, (tile,info) in esper.get_components(Tile,StatInfo):
            x, y = tile.position
            self._gfxdraw.box(self._surface, [x-tile_w//2, y-tile_h//2, tile_w, tile_h], COLORS[tile.flag])
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
