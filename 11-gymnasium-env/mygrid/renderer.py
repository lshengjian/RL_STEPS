import numpy as np
from os import path
from gymnasium.error import DependencyNotInstalled
from .config import *
from .world import World
class Renderer:
    def __init__(self,world:World,FPS:int=4):
        self.world=world
        self.FPS=FPS
        self._surface = None
        self.clock = None
        nrow=self.world.nrow
        ncol=self.world.ncol
        # pygame utils
        self.window_size = (TILE_SIZE[0] * ncol, TILE_SIZE[1] * nrow)
        self.cell_size = TILE_SIZE


    def render(self, mode:str,visits:np.ndarray):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self._surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("My Mini Grid")
                self._surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self._surface = pygame.Surface(self.window_size)

        assert (
            self._surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self._surface.fill((0, 0, 0))
        desc = self.world.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        nrow=self.world.nrow
        ncol=self.world.ncol
        
        for y in range(nrow):
            for x in range(ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)
                flag=desc[y][x]
                flag = flag.decode()
                color=COLORS[flag]
                gfxdraw.box(self._surface,rect,color)
                s=self.world.idx2state(y,x)
                if visits[s,STAY]:
                    gfxdraw.aacircle(
                        self._surface,
                        int(pos[0]+TILE_SIZE[0]/2),
                        int(pos[1]+TILE_SIZE[1]/2),
                        int(TILE_SIZE[1] / 6),
                        (255, 255, 255),
                    )
                for k,d in ACT_DIRS.items():
                    max_num=max(visits[s])
                    if visits[s,k]:
                        scale=visits[s,k]/max_num
                        x0,y0=pos[0]+TILE_SIZE[0]//2,pos[1]+TILE_SIZE[1]//2
                        dx,dy=d[0]*TILE_SIZE[0]//2,d[1]*TILE_SIZE[1]//2
                        dx*=scale
                        dy*=scale
                        gfxdraw.line(self._surface,x0,y0,x0+int(dx),y0+int(dy),(255, 255, 255))
        
        # paint the elf
        bot_row, bot_col = self.world.state2idx(self.world.state)
        pos = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        #last_action = self.world.lastaction if self.world.lastaction is not None else 0
        gfxdraw.filled_circle(
            self._surface,
            int(pos[0]+TILE_SIZE[0]/2),
            int(pos[1]+TILE_SIZE[1]/2),
            int(TILE_SIZE[1] / 4),
            (249, 12, 3)
         )
        

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.FPS)
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._surface)), axes=(1, 0, 2)
            )



 