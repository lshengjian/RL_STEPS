import numpy as np
from gymnasium.error import DependencyNotInstalled
from ..data import *

import esper

class RenderSystem(esper.Processor):
    def __init__(self,render_mode,FPS:int=4):
        self.render_mode=render_mode

        self.FPS=FPS
        self._surface = None
        self._clock = None
        self.window_size = G.WIN_SIZE
        w,h=G.WIN_SIZE
        row,col=G.GRID_SIZE
        # pygame utils
        G.TILE_SIZE=(w//col,h//row)
       
        self.render()

    def process(self):
        if self._surface is None:
            return
        # Clear the window:
        #self._surface.fill(self.clear_color)
        # This will iterate over every Entity that has this Component, and blit it:
        for _, tile in esper.get_component(Tile):
            #self.window.blit(rend.image, (rend.x, rend.y))
            x,y=tile.position
            w,h=G.TILE_SIZE
            r=min(w,h)
            self._gfxdraw.box(self._surface,[x-w//2,y-h//2,w,h],tile.color)
        for _,(_,tile) in esper.get_components(Focus,Tile):
            for i in range(NUM_ACTIONS):
                x,y=tile.side(i)
                self._gfxdraw.filled_circle(self._surface,x,y,r//10,(5,5,5))

            
        # Flip the framebuffers
        self._pygame.display.flip()  
        if self.render_mode == "human":
            self._pygame.event.pump()
            self._pygame.display.update()
            if self.FPS>0:
                self._clock.tick(self.FPS) 

        '''
        
        #self._surface.fill((0, 0, 0))
        desc = self.desc
        nrow,ncol=G.GRID_SIZE
        w,h=self.cell_size
        r=min(w,h)/2
        for y in range(nrow):
            for x in range(ncol):
                pos = [x * w, y * h]
                rect = (*pos, *self.cell_size)
                flag=desc[y][x]
                flag = flag.decode()
                color=COLORS[flag]
                gfxdraw.box(self._surface,rect,color)
                s=self.world.idx2state(y,x)
                total=sum(visits[s])
                if visits[s,STAY]:
                    gfxdraw.aacircle(
                        self._surface,
                        int(pos[0]+w/2),
                        int(pos[1]+h/2),
                        int(r / 2*visits[s,STAY]/total),
                        VISITE_COLOR
                    )
                for k,d in ACT_DIRS.items():
                    if visits[s,k]:
                        scale=visits[s,k]/total
                        x0,y0=pos[0]+w//2,pos[1]+h//2
                        dx,dy=d[0]*w//2,d[1]*h//2
                        dx*=scale
                        dy*=scale
                        gfxdraw.line(self._surface,x0,y0,x0+int(dx),y0+int(dy),VISITE_COLOR)
                
                suf=self._font.render(f'{V[s]:.2f}',1,VISITE_COLOR,color)
                pos[0]+=self.cell_size[0]*0.05
                pos[1]+=self.cell_size[1]*0.05
                self._surface.blit(suf,pos)
        
        # paint the elf
        bot_row, bot_col = self.world.state2idx(self.world.state)
        pos = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        #last_action = self.world.lastaction if self.world.lastaction is not None else 0
        gfxdraw.filled_circle(
            self._surface,
            int(pos[0]+w/2),
            int(pos[1]+h/2),
            int(r / 2),
            AGENT_COLOR
        )
        for i in range(1,nrow):
            yi=i*h
            gfxdraw.hline(self._surface,0,self.window_size[0],yi,TEXT_COLOR)
        for i in range(1,ncol):
            xi=i*w
            gfxdraw.vline(self._surface,xi,0,self.window_size[1],TEXT_COLOR)

        '''     

    def render(self):
        #,visits:np.ndarray,V:np.ndarray=None
        mode=self.render_mode
        if mode not in['human','rgb_array']:
            return
        try:
            import pygame
            from pygame import gfxdraw

            self._pygame=pygame
            self._gfxdraw=gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium`"
            ) from e

        if self._surface is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("使用方向键移动焦点")
                self._font  =  pygame.font.Font(None,26)
                self._surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self._surface = pygame.Surface(self.window_size)

        assert (
            self._surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self._clock is None:
            self._clock = pygame.time.Clock()

        #self.process()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._surface)), axes=(1, 0, 2)
            )



 