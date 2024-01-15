import numpy as np
from gymnasium.error import DependencyNotInstalled
from typing import  Tuple
from ..core import Model, Action, DIR_VECTOR, TEXT_COLOR, Transition,Plugin


class Renderer(Plugin):
    def __init__(self,
                 model: Model,
                 delay: int = 100,
                 render_mode: str = None,
                 fps=4,
                 win_size=(800, 600)):
        super().__init__(model, delay)
        self.render_mode = render_mode

        self.fps = fps
        self._surface = None
        self._clock = None
        self.window_size = win_size
        w, h = win_size
        row, col = model.nrow, model.ncol
        self.tile_size = (w//col, h//row)
        self.render()

    @property
    def agent_position(self) -> Tuple[int, int]:
        w, h = self.tile_size
        r, c = self._model.state2idx(self._model.state)
        return (c*w+w//2, r*h+h//2)

    def side(self, s: int, a: Action) -> Tuple[int, int]:
        w, h = self.tile_size
        r, c = self._model.state2idx(s)
        x, y = c*w+w//2, r*h+h//2
        dx, dy = DIR_VECTOR[a]
        dx = int(w/3*dx)
        dy = int(h/3*dy)
        return (x+dx, y+dy)

    def line(self, x1, y1, x2, y2, color):
        
        self._gfxdraw.line(self._surface, x1, y1, x2, y2, color)

    def circle(self, x, y, r, color):
        self._gfxdraw.circle(self._surface, x, y, r, color)

    def filled_circle(self, x, y, r, color):
        self._gfxdraw.filled_circle(self._surface, x, y, r, color)

    def filled_polygon(self, data, color):
        self._gfxdraw.filled_polygon(self._surface, data, color)

    def draw_text(self, x, y, msg, isBig=False):
        textSerface = self._font2.render(
            msg, True, TEXT_COLOR) if isBig else self._font.render(msg, True, TEXT_COLOR)
        self._surface.blit(textSerface, (x, y))

    def update(self, t: Transition):
        if self._surface is None:
            return

        w, h = self.tile_size
        r = min(w, h)
        self._surface.fill((255, 255, 255))
        self.draw_tiles(w, h)

    def flip_wait(self):
        
        if self.render_mode == "human":
            self._pygame.display.flip()
            self._pygame.event.pump()
            self._pygame.display.update()
            if self.fps > 0:
                self._clock.tick(self.fps)

    def draw_tiles(self, tile_w, tile_h):
        for i in range(self._model.nS):
            x, y = self.side(i, Action.STAY)
            color = self._model.get_color(i)
            self._gfxdraw.box(
                self._surface, [x-tile_w//2, y-tile_h//2, tile_w, tile_h], color)
        r, c = self._model.nrow, self._model.ncol
        for i in range(1, r):
            self._gfxdraw.hline(self._surface, 0, tile_w *
                                c, i*tile_h, (0, 0, 0))
        for i in range(1, c):
            self._gfxdraw.vline(self._surface, i*tile_w,
                                0, tile_h*r, (0, 0, 0))

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
                w, h = self.tile_size
                r = min(w, h)
                # print(r)
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
