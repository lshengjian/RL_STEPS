import pygame
from pygame import gfxdraw
from pathlib import Path

import esper


FPS = 60
RESOLUTION = 720, 480

dir=Path(__file__).parent

##################################
#  Define some Components:
##################################
class Velocity:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class Renderable:
    def __init__(self, color, posx, posy, depth=0):
        self.color = color

        self.depth = depth
        self.x = posx
        self.y = posy
        # self.w = image.get_width()
        # self.h = image.get_height()


################################
#  Define some Processors:
################################
class MovementProcessor:
    def __init__(self, minx, maxx, miny, maxy):
        super().__init__()
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy

    def process(self):
        # This will iterate over every Entity that has BOTH of these components:
        for ent, (vel, rend) in esper.get_components(Velocity, Renderable):
            # Update the Renderable Component's position by it's Velocity:
            rend.x += vel.x
            rend.y += vel.y
            # An example of keeping the sprite inside screen boundaries. Basically,
            # adjust the position back inside screen boundaries if it tries to go outside:
            rend.x = max(self.minx, rend.x)
            rend.y = max(self.miny, rend.y)
            rend.x = min(self.maxx - 64, rend.x)
            rend.y = min(self.maxy - 64, rend.y)


class RenderProcessor:
    def __init__(self, window, clear_color=(0, 0, 0)):
        super().__init__()
        self.window = window
        
        self.clear_color = clear_color
    def get_points(self,pos):
        x,y=pos
        return [(x,y-5),(x+20,y-5),(x+20,y-10),(x+30,y),(x+20,y+10),(x+20,y+5),(x,y+5)]
        

    def process(self):
        # Clear the window:
        self.window.fill(self.clear_color)
        # This will iterate over every Entity that has this Component, and blit it:
        for ent, rend in esper.get_component(Renderable):
            #self.window.blit(rend.image, (rend.x, rend.y))
            gfxdraw.rectangle(self.window,[rend.x,rend.y,64,64],rend.color)
            gfxdraw.polygon(self.window,self.get_points((rend.x+24, rend.y+24)),(235,255,0))
        # Flip the framebuffers
        pygame.display.flip()


################################
#  The main core of the program:
################################
def run():
    # Initialize Pygame stuff
    pygame.init()
    window = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption("Esper Pygame example")
    clock = pygame.time.Clock()
    pygame.key.set_repeat(1, 1)

    # Initialize Esper world, and create a "player" Entity with a few Components.
    player = esper.create_entity()
    esper.add_component(player, Velocity(x=0, y=0))
    esper.add_component(player, Renderable((255,0,0), posx=100, posy=100))
    # Another motionless Entity:
    enemy = esper.create_entity()
    esper.add_component(enemy, Renderable((0,255,0), posx=400, posy=250))

    # Create some Processor instances, and asign them to be processed.
    render_processor = RenderProcessor(window=window)
    movement_processor = MovementProcessor(minx=0, maxx=RESOLUTION[0], miny=0, maxy=RESOLUTION[1])

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    # Here is a way to directly access a specific Entity's
                    # Velocity Component's attribute (y) without making a
                    # temporary variable.
                    esper.component_for_entity(player, Velocity).x = -3
                elif event.key == pygame.K_RIGHT:
                    # For clarity, here is an alternate way in which a
                    # temporary variable is created and modified. The previous
                    # way above is recommended instead.
                    player_velocity_component = esper.component_for_entity(player, Velocity)
                    player_velocity_component.x = 3
                elif event.key == pygame.K_UP:
                    esper.component_for_entity(player, Velocity).y = -3
                elif event.key == pygame.K_DOWN:
                    esper.component_for_entity(player, Velocity).y = 3
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    esper.component_for_entity(player, Velocity).x = 0
                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    esper.component_for_entity(player, Velocity).y = 0

        # A single call to e.process() will update all Processors:
        render_processor.process()
        movement_processor.process()

        clock.tick(FPS)


if __name__ == "__main__":
    run()
    pygame.quit()