from ..data import G, NUM_ACTIONS, Tile, Focus,DIR_TO_VEC,Action
import esper

class FocusControl(esper.Processor):
    def __init__(
        self,
        pygame
    ) -> None:
        self._pygame = pygame
        self._cache=None
        #self.running = True
    def process(self):
        pygame=self._pygame
        if  pygame is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                esper.dispatch_event("APP_QUIT")
                #pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                key=pygame.key.name(event.key)
                self.key_handler(key)
        

        

    def key_handler(self,key):
        #print(key)
        if key == "escape":
            esper.dispatch_event("APP_QUIT")
            return

        dir_map = {
            "left": DIR_TO_VEC[Action.LEFT],
            "right": DIR_TO_VEC[Action.RIGHT],
            "up": DIR_TO_VEC[Action.UP],
            "down": DIR_TO_VEC[Action.DOWN], 
            
        }
        data=esper.get_component(Focus)
        r,c=0,0
        if len(data)<1:
            data=esper.get_component(Tile)
            e,t=data[0]
            r,c=t.row,t.col
            f=Focus(r,c)
            esper.add_component(e,f)
            #self._cache=(e,f)
        else:
            e,f=data[0]
            r,c=f.row,f.col
        if key not in dir_map.keys():
            return

        dir=dir_map[key]
        r+=dir[1]
        c+=dir[0]
        if c<0:c=0
        elif c>G.GRID_SIZE[1]-1:c=G.GRID_SIZE[1]-1
        if r<0:r=0
        elif r>G.GRID_SIZE[0]-1:r=G.GRID_SIZE[0]-1
        f.col=c
        f.row=r
        esper.remove_component(e,Focus)
        for e,t in esper.get_component(Tile):
            if t.row==r and t.col==c:
                 esper.add_component(e,f)
                 break

        
        

        
        