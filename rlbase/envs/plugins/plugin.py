#from ..event_center import  EventCenter
from ..state import  State
from ..data import Transition
class Plugin:
    def __init__(
        self,
        state:State,
        delay:int=0
        #center:EventCenter
    ) -> None:
        self.state=state
        self.delay=delay

    def update(self,t:Transition):
        pass
    # def reset(self ):
    #     pass
