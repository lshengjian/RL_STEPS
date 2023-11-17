from ..core import Model, Transition

class Plugin:
    def __init__(
        self,
        model: Model,
        delay: int = 0
    ) -> None:
        self.model = model
        self.delay = delay

    def update(self, t: Transition):
        pass
    def reset(self ):
        pass
