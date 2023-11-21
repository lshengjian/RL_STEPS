from ..core import Model, Transition


class Plugin:
    def __init__(
        self,
        model: Model,
        delay: int = 0
    ):
        self._model = model
        self._delay = delay
        self._enable=True

    @property
    def enable(self):
        return self._enable
    
    @property
    def delay(self):
        return self._delay
        
    @enable.setter
    def enable(self, value):
        self._enable=value

    def update(self, t: Transition):
        pass

    def reset(self):
        self._enable=True
        
