from typing import Any ,Callable
from types import MethodType
from weakref import ref 
from weakref import WeakMethod
class EventCenter:
    def __init__(
        self
    ) -> None:
        self.registry={}

    def set_handler(self,name: str, func: Callable[..., None]) -> None:
        """Register a function to handle the named event type.

        After registering a function (or method), it will receive all
        events that are dispatched by the specified name.

        .. note:: A weak reference is kept to the passed function,
        """
        if name not in self.registry:
            self.registry[name] = set()

        if isinstance(func, MethodType):
            self.registry[name].add(WeakMethod(func, self._make_callback(name)))
        else:
            self.registry[name].add(ref(func, self._make_callback(name)))


    def remove_handler(self,name: str, func: Callable[..., None]) -> None:
        """Unregister a handler from receiving events of this name.

        If the passed function/method is not registered to
        receive the named event, or if the named event does
        not exist, this function call will pass silently.
        """
        if func not in self.registry.get(name, []):
            return

        self.registry[name].remove(func)
        if not self.registry[name]:
            del self.registry[name]

    def dispatch_event(self,name: str, *args: Any) -> None:
        """Dispatch an event by name, with optional arguments.

        Any handlers set with the :py:func:`esper.set_handler` function
        will recieve the event. If no handlers have been set, this
        function call will pass silently.

        :note:: If optional arguments are provided, but set handlers
                do not account for them, it will likely result in a
                TypeError or other undefined crash.
        """
        for func in self.registry.get(name, []):
            func()(*args)


    def _make_callback(self,name: str) -> Callable[[Any], None]:
        """Create an internal callback to remove dead handlers."""
        def callback(weak_method: Any) -> None:
            self.registry[name].remove(weak_method)
            if not self.registry[name]:
                del self.registry[name]

        return callback
