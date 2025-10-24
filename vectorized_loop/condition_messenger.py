from pyro.poutine.handlers import _make_handler
from pyro.poutine.trace_struct import Trace
from pyro.poutine import condition_messenger
from typing import Callable, overload

from .runtime import getter, _RUNTIME_MODE, _DIM_ALLOCATOR, _ENUM_ALLOCATOR, _INDEX_STACK


class ConditionMessenger(condition_messenger.ConditionMessenger):

    def __init__(self, data) -> None:
        self.data = data

    def _pyro_sample(self, msg) -> None:
        name = msg["name"]

        if name in self.data:
            value = self.data.nodes[name] if isinstance(self.data, Trace) else self.data[name]
            if _RUNTIME_MODE["infer_shapes"]:
                msg["value"] = value[(0,) * len(_INDEX_STACK._site_index)]
            else:
                msg["value"] = getter(value, msg["fn"].event_shape, _INDEX_STACK._site_index)
            msg["is_observed"] = msg["value"] is not None


@overload
def condition(
    data,
) -> ConditionMessenger: ...


@_make_handler(ConditionMessenger)
@overload
def condition(
    fn: Callable,
    data,
) -> Callable: ...