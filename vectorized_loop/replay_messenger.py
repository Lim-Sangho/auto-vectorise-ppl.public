from pyro.poutine.handlers import _make_handler
from pyro.poutine import replay_messenger
from typing import Callable, overload

from .runtime import getter, _RUNTIME_MODE, _DIM_ALLOCATOR, _ENUM_ALLOCATOR, _INDEX_STACK
from .ops import transpose


class ReplayMessenger(replay_messenger.ReplayMessenger):

    def __init__(self, trace = None):
        self.trace = trace
        self.transposed = False

    def _pyro_sample(self, msg):
        name = msg["name"]

        if msg["infer"].get("name") is not None and not self.transposed:
            actual_dim = _DIM_ALLOCATOR._name_to_dim[msg["infer"].get("name")]
            target_dim = _ENUM_ALLOCATOR.allocate(name)
            self.trace.nodes[name]["value"] = transpose(self.trace.nodes[name]["value"], actual_dim, target_dim, self.trace.nodes[name]["fn"].event_dim)
            self.transposed = True
    
        if self.trace is not None and name in self.trace and not msg["is_observed"]:
            guide_msg = self.trace.nodes[name]
            if _RUNTIME_MODE["infer_shapes"]:
                # msg["value"] = guide_msg["value"][(0,) * len(guide_msg["fn"].batch_shape)]
                msg["value"] = guide_msg["value"][(0,) * len(_INDEX_STACK._site_index)]
            else:
                msg["value"] = getter(guide_msg["value"], msg["fn"].event_shape, _INDEX_STACK._site_index)
            msg["infer"] = guide_msg["infer"]
            msg["done"] = True


@overload
def replay(
    fn: None,
    *args,
    **kwargs,
) -> ReplayMessenger: ...


@overload
def replay(
    fn: Callable,
    *args,
    **kwargs,
) -> Callable: ...


@_make_handler(ReplayMessenger)
def replay(fn): ...