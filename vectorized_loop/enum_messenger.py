from pyro.poutine.handlers import _make_handler
from pyro.poutine.messenger import Messenger
from .runtime import _RUNTIME_MODE, _DIM_ALLOCATOR, _ENUM_ALLOCATOR, _INDEX_STACK
from .ops import transpose


def enumerate_site(msg):
    fn, num_samples = msg["fn"], msg["infer"].get("num_samples")
    if num_samples is None:
        # Enumerate all possible values of the distribution.
        value = fn.enumerate_support(expand=False)
    else:
        # Monte Carlo sample the distribution.
        value = fn(sample_shape=(num_samples,))
    assert value.dim() == 1 + len(fn.batch_shape) + len(fn.event_shape)
    return value


class EnumMessenger(Messenger):

    def __init__(self, name: str = None):
        self.name = name

    def _pyro_sample(self, msg):
        # Runs only if not in the shape inference mode.
        if _RUNTIME_MODE["infer_shapes"]:
            return
        
        # Runs only if the site is enumerated in parallel.
        if msg["infer"].get("enumerate") != "parallel":
            return

        # Already enumerated in the guide.
        if self.name is not None:
            msg["infer"]["name"] = self.name
            return
        
        value = enumerate_site(msg)
        actual_dim = -1 - len(msg["fn"].batch_shape)
        target_dim = _ENUM_ALLOCATOR.allocate(msg["name"])
        event_dim = msg["fn"].event_dim

        value = transpose(value, actual_dim, target_dim, event_dim)
        msg["infer"]["_enumerate_dim"] = target_dim
        msg["value"] = value
        msg["done"] = True

    
@_make_handler(EnumMessenger)
def enum(fn): ...