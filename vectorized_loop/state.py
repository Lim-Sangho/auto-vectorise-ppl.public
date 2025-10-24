from __future__ import annotations
from abc import *

from pyro.poutine import block
from .runtime import (_RUNTIME_MODE, _ENV, _BATCH_SHAPES, _EVENT_SHAPES,
                      _DIM_ALLOCATOR, _ENUM_ALLOCATOR, _INDEX_STACK, _VAR_STACK,
                      clear_env, init_env, setter, getter)

import torch


def vectorize(fn):
    def wrapper(*args, **kwargs):
        class Wrapper(State):
            model = fn
        wrapper = Wrapper()
        return wrapper.run(*args, **kwargs)
    return wrapper


class State(metaclass=ABCMeta):
    
    def __init__(self):
        clear_env()

    @abstractmethod
    def model(self, *args, **kwargs):
        pass
    
    def run(self, *args, **kwargs):
        # Infers the shapes of the variables.
        with block():
            with torch.random.fork_rng(devices=[torch.get_default_device()], enabled=True):
                _RUNTIME_MODE["infer_shapes"] = True
                self.model(*args, **kwargs)
                _RUNTIME_MODE["infer_shapes"] = False
                init_env()

        # Allocates the dimensions for the enumerated sites.
        _ENUM_ALLOCATOR.set_min_batch_dim(min(_DIM_ALLOCATOR._dim_to_name, default=0))

        # Runs the model.
        return self.model(*args, **kwargs)

    def __setattr__(self, var: str, value: torch.Tensor):
        value = torch.as_tensor(value, dtype=torch.float32)

        if _RUNTIME_MODE["infer_shapes"]:
            _BATCH_SHAPES[var] = torch.broadcast_shapes(_BATCH_SHAPES[var], _INDEX_STACK._env_shape)
            _EVENT_SHAPES[var] = torch.broadcast_shapes(_EVENT_SHAPES[var], value.shape)
            _ENV[var] = value.clone()

        else:
            _VAR_STACK.write_var(var)
            _ENV[var] = setter(_ENV[var], value, _EVENT_SHAPES[var], _INDEX_STACK._env_index)

    def __getattr__(self, var: str) -> torch.Tensor:
        if _RUNTIME_MODE["infer_shapes"]:
            return _ENV[var].clone()

        else:
            _VAR_STACK.read_var(var)
            return getter(_ENV[var], _EVENT_SHAPES[var], _INDEX_STACK._env_index)
    
