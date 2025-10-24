from __future__ import annotations
from pyro.poutine.messenger import Messenger
from pyro.poutine.subsample_messenger import CondIndepStackFrame, _Subsample
from .runtime import (_RUNTIME_MODE, _ENV, _BATCH_SHAPES, _EVENT_SHAPES,
                      _DIM_ALLOCATOR, _ENUM_ALLOCATOR, _INDEX_STACK, _VAR_STACK,
                      copy_env, get_enum_batch_shapes, process_index)
from .trace_messenger import remove_sites
from .ops import arange

import torch
import logging
from collections import defaultdict
logger = logging.getLogger(__name__)


class VectorizedLoopMessenger(Messenger):
    
    def __init__(self, name, size, dim, vectorized, history, device):
        self.name = name
        self.size = size
        self.dim = _DIM_ALLOCATOR.allocate(self.name, dim)
        self.vectorized = vectorized
        self.history = min(history, self.size - 1)
        self.counter = None if vectorized else -1
        self.indices = arange(self.size, self.dim).to(device)
        self.sites = set()
        self.enum_sites = set()
        self.free_vars = set()

    def __enter__(self):
        super().__enter__()
        _INDEX_STACK.push(self.dim, self.size, self.indices, self.counter, self.vectorized)
        _VAR_STACK.push()
        return self.indices if self.vectorized else None
    
    def __exit__(self, exc_type, exc_value, traceback):
        _INDEX_STACK.pop(self.dim)
        self.free_vars |= _VAR_STACK.pop()
        return super().__exit__(exc_type, exc_value, traceback)

    def __iter__(self):
        if _RUNTIME_MODE["infer_shapes"]:
            with self:
                yield 0

        elif self.vectorized:
            repeat = 0
            prev_env = copy_env()
            while True:
                with self:
                    yield self.indices
                repeat += 1
                if repeat == self.history + 1 or self._check_fixed_pt(self.dim, prev_env, self.free_vars):
                    break
                prev_env = copy_env()
                self._shift(self.dim, self.enum_sites, self.free_vars)
                remove_sites(self.sites)
            self._shift_last(self.dim)
            logger.debug(f"{self.name} (vectorized): repeat {repeat}")

        else:
            for i in self.indices:
                self.counter += 1
                with self:
                    yield i.item()
                if i.item() < self.size - 1:
                    self._shift(self.dim, self.enum_sites, self.free_vars)
            self._shift_last(self.dim)

    def _process_message(self, msg):
        frame = CondIndepStackFrame(name=self.name, dim=self.dim, size=self.size, counter=self.counter, full_size=self.size)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]
        if not _RUNTIME_MODE["infer_shapes"] and self.vectorized and not msg["done"] and msg["type"] == "sample" and not isinstance(msg["fn"], _Subsample):
            msg["fn"] = msg["fn"].expand(torch.broadcast_shapes(msg["fn"].batch_shape, self.indices.shape))
        if msg["type"] == "sample" and not isinstance(msg["fn"], _Subsample):
            self.sites.add(msg["name"])
        if msg["infer"].get("enumerate") == "parallel":
            self.enum_sites.add(msg["name"])

    def _shift(self, dim: int, enum_sites: set[str], free_vars: set[str]):
        for var in free_vars:
            enum_shape, batch_shape = get_enum_batch_shapes(_ENV[var], len(_EVENT_SHAPES[var])) 
            if len(batch_shape) >= -dim and batch_shape[dim] > 1:
                index = process_index(_INDEX_STACK._env_index, batch_shape, enum_shape, "w")
                _ENV[var][index] = _ENV[var][index].roll(shifts=1, dims=dim + len(enum_shape) + len(batch_shape))
                source_index = list(index); source_index[dim] = 0
                target_index = list(index); target_index[dim] = -1
                _ENV[var][tuple(target_index)] = _ENV[var][tuple(source_index)].clone()
            if len(enum_sites) > 0:
                _ENV[var] = _ENUM_ALLOCATOR.shift(_ENV[var], len(_EVENT_SHAPES[var]), enum_sites)
        
    def _shift_last(self, dim: int):
        for var in _ENV:
            enum_shape, batch_shape = get_enum_batch_shapes(_ENV[var], len(_EVENT_SHAPES[var]))
            if len(batch_shape) >= -dim and batch_shape[dim] > 1:
                index = process_index(_INDEX_STACK._env_index, batch_shape, enum_shape, "w")
                parent_index = list(index); parent_index[dim] = -1
                _ENV[var][index] = _ENV[var][index].roll(shifts=1, dims=dim + len(enum_shape) + len(batch_shape))
                _ENV[var][index] = _ENV[var][tuple(parent_index)].clone()

    def _check_fixed_pt(self, dim: int, prev_env: defaultdict[str, torch.Tensor], free_vars: set[str]) -> bool:
        for var in free_vars:
            _, batch_shape = get_enum_batch_shapes(_ENV[var], len(_EVENT_SHAPES[var]))
            if len(batch_shape) >= -dim and batch_shape[dim] > 1:
                val, prev_val = _ENV[var], prev_env[var]
                isnan, prev_isnan = torch.isnan(val), torch.isnan(prev_val)
                if not torch.equal(val[~isnan], prev_val[~prev_isnan]) or not torch.equal(isnan, prev_isnan):
                    return False
        return True


def range(name, size, dim=None, vectorized=False, history=float("inf"), device=None):
    return VectorizedLoopMessenger(name, size, dim, vectorized, history, device=torch.get_default_device() if device is None else device)