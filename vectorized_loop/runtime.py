from __future__ import annotations
from collections import defaultdict
from pyro.ops.provenance import track_provenance, get_provenance, ProvenanceTensor
from .ops import arange, transpose

import torch

_RUNTIME_MODE = {"infer_shapes": False}
_ENV = defaultdict(lambda: torch.tensor(torch.nan))
_BATCH_SHAPES = defaultdict(tuple)
_EVENT_SHAPES = defaultdict(tuple)


def clear_env():
    _ENV.clear() 
    _BATCH_SHAPES.clear()
    _EVENT_SHAPES.clear()


def init_env():
    for var in _ENV:
        batch_shape = tuple(torch.as_tensor(_BATCH_SHAPES[var]) + 1)
        _ENV[var] = torch.full(batch_shape + _EVENT_SHAPES[var], torch.nan)


def copy_env() -> defaultdict[str, torch.Tensor]:
    return defaultdict(lambda: torch.tensor(torch.nan), {key: value.clone() for key, value in _ENV.items()})


class DimAllocator:

    def __init__(self):
        self._name_to_dim: dict[str, int] = {}
        self._dim_to_name: dict[int, str] = {}

    clear = __init__

    def allocate(self, name: str, dim: int) -> int:
        # Allocates a new dimension for the given name.
        assert dim is None or dim < 0
        if name not in self._name_to_dim:
            if dim is None:
                dim = self.lookup()
            if dim in self._dim_to_name and self._dim_to_name[dim] != name:
                raise ValueError("Dimension %d is already allocated for %s." % (dim, self._dim_to_name[dim]))
            self._name_to_dim[name] = dim
            self._dim_to_name[dim] = name
        elif dim is not None and self._name_to_dim[name] != dim:
            raise ValueError("Dimension %d is already allocated for %s." % (dim, self._dim_to_name[dim]))
        return self._name_to_dim[name]

    def lookup(self) -> int:
        # Looks up an available dimension.
        available_dim = -1
        while available_dim in self._dim_to_name:
            available_dim += -1
        return available_dim


_DIM_ALLOCATOR = DimAllocator()


class EnumAllocator:

    def __init__(self):
        self._site_to_dim: defaultdict[str, list[int]] = defaultdict(list)
        self._dim_to_site: dict[int, str] = {}
        self._min_batch_dim: int = 0
        self._min_allocated_dim : int = 0

    clear = __init__

    def set_min_batch_dim(self, min_batch_dim: int):
        assert min_batch_dim <= 0
        self._min_batch_dim = min_batch_dim
        self._min_allocated_dim = min(self._min_allocated_dim, min_batch_dim)

    def allocate(self, site: str, new: bool = False) -> int:
        if new or (site not in self._site_to_dim) or (len(self._site_to_dim[site]) == 0):
            dim = self.lookup()
            self._site_to_dim[site] = self._site_to_dim[site] + [dim]
            self._dim_to_site[dim] = site
            return dim 
        return self._site_to_dim[site][0]
        
    def lookup(self) -> int:
        self._min_allocated_dim += -1
        return self._min_allocated_dim
    
    def shift(self, value: torch.Tensor, event_dim: int, enum_sites: set[str]) -> torch.Tensor:
        for site in enum_sites:
            dim = self._site_to_dim[site]
            for i in range(-1, -len(dim) - 1, -1):
                source_dim = dim[i]
                if value.ndim < -(source_dim - event_dim) or value.shape[source_dim - event_dim] == 1:
                    continue
                target_dim = self.allocate(site, new=True) if i == -1 else dim[i + 1]
                value = transpose(value, source_dim, target_dim, event_dim)
        return value
    

_ENUM_ALLOCATOR = EnumAllocator()


class IndexStack:
    
    def __init__(self):
        self._site_shape: list[int] = []  # e.g.) [5, 3, 4, 1, 2] if dim -5, -4, -3, -1 are used
        self._env_shape: list[int] = []  # e.g.) [3, 1, 1, 2] if dim -4, -1 are vectorized
        self._site_index: list[torch.LongTensor, int] = []  # e.g.) [i, tensor([0, 1, 2]), j, None, tensor([0, 1])] if counter=i in dim -5 and counter=j in dim -3
        self._env_index: list[torch.LongTensor, int] = []  # e.g.) [tensor([0, 1, 2]), None, None, tensor([0, 1])]

    clear = __init__

    def push(self, dim: int, size: int, indices: torch.LongTensor, counter: int, vectorized: bool):
        # Pushes the information.
        if -dim > len(self._site_shape):
            self._site_shape = [1] * (-dim - len(self._site_shape)) + self._site_shape
            self._site_index = [None] * (-dim - len(self._site_index)) + self._site_index
        self._site_shape[dim] = size
        self._site_index[dim] = indices if vectorized else counter

        if vectorized:
            if -dim > len(self._env_shape):
                self._env_shape = [1] * (-dim - len(self._env_shape)) + self._env_shape
                self._env_index = [None] * (-dim - len(self._env_index)) + self._env_index
            self._env_shape[dim] = size
            self._env_index[dim] = indices
        
    def pop(self, dim: int):
        # Pops the information.
        if -dim <= len(self._site_shape):
            self._site_shape[dim] = 1
            self._site_index[dim] = None
            while self._site_shape and self._site_shape[0] == 1:
                self._site_shape.pop(0)
                self._site_index.pop(0)        

        if -dim <= len(self._env_shape):
            self._env_shape[dim] = 1
            self._env_index[dim] = None
            while self._env_shape and self._env_shape[0] == 1:
                self._env_shape.pop(0)
                self._env_index.pop(0)


_INDEX_STACK = IndexStack()


class VarStack:

    def __init__(self):
        self._bound_vars: defaultdict[str, torch.BoolTensor] = defaultdict(lambda: torch.tensor(False))
        self._free_vars: defaultdict[str, torch.BoolTensor] = defaultdict(lambda: torch.tensor(False))
        self._bound_vars_stack: list[defaultdict[str, torch.BoolTensor]] = []
        self._free_vars_stack: list[defaultdict[str, torch.BoolTensor]] = []

    clear = __init__

    def union(self, a: defaultdict[str, torch.BoolTensor], b: defaultdict[str, torch.BoolTensor]) -> defaultdict[str, torch.BoolTensor]:
        result = defaultdict(lambda: torch.tensor(False))
        for var in (a.keys() | b.keys()):
            result[var] = a[var] | b[var]
        return result

    def difference(self, a: defaultdict[str, torch.BoolTensor], b: defaultdict[str, torch.BoolTensor]) -> defaultdict[str, torch.BoolTensor]:
        result = defaultdict(lambda: torch.tensor(False))
        for var in (a.keys() | b.keys()):
            result[var] = a[var] & (~b[var])
        return result

    def write_var(self, var: str):
        self._bound_vars[var] = self._bound_vars[var] | (_BRANCH_STACK.get() & (~self._free_vars[var]))
    
    def read_var(self, var: str):
        self._free_vars[var] = self._free_vars[var] | (_BRANCH_STACK.get() & (~self._bound_vars[var]))

    def push(self):
        self._bound_vars_stack.append(self._bound_vars.copy())
        self._free_vars_stack.append(self._free_vars.copy())
        self._bound_vars.clear()
        self._free_vars.clear()

    def pop(self) -> set[str]:
        bound_vars = self._bound_vars
        free_vars = self._free_vars

        prev_bound_vars = self._bound_vars_stack.pop()
        prev_free_vars = self._free_vars_stack.pop()

        self._bound_vars = self.union(prev_bound_vars, self.difference(bound_vars, prev_free_vars))
        self._free_vars = self.union(prev_free_vars, self.difference(free_vars, prev_bound_vars))

        return {var for var in (bound_vars.keys() | free_vars.keys()) if (~bound_vars[var]).any()}


_VAR_STACK = VarStack()
        

class BranchStack:

    def __init__(self):
        self._conds: list[torch.BoolTensor] = [torch.tensor(True)]

    clear = __init__
        
    def push(self, cond: torch.BoolTensor):
        self._conds.append(self._conds[-1] & cond)

    def pop(self) -> torch.BoolTensor:
        return self._conds.pop()

    def get(self) -> torch.BoolTensor:
        return self._conds[-1]


_BRANCH_STACK = BranchStack()


def clear_allocators():
    _DIM_ALLOCATOR.clear()
    _ENUM_ALLOCATOR.clear()
    _INDEX_STACK.clear()
    _VAR_STACK.clear()
    _BRANCH_STACK.clear()


def get_enum_batch_shapes(value: torch.Tensor, event_dim: int) -> tuple[torch.LongTensor]:
    # Returns enum shape and batch shape.
    shape = value.shape  # (enum | batch | event)
    enum_batch_shape = shape[:len(shape)-event_dim]  # (enum | batch)
    min_batch_dim = _ENUM_ALLOCATOR._min_batch_dim # -2
    if len(enum_batch_shape) <= -min_batch_dim:
        enum_shape = ()  # (enum,)
        batch_shape = enum_batch_shape  # (batch,)
    else:
        enum_shape = enum_batch_shape[:len(enum_batch_shape) + min_batch_dim]  # (enum,)
        batch_shape = enum_batch_shape[len(enum_batch_shape) + min_batch_dim:]  # (batch,)
    return enum_shape, batch_shape


def process_index(batch_index: tuple[torch.LongTensor], batch_shape: tuple[int], enum_shape: tuple[int], mode: str) -> tuple[torch.LongTensor]:
    # Processes the given batch index to the full index with enum dims.
    enum_index = (arange(size, dim - len(enum_shape) + _ENUM_ALLOCATOR._min_batch_dim) for dim, size in enumerate(enum_shape))

    batch_index = list(batch_index).copy()
    if len(batch_shape) > len(batch_index):
        batch_index = [None] * (len(batch_shape) - len(batch_index)) + batch_index 
    batch_index = batch_index[len(batch_index)-len(batch_shape):]

    for dim in range(len(batch_index)):
        dim = dim - len(batch_index)
        if batch_index[dim] is None or batch_shape[dim] == 1:
            if mode == "w":
                batch_index[dim] = arange(batch_shape[dim], dim)
            elif mode == "r":
                batch_index[dim] = -1
            else:
                raise ValueError("Invalid mode: %s" % mode)
                
    return tuple(enum_index) + tuple(batch_index)


def is_index_full(tensor: torch.Tensor, index: tuple[torch.LongTensor]) -> bool:
    # Checks if the given index is full.
    for i, size in enumerate(tensor.shape):
        if i < len(index) and torch.as_tensor(index[i]).numel() < size:
            return False
    return True


def match_enum_shapes(target_enum_shape: tuple[int], source_enum_shape: tuple[int]) -> list[list[int]]:
    site_to_dim = _ENUM_ALLOCATOR._site_to_dim
    dim_to_site = _ENUM_ALLOCATOR._dim_to_site
    min_batch_dim = _ENUM_ALLOCATOR._min_batch_dim
    result = []
    for i in range(-len(target_enum_shape), 0, 1):
        if target_enum_shape[i] > 1:
            target_dim = i + min_batch_dim
            dims = [target_dim]
            for source_dim in site_to_dim[dim_to_site[target_dim]]:
                j = source_dim - min_batch_dim
                if len(source_enum_shape) >= -j and source_enum_shape[j] > 1:
                    dims.append(source_dim)
                    break
            if len(dims) == 1 or dims[0] != dims[1]:
                result.append(dims)
    return result


def setter(target: torch.Tensor, source: torch.Tensor, event_shape: tuple[int], batch_index: tuple[torch.LongTensor]) -> torch.Tensor:
    target_enum_shape, batch_shape = get_enum_batch_shapes(target, len(event_shape))
    source_enum_shape, _ = get_enum_batch_shapes(source, len(event_shape))

    result = target
    for dims in match_enum_shapes(target_enum_shape, source_enum_shape):
        if len(dims) == 1:
            result = result.index_select(dims[0], torch.tensor(0))
        else:
            result = result.transpose(dims[0], dims[1])
    
    while result.dim() >= 1 and result.shape[0] == 1:
        result = result.squeeze(0)
    
    if len(source_enum_shape) > 0:
        batch_shape = (1,) * (-_ENUM_ALLOCATOR._min_batch_dim - len(batch_shape)) + batch_shape
    result_shape = source_enum_shape + batch_shape + event_shape
    result = result.expand(result_shape).clone()

    index = process_index(batch_index, batch_shape, source_enum_shape, "w")
    
    condition = _BRANCH_STACK.get()
    if torch.all(condition):
        branch_source = source
    else:
        condition = condition.reshape(condition.shape + (1,) * len(event_shape))
        branch_source = torch.where(condition, source, result[index])

    # For backward efficiency.
    if is_index_full(result, index):
        result = torch.as_tensor(branch_source, dtype=result.dtype).broadcast_to(result.shape)
    else:
        result[index] = torch.as_tensor(branch_source, dtype=result.dtype)
    
    # For TraceGraph_ELBO.
    if isinstance(target, ProvenanceTensor) or isinstance(source, ProvenanceTensor):
        result = track_provenance(result, get_provenance([target, source]))

    return result


def getter(source: torch.Tensor, event_shape: tuple[int], batch_index: tuple[torch.LongTensor]) -> torch.Tensor:
    enum_shape, batch_shape = get_enum_batch_shapes(source, len(event_shape))
    index = process_index(batch_index, batch_shape, enum_shape, "r")
    return source[index].clone()