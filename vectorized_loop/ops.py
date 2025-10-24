from __future__ import annotations
from functools import reduce
import torch


class Index:
    """
    Indexing operation that supports NaN propagation.
    It currently does not support
    1) `self.data` with batch and enum dimensions,
    2) slice indexing,
    3) ellipsis indexing.
    """

    def __init__(self, data: torch.Tensor):
        self.data = data.clone()

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)

        safe_idx = ()
        isnan_idx = ()
        for i in idx:
            i = torch.as_tensor(i).clone()
            isnan = torch.isnan(i)
            i[isnan] = 0
            i = i.long()
            safe_idx += (i,)
            isnan_idx += (isnan,)
        result = self.data[safe_idx]
        isnan_idx = reduce(torch.logical_or, isnan_idx)
        result[isnan_idx] = float("nan")
        return result


def arange(n: int, dim: int):
    assert dim < 0
    return torch.arange(n).reshape((-1,) + (1,) * (-dim - 1))


def transpose(value: torch.Tensor, dim0: int, dim1: int, event_dim: int):
    assert dim0 < 0 and dim1 < 0 and event_dim >= 0
    min_dim = min(dim0, dim1)

    if -min_dim > (value.ndim - event_dim):
        diff = - (value.ndim - event_dim) - min_dim
        value = value.reshape((1,) * diff + value.shape)
    
    value = value.transpose(dim0 - event_dim, dim1 - event_dim)

    while value.ndim > 0 and value.size(0) == 1:
        value = value.squeeze(0)

    return value


def cat(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
    """
    Concatenate tensors along a dimension, broadcasting as necessary.
    """
    assert dim < 0
    shapes = []
    for t in tensors:
        shape = list(t.shape)
        shape = [1] * max(0, -dim - len(shape)) + shape
        shape[dim] = 1
        shapes.append(shape)
    expand_shape = list(torch.broadcast_shapes(*shapes))
    expand_shape[dim] = -1
    return torch.cat([t.expand(expand_shape) for t in tensors], dim=dim)
