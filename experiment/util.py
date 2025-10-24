from __future__ import annotations
from scipy import stats

import os
import torch
import logging
import numpy as np
import contextlib


_TIMEIT = True

@contextlib.contextmanager
def time_it_context(enabled=True):
    "A context manager to measure the time of a block of code"
    global _TIMEIT
    original_timeit = _TIMEIT
    _TIMEIT = enabled
    try:
        yield
    finally:
        _TIMEIT = original_timeit


def time_it(fn=None):
    "Decorator to measure the time of a function"
    "If no function is provided, return a context manager"
    if fn is None:
        return TimeIt()
    
    if not _TIMEIT:
        return fn

    else:
        def wrapper(*args, **kwargs):
            with TimeIt() as t:
                result = fn(*args, **kwargs)
            print(f"{fn}: {t}")
            return result
        return wrapper


class TimeIt:
    "Context manager to measure the time of a block of code"

    def __init__(self):
        self.time = 0
        if _TIMEIT:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if _TIMEIT:
            torch.cuda.synchronize()
            self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _TIMEIT:
            self.end.record()
            torch.cuda.synchronize()
            self.time += self.start.elapsed_time(self.end) / 1000

    def __repr__(self):
        return f"{self.time:.4f} s"


@contextlib.contextmanager
def suppress_backward():
    """
    A context manager to temporarily disable torch.Tensor.backward().
    """
    _original_backward = torch.Tensor.backward
    torch.Tensor.backward = lambda *args, **kwargs: None
    
    try:
        yield
    finally:
        torch.Tensor.backward = _original_backward


@contextlib.contextmanager
def suppress_reset_peak_memory_stats():
    """
    A context manager to temporarily disable torch.cuda.reset_peak_memory_stats().
    """
    if torch.cuda.is_available():
        _original_reset_peak_memory_stats = torch.cuda.reset_peak_memory_stats
        torch.cuda.reset_peak_memory_stats = lambda *args, **kwargs: None
        
        try:
            yield
        finally:
            torch.cuda.reset_peak_memory_stats = _original_reset_peak_memory_stats
    else:
        yield


def set_device(device: int):
    if device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    device = "cpu" if (not torch.cuda.is_available() or device == -1) else "cuda"
    torch.set_default_device(device)
    return device


def mean_and_ci(data, axis=-1, ci=0.95):
    n_sample = data.shape[axis]
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis, ddof=1) / np.sqrt(n_sample)
    h = std * stats.t.ppf((1 + ci) / 2, n_sample - 1)
    return mean, h


def plot_mean_and_ci(ax, x, y, ci=0.95, fill_between=True, **kwargs):
    # x: (n_data,)
    # y: (n_data, n_sample)
    assert x.ndim == 1
    assert y.ndim == 2
    assert y.shape[0] == x.shape[0]

    mean_y, conf_y = mean_and_ci(y, axis=1, ci=ci)  # (n_data,)

    line, = ax.plot(
        x,
        mean_y,
        *(v for k, v in kwargs.items() if k in {"fmt"}),
        **{k: v for k, v in kwargs.items() if k in
           {"color", "label", "linestyle", "linewidth", "markersize", "markerfacecolor", "markeredgecolor", "dashes"}},
    )

    if fill_between:
        ax.fill_between(
            x,
            mean_y - conf_y,
            mean_y + conf_y,
            color=line.get_color(),
            **{k: v for k, v in kwargs.items() if k in {"alpha"}},
        )


def setup_logger(name, level=logging.INFO):
    handler = logging.FileHandler(name, mode="a")        
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def l2_norm(x):
    x = torch.as_tensor(x).float()
    return torch.norm(x).item()

def linf_norm(x):
    x = torch.as_tensor(x).float()
    return torch.max(torch.abs(x)).item()

def l2_dist(x, y):
    return l2_norm(x - y)

def linf_dist(x, y):
    return linf_norm(x - y)

def l2_ratio(x, y):
    return l2_dist(x, y) / l2_norm(y)

def linf_ratio(x, y):
    return linf_dist(x, y) / linf_norm(y)

def l2_ratio_dict(x, y):
    if set(x.keys()) != set(y.keys()):
        raise ValueError("Keys of x and y must be the same")
    
    values_x = []
    values_y = []
    for key in x.keys():
        values_x.append(x[key].flatten())
        values_y.append(y[key].flatten())
    values_x = torch.cat(values_x)
    values_y = torch.cat(values_y)
    return l2_ratio(values_x, values_y)