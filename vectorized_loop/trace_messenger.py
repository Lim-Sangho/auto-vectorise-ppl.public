from __future__ import annotations
from pyro.poutine import trace_messenger
from pyro.poutine.trace_struct import Trace
from pyro.poutine.handlers import _make_handler
from pyro.poutine.subsample_messenger import _Subsample
from pyro.poutine.runtime import _PYRO_STACK
from typing import Callable, overload

from .runtime import _RUNTIME_MODE, _INDEX_STACK, setter
from .distributions import BranchDistribution

import sys
import torch


class TraceMessenger(trace_messenger.TraceMessenger):

    def __call__(self, fn):
        return TraceHandler(self, fn)

    def _pyro_post_sample(self, msg):
        if _RUNTIME_MODE["infer_shapes"] or self.param_only:
            return
        assert msg["name"] is not None
        assert msg["infer"] is not None
        if msg["infer"].get("_do_not_trace"):
            assert msg["infer"].get("is_auxiliary")
            assert not msg["is_observed"]
            return
        add_node(self.trace, msg["name"], **msg.copy())

    def _pyro_post_param(self, msg):
        assert msg["name"] is not None
        add_node(self.trace, msg["name"], **msg.copy())


class TraceHandler(trace_messenger.TraceHandler):
    
    def __call__(self, *args, **kwargs):
        with self.msngr:
            add_node(self.msngr.trace,
                "_INPUT", name="_INPUT", type="args", args=args, kwargs=kwargs
            )
            try:
                ret = self.fn(*args, **kwargs)
            except (ValueError, RuntimeError) as e:
                exc_type, exc_value, traceback = sys.exc_info()
                shapes = self.msngr.trace.format_shapes()
                assert exc_type is not None
                exc = exc_type("{}\n{}".format(exc_value, shapes))
                exc = exc.with_traceback(traceback)
                raise exc from e
            add_node(self.msngr.trace,
                "_RETURN", name="_RETURN", type="return", value=ret
            )
        return ret


def add_node(trace: Trace, site_name, **msg):
    if msg["type"] == "sample" and not isinstance(msg["fn"], _Subsample):
        site_shape = tuple(_INDEX_STACK._site_shape)

        if site_name not in trace:
            trace.nodes[site_name] = msg.copy()

            event_shape = msg["fn"].event_shape
            trace.nodes[site_name]["fn"] = BranchDistribution(site_shape, event_shape)
            trace.nodes[site_name]["value"] = torch.full(site_shape + event_shape, float("nan"))
            if msg["mask"] is not None:
                trace.nodes[site_name]["mask"] = torch.full(site_shape, True)

        site_index = _INDEX_STACK._site_index
        cond = setter(torch.full(site_shape, False), torch.tensor(True), (), site_index)
        trace.nodes[site_name]["fn"].add(torch.tensor(True) if torch.all(cond) else cond, msg["fn"])
        trace.nodes[site_name]["value"] = setter(trace.nodes[site_name]["value"], msg["value"], trace.nodes[site_name]["fn"].event_shape, site_index)

        if msg["mask"] is not None:
            trace.nodes[site_name]["mask"] = setter(trace.nodes[site_name]["mask"], msg["mask"], (), site_index)
        
    else:
        trace.nodes[site_name] = msg


# def add_node(trace: Trace, site_name, **msg):
#     if msg["type"] == "sample" and not isinstance(msg["fn"], _Subsample):
#         site_shape = tuple(_INDEX_STACK._site_shape)
#         site_index = tuple(_INDEX_STACK._site_index)
#         batch_shape = msg["fn"].batch_shape
#         event_shape = msg["fn"].event_shape

#         if site_name not in trace:
#             trace.nodes[site_name] = msg.copy()

#             expand_dist = msg["fn"].expand(torch.broadcast_shapes(batch_shape, site_shape))
#             base_dist = get_base_dist(expand_dist)
#             for arg in base_dist.arg_constraints:
#                 setattr(base_dist, arg, torch.full(getattr(base_dist, arg).shape, float("nan")))
#             trace.nodes[site_name]["fn"] = expand_dist
#             trace.nodes[site_name]["value"] = torch.full(site_shape + event_shape, float("nan"))
#             if msg["mask"] is not None:
#                 trace.nodes[site_name]["mask"] = torch.full(site_shape, True)

#         base_dist_new = get_base_dist(msg["fn"])
#         base_dist_old = get_base_dist(trace.nodes[site_name]["fn"])
#         for arg in base_dist_old.arg_constraints:
#             arg_shape = getattr(base_dist_old, arg).shape
#             setattr(base_dist_old, arg, setter(getattr(base_dist_old, arg), getattr(base_dist_new, arg), arg_shape, site_index))
#         trace.nodes[site_name]["value"] = setter(trace.nodes[site_name]["value"], msg["value"], event_shape, site_index)
#         if msg["mask"] is not None:
#             trace.nodes[site_name]["mask"] = setter(trace.nodes[site_name]["mask"], msg["mask"], (), site_index)

#     else:
#         trace.nodes[site_name] = msg


def get_base_dist(dist):
    """
    Get the base distribution of a distribution (e.g., Independent -> Normal, Beta -> Dirichlet)
    """
    base_dist = getattr(dist, "base_dist", dist)
    base_dist = getattr(base_dist, "_dirichlet", base_dist)
    return base_dist


def remove_sites(sites: set[str]):
    for messenger in _PYRO_STACK:
        if isinstance(messenger, TraceMessenger):
            for site in sites:
                messenger.trace.nodes.pop(site, None)


@overload
def trace(
    fn: None,
    *args,
    **kwargs,
) -> TraceMessenger: ...


@overload
def trace(
    fn: Callable,
    *args,
    **kwargs,
) -> TraceHandler: ...


@_make_handler(TraceMessenger)
def trace(fn): ...