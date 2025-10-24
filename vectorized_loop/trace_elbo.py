from pyro import infer
from pyro.infer.util import is_validation_enabled
from pyro.poutine.util import prune_subsample_sites

from pyro.infer.util import (
    is_validation_enabled,
)
from pyro.util import check_if_enumerated

from .trace_messenger import trace
from .replay_messenger import replay


class Trace_ELBO(infer.Trace_ELBO):
    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, args, kwargs
        )
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace


def get_importance_trace(
    graph_type, max_plate_nesting, model, guide, args, kwargs
):
    guide_trace = trace(guide, graph_type=graph_type).get_trace(
        *args, **kwargs
    )
    model_trace = trace(
        replay(model, trace=guide_trace), graph_type=graph_type
    ).get_trace(*args, **kwargs)

    guide_trace = prune_subsample_sites(guide_trace)
    model_trace = prune_subsample_sites(model_trace)

    model_trace.compute_log_prob()
    guide_trace.compute_score_parts()
    
    return model_trace, guide_trace