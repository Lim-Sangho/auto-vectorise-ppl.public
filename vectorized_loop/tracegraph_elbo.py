from pyro import infer
from pyro.infer.tracegraph_elbo import TrackNonReparam
from pyro.infer.util import is_validation_enabled
from pyro.util import check_if_enumerated
from .trace_elbo import get_importance_trace

class TraceGraph_ELBO(infer.TraceGraph_ELBO):

    def _get_trace(self, model, guide, args, kwargs):
        with TrackNonReparam():
            model_trace, guide_trace = get_importance_trace(
                "flat", self.max_plate_nesting, model, guide, args, kwargs
            )
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace