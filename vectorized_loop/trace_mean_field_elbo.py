from pyro import infer
from pyro.infer.util import is_validation_enabled
from pyro.util import check_if_enumerated
from .trace_elbo import get_importance_trace


class TraceMeanField_ELBO(infer.TraceMeanField_ELBO):
    
    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, args, kwargs
        )
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
            infer.trace_mean_field_elbo._check_mean_field_requirement(model_trace, guide_trace)
        return model_trace, guide_trace