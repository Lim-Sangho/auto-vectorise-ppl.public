import sys
sys.dont_write_bytecode = True

import pyro
assert pyro.__version__.startswith("1.9.1")
pyro.enable_validation(False)

from . import distributions
from .state import State, vectorize
from .vectorized_loop_messenger import VectorizedLoopMessenger, range
from .trace_messenger import TraceMessenger, trace
from .replay_messenger import ReplayMessenger, replay
from .condition_messenger import ConditionMessenger, condition
from .branch_messenger import BranchMessenger, branch
from .enum_messenger import EnumMessenger, enum
from .trace_elbo import Trace_ELBO
from .tracegraph_elbo import TraceGraph_ELBO
from .trace_mean_field_elbo import TraceMeanField_ELBO
from .runtime import (_RUNTIME_MODE, _ENV, _BATCH_SHAPES, _EVENT_SHAPES,
                      _DIM_ALLOCATOR, _ENUM_ALLOCATOR, _INDEX_STACK, _VAR_STACK,
                      clear_env, clear_allocators)

__all__ = ['vectorize',
           'State',
           'distributions',
           '_RUNTIME_MODE',
           '_ENV',
           '_BATCH_SHAPES',
           '_EVENT_SHAPES',
           '_DIM_ALLOCATOR',
           '_ENUM_ALLOCATOR',
           '_INDEX_STACK',
           '_VAR_STACK',
           '_BRANCH_STACK',
           'clear_env',
           'clear_allocators',
           'VectorizedLoopMessenger',
           'range',
           'TraceMessenger',
           'trace',
           'ReplayMessenger',
           'replay',
           'ConditionMessenger',
           'condition',
           'BranchMessenger',
           'branch',
           'EnumMessenger',
           'enum',
           'Trace_ELBO',
           'TraceGraph_ELBO',
           'TraceMeanField_ELBO',
           ]