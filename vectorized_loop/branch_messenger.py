from __future__ import annotations
from .runtime import _RUNTIME_MODE, _BRANCH_STACK, _VAR_STACK, _ENUM_ALLOCATOR
import torch


class BranchMessenger:
    """
    Several restriction rules are applied to the condition and the branch body.
    1) The condition must not contain any enum dimensions. It only supports for the conditions with batch dimensions.
    2) For every variable `x` with enum dimensions, if `x` is written in a certain branch body,
       then `x` must not be used as a free variable in the other branch bodies.

    For example, the following model may result in a wrong inference:

    @vec.vectorize
    def model(s: vec.State):
        P = torch.rand(5, 5)
        s.x = 0
        for s.i in markov("markov", 10, vectorized=True):
            with branch(s.i <= 4):
                s.x = pyro.sample("x", dist.Categorical(Index(P)[s.x]), infer={"enumerate": "parallel"})
            with branch(s.i >= 5):
                s.x = pyro.sample("x", dist.Categorical(Index(P)[4 - s.x]), infer={"enumerate": "parallel"})

    because the variable `x` is used as a free variable in the other branch body, violating the rule 2).
    Instead, the following equivalent model avoids the violation:

    @vec.vectorize
    def model(s: vec.State):
        P = torch.rand(5, 5)
        s.x = 0
        for s.i in markov("markov", 10, vectorized=True):
            s.x_prev = s.x
            with branch(s.i <= 4):
                s.x = pyro.sample("x", dist.Categorical(Index(P)[s.x_prev]), infer={"enumerate": "parallel"})
            with branch(s.i >= 5):
                s.x = pyro.sample("x", dist.Categorical(Index(P)[4 - s.x_prev]), infer={"enumerate": "parallel"})

    because the variable `x` is not used as a free variable in the other branch body.
    """
    
    def __init__(self, cond: torch.BoolTensor):
        self.cond = torch.as_tensor(cond, dtype=torch.bool)

    def __enter__(self, *args):
        _BRANCH_STACK.push(self.cond)
        _VAR_STACK.push()
        return self

    def __exit__(self, *args):
        _BRANCH_STACK.pop()
        _VAR_STACK.pop()


def branch(cond):
    cond = torch.as_tensor(cond, dtype=torch.bool)
    if _RUNTIME_MODE["infer_shapes"] and cond.numel() != 1:
        raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")
    if not _RUNTIME_MODE["infer_shapes"] and cond.dim() > -_ENUM_ALLOCATOR._min_batch_dim:
        raise NotImplementedError("Branch messenger does not support enumerated conditions yet")
    if _RUNTIME_MODE["infer_shapes"] or torch.all(cond):
        cond = torch.tensor(True)
    if torch.all(~cond):
        cond = torch.tensor(False)
    return BranchMessenger(cond)