from __future__ import annotations
import torch
import numpy as np
import pyro.distributions as dist
from pyro.distributions import Distribution


class NoClampCategorical(dist.Categorical):
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        if "logits" not in self.__dict__:
            self.logits = self.probs.log()
        value = value.long().unsqueeze(-1)
        value, logits = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return logits.gather(-1, value).squeeze(-1)


class NoClampBernoulli(dist.Bernoulli):
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        if "logits" not in self.__dict__:
            self.logits = self.probs.log()
        value, logits = torch.broadcast_tensors(value, self.logits)
        return -torch.nn.functional.binary_cross_entropy_with_logits(logits, value, reduction="none")
    

class BranchDistribution(Distribution):
    """
    A distribution over multiple branches and distributions.
    """

    def __init__(self, site_shape: tuple[int], event_shape):
        self.site_shape = site_shape
        self.event_shape = event_shape
        self.conds = []
        self.dists = []

    def __repr__(self):
        return f"BranchDistribution(site_shape={self.site_shape}, event_shape={self.event_shape}, conds={self.conds}, dists={self.dists})"

    def sample(self):
        raise NotImplementedError

    def log_prob(self, value: torch.Tensor):
        result = torch.zeros(self.site_shape)
        for cond, dist in zip(self.conds, self.dists):
            log_prob = dist.log_prob(value)
            result = torch.where(cond, log_prob, result)
            # result += torch.where(cond, log_prob, 0)
        return result

    def add(self, cond: torch.BoolTensor, dist: Distribution):
        if dist.event_shape != self.event_shape:
            raise ValueError(f"Expected event_shape {self.event_shape}, but got {dist.event_shape}")
        self.conds.append(cond)
        self.dists.append(dist)

    @property
    def has_rsample(self) -> bool:
        return np.all([dist.has_rsample for dist in self.dists])

    @property
    def batch_shape(self) -> tuple[int]:
        return self.site_shape

    @property
    def batch_dim(self) -> int:
        return len(self.site_shape)
    
    @property
    def event_dim(self) -> int:
        return len(self.event_shape)
