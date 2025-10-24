import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, "../"))
sys.path.append(os.path.join(path, "../../"))
sys.dont_write_bytecode = True

import pyro
import torch
import numpy as np
import pandas as pd
import pyro.distributions as dist

from pyro import poutine
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDelta

import vectorized_loop as vec
from vectorized_loop.distributions import NoClampCategorical as Categorical
from vectorized_loop.ops import Index, cat

from functools import partial
from util import time_it

from pyro.distributions.hmm import _sequential_logmatmulexp


def marginalize(log_prob_M, log_prob_D, log_prob_H, log_prob_obs, log_prob_probsMM, log_prob_probsMDD, log_prob_probsDHH, log_prob_locs, log_prob_scale):
    log_prob_M, log_prob_D, log_prob_H, log_prob_obs, log_prob_probsMM, log_prob_probsMDD, log_prob_probsDHH, log_prob_locs, log_prob_scale = \
        map(torch.Tensor.contiguous, (log_prob_M, log_prob_D, log_prob_H, log_prob_obs, log_prob_probsMM, log_prob_probsMDD, log_prob_probsDHH, log_prob_locs, log_prob_scale))
    
    log_prob_H_obs = log_prob_H + log_prob_obs                            # [_H, H, D, M, nH, nD, nM, nY]
    log_prob_H_obs = log_prob_H_obs.permute([2, 3, 5, 6, 7, 4, 0, 1])     # [D, M, nD, nM, nY, nH, _H, H]
    log_prob_H_obs = _sequential_logmatmulexp(log_prob_H_obs)             # [D, M, nD, nM, nY, _H, H]
    log_prob_H_obs = log_prob_H_obs[..., 0, :]                            # [D, M, nD, nM, nY, H]
    log_prob_H_obs = log_prob_H_obs.logsumexp(-1)                         # [D, M, nD, nM, nY]

    log_prob_DH_obs = log_prob_D.squeeze() + log_prob_H_obs           # [_D, D, M, nD, nM, nY]
    log_prob_DH_obs = log_prob_DH_obs.permute([2, 4, 5, 3, 0, 1])     # [M, nM, nY, nD, _D, D]
    log_prob_DH_obs = _sequential_logmatmulexp(log_prob_DH_obs)       # [M, nM, nY, _D, D]
    log_prob_DH_obs = log_prob_DH_obs[..., 0, :]                      # [M, nM, nY, D]
    log_prob_DH_obs = log_prob_DH_obs.logsumexp(-1)                   # [M, nM, nY]

    log_prob_MDH_obs = log_prob_M.squeeze() + log_prob_DH_obs      # [_M, M, nM, nY]
    log_prob_MDH_obs = log_prob_MDH_obs.permute([3, 2, 0, 1])      # [nY, nM, _M, M]
    log_prob_MDH_obs = _sequential_logmatmulexp(log_prob_MDH_obs)  # [nY, _M, M]
    log_prob_MDH_obs = log_prob_MDH_obs[..., 0, :]                 # [nY, M]
    log_prob_MDH_obs = log_prob_MDH_obs.logsumexp(-1)              # [nY]
    log_prob_MDH_obs = log_prob_MDH_obs.sum()                      # []

    loss = -(
        log_prob_MDH_obs
        + log_prob_probsMM
        + log_prob_probsMDD
        + log_prob_probsDHH
        + log_prob_locs
        + log_prob_scale
    )
    return loss


def svi_manual(model, guide, optim, *args, **kwargs):
    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = poutine.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = poutine.trace(poutine.replay(poutine.enum(model, first_available_dim=-5), tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()                       

        with time_it() as t_reduce:
                                                                                                                       # [_M _D _H  H  D  M | nH  nD  nM  nY]
            log_prob_M = tr_model.nodes["M"]["log_prob"][(None,) * 4].transpose(0, 5).transpose(4, 5)                  # [_M, 1, 1, 1, 1, M |  1,  1, nM, nY]
            log_prob_D = tr_model.nodes["D"]["log_prob"][(None,) * 1].transpose(3, 4).transpose(0, 2).transpose(1, 3)  #    [_D, 1, 1, D, M |  1, nD, nM, nY]
            log_prob_H = tr_model.nodes["H"]["log_prob"].transpose(2, 4).transpose(0, 3).transpose(1, 2)[0, 0]         #       [_H, H, D, 1 | nH, nD, nM, nY]
            log_prob_obs = tr_model.nodes["obs"]["log_prob"].transpose(4, 5).transpose(2, 4).transpose(0, 3)[0, 0, 0]  #           [H, D, M | nH, nD, nM, nY]

            log_prob_probsMM = tr_model.nodes["probsMM"]["log_prob"]
            log_prob_probsMDD = tr_model.nodes["probsMDD"]["log_prob"]
            log_prob_probsDHH = tr_model.nodes["probsDHH"]["log_prob"]
            log_prob_locs = tr_model.nodes["locs"]["log_prob"]
            log_prob_scale = tr_model.nodes["scale"]["log_prob"]

            loss = marginalize(log_prob_M, log_prob_D, log_prob_H, log_prob_obs, log_prob_probsMM, log_prob_probsMDD, log_prob_probsDHH, log_prob_locs, log_prob_scale)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_manual(data, hidden_dims, is_guide=False):
    hM, hD, hH = hidden_dims
    nY, nM, nD, nH = data.shape

    probsMM = pyro.sample("probsMM", dist.Dirichlet(torch.ones(hM)).expand((hM+1,)).to_event(1))  # (hM+1, hM)
    probsMDD = pyro.sample("probsMDD", dist.Dirichlet(torch.ones(hD)).expand((hM, hD+1)).to_event(2))  # (hM, hD+1, hD)
    probsDHH = pyro.sample("probsDHH", dist.Dirichlet(torch.ones(hH)).expand((hD, hH+1)).to_event(2))  # (hD, hH+1, hH)
    locs = pyro.sample("locs", dist.Normal(2, 1).expand((hM, hD, hH)).to_event(3))  # (hM, hD, hH)
    scale = pyro.sample("scale", dist.LogNormal(0, 1))  # ()

    if not is_guide:

        M_prev = pyro.sample( # (_M | 1, 1, 1, 1)
            "M",
            Categorical(logits=torch.zeros(hM)), # (hM,)
            infer={"enumerate": "parallel", "_do_not_trace": True, "is_auxiliary": True}
        )
        M_prev = cat([torch.ones(1, 1, dtype=torch.long) * -1, M_prev.repeat(1, 1, 1, nM-1, nY)], dim=-2) # (_M | 1, 1, nM, nY)
        M_curr = pyro.sample( # (M, 1 | 1, 1, 1, 1)
            "M",
            Categorical(logits=probsMM[M_prev].log()), # (_M | 1, 1, nM, nY | hM)
            infer={"enumerate": "parallel"}
        ) # log_prob: (M, _M | 1, 1, nM, nY)

        D_prev = pyro.sample( # (_D, 1, 1 | 1, 1, 1, 1)
            "D",
            Categorical(logits=torch.zeros(hD)), # (hD,)
            infer={"enumerate": "parallel", "_do_not_trace": True, "is_auxiliary": True}
        )
        D_prev = cat([torch.ones(1, 1, 1, dtype=torch.long) * -1, D_prev.repeat(1, 1, 1, 1, nD-1, nM, nY)], dim=-3) # (_D, 1, 1 | 1, nD, nM, nY)
        D_curr = pyro.sample( # (D, 1, 1, 1 | 1, 1, 1, 1)
            "D",
            Categorical(logits=probsMDD[M_curr, D_prev].log()), # (_D, M, 1 | 1, nD, nM, nY | hD)
            infer={"enumerate": "parallel"}
        ) # log_prob: (D, _D, M, 1 | 1, nD, nM, nY)

        H_prev = pyro.sample( # (_H, 1, 1, 1, 1 | 1, 1, 1, 1)
            "H",
            Categorical(logits=torch.zeros(hH)), # (hH,)
            infer={"enumerate": "parallel", "_do_not_trace": True, "is_auxiliary": True}
        )
        H_prev = cat([torch.ones(1, 1, 1, 1, dtype=torch.long) * -1, H_prev.repeat(1, 1, 1, 1, 1, nH-1, nD, nM, nY)], dim=-4) # (_H, 1, 1, 1, 1 | nH, nD, nM, nY)
        H_curr = pyro.sample( # (H, 1, 1, 1, 1, 1 | 1, 1, 1, 1)
            "H",
            Categorical(logits=probsDHH[D_curr, H_prev].log()), # (_H, D, 1, 1, 1 | nH, nD, nM, nY | hH)
            infer={"enumerate": "parallel"}
        ) # log_prob: (H, _H, D, 1, 1, 1 | nH, nD, nM, nY)

        pyro.sample(
            "obs",
            dist.Normal(locs[M_curr, D_curr, H_curr], scale), # (H, 1, D, 1, M, 1 | 1, 1, 1, 1)
            obs=data.permute([3, 2, 1, 0]), # (nH, nD, nM, nY)
        ) # log_prob: (H, 1, D, 1, M, 1 | nH, nD, nM, nY)


def svi_seq(model, guide, optim, *args, **kwargs):
    nY, nM, nD, nH = args[0].shape

    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = poutine.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = poutine.trace(poutine.replay(poutine.enum(model, first_available_dim=-1), tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()                       

        with time_it() as t_reduce:
            log_prob_M_0 = torch.stack([tr_model.nodes[f"M_{i}_0"]["log_prob"]
                for i in range(nY)], dim=-1)  # (M_0 | nY)
            log_prob_M_1_12 = torch.stack([torch.stack([tr_model.nodes[f"M_{i}_{j}"]["log_prob"].permute([0, 1] if j % 2 == 0 else [1, 0])
                for j in range(1, nM)], dim=-1)
                for i in range(nY)], dim=-1)  # (_M, M | nM-1, nY)
            log_prob_D_0 = torch.stack([torch.stack([tr_model.nodes[f"D_{i}_{j}_0"]["log_prob"].squeeze()
                for j in range(nM)], dim=-1)
                for i in range(nY)], dim=-1)  # (D_0, M | nM, nY)
            log_prob_D_1_31 = torch.stack([torch.stack([torch.stack([tr_model.nodes[f"D_{i}_{j}_{k}"]["log_prob"].squeeze().permute([0, 1, 2] if k % 2 == 0 else [1, 0, 2])
                for k in range(1, nD)], dim=-1)
                for j in range(nM)], dim=-1)
                for i in range(nY)], dim=-1)  # (_D, D, M | nD-1, nM, nY)
            log_prob_H_0 = torch.stack([torch.stack([torch.stack([tr_model.nodes[f"H_{i}_{j}_{k}_0"]["log_prob"].squeeze()
                for k in range(nD)], dim=-1)
                for j in range(nM)], dim=-1) 
                for i in range(nY)], dim=-1)  # (H, D | nD, nM, nY)
            log_prob_H_1_24 = torch.stack([torch.stack([torch.stack([torch.stack([tr_model.nodes[f"H_{i}_{j}_{k}_{l}"]["log_prob"].squeeze().permute([0, 1, 2] if l % 2 == 0 else [1, 0, 2])
                for l in range(1, nH)], dim=-1)
                for k in range(nD)], dim=-1)
                for j in range(nM)], dim=-1)
                for i in range(nY)], dim=-1)  # (_H, H, D | nH-1, nD, nM, nY)
            log_prob_obs = torch.stack([torch.stack([torch.stack([torch.stack([tr_model.nodes[f"obs_{i}_{j}_{k}_{l}"]["log_prob"].squeeze()
                for l in range(nH)], dim=-1)
                for k in range(nD)], dim=-1)
                for j in range(nM)], dim=-1)
                for i in range(nY)], dim=-1)  # (H, D, M | nH, nD, nM, nY)

                                                                                         # [_M _D _H  H  D  M | nH    nD    nM    nY]
            log_prob_M_0 = log_prob_M_0[:, None, None, None]                             # [                M | 1,    1,     1,   nY]
            log_prob_M_1_12 = log_prob_M_1_12[:, None, None, None, None, :, None, None]  # [_M, 1, 1, 1, 1, M | 1,    1,  nM-1,   nY]
            log_prob_D_0 = log_prob_D_0[:, :, None, None]                                # [             D, M | 1,    1,    nM,   nY]
            log_prob_D_1_31 = log_prob_D_1_31[:, None, None, :, :, None]                 # [   _D, 1, 1, D, M | 1,    nD-1, nM,   nY]
            log_prob_H_0 = log_prob_H_0[:, :, None, None]                                # [          H, D, 1 | 1,    nD,   nM,   nY]
            log_prob_H_1_24 = log_prob_H_1_24[:, :, :, None]                             # [      _H, H, D, 1 | nH-1, nD,   nM,   nY]

            log_prob_M = cat([log_prob_M_0, log_prob_M_1_12], dim=-2)   # [_M, 1, 1, 1, 1, M |  1,  1, nM, nY]
            log_prob_D = cat([log_prob_D_0, log_prob_D_1_31], dim=-3)   # [   _D, 1  1, D, M |  1, nD, nM, nY]
            log_prob_H = cat([log_prob_H_0, log_prob_H_1_24], dim=-4)   # [      _H, H, D, 1 | nH, nD, nM, nY]
            log_prob_obs = log_prob_obs                                 # [          H, D, M | nH, nD, nM, nY)

            log_prob_probsMM = tr_model.nodes["probsMM"]["log_prob"]
            log_prob_probsMDD = tr_model.nodes["probsMDD"]["log_prob"]
            log_prob_probsDHH = tr_model.nodes["probsDHH"]["log_prob"]
            log_prob_locs = tr_model.nodes["locs"]["log_prob"]
            log_prob_scale = tr_model.nodes["scale"]["log_prob"]

            loss = marginalize(log_prob_M, log_prob_D, log_prob_H, log_prob_obs, log_prob_probsMM, log_prob_probsMDD, log_prob_probsDHH, log_prob_locs, log_prob_scale)
    
        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_seq(data, hidden_dims, is_guide=False):
    hM, hD, hH = hidden_dims
    nY, nM, nD, nH = data.shape

    probsMM = pyro.sample("probsMM", dist.Dirichlet(torch.ones(hM)).expand((hM+1,)).to_event(1))  # (hM+1, hM)
    probsMDD = pyro.sample("probsMDD", dist.Dirichlet(torch.ones(hD)).expand((hM, hD+1)).to_event(2))  # (hM, hD+1, hD)
    probsDHH = pyro.sample("probsDHH", dist.Dirichlet(torch.ones(hH)).expand((hD, hH+1)).to_event(2))  # (hD, hH+1, hH)
    locs = pyro.sample("locs", dist.Normal(2, 1).expand((hM, hD, hH)).to_event(3))  # (hM, hD, hH)
    scale = pyro.sample("scale", dist.LogNormal(0, 1))  # ()

    if not is_guide:

        for i in range(nY):
            M = -1
            for j in pyro.markov(range(nM)):
                M = pyro.sample(
                    f"M_{i}_{j}",
                    Categorical(logits=probsMM[M].log()),
                    infer={"enumerate": "parallel"},
                )
                D = -1
                for k in pyro.markov(range(nD)):
                    D = pyro.sample(
                        f"D_{i}_{j}_{k}",
                        Categorical(logits=probsMDD[M, D].log()),
                        infer={"enumerate": "parallel"},
                    )
                    H = -1
                    for l in pyro.markov(range(nH)):
                        H = pyro.sample(
                            f"H_{i}_{j}_{k}_{l}",
                            Categorical(logits=probsDHH[D, H].log()),
                            infer={"enumerate": "parallel"}
                        )
                        pyro.sample(
                            f"obs_{i}_{j}_{k}_{l}",
                            dist.Normal(locs[M, D, H], scale),
                            obs=data[i, j, k, l],
                        )


def svi_plate(model, guide, optim, *args, **kwargs):
    nY, nM, nD, nH = args[0].shape

    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = poutine.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = poutine.trace(poutine.replay(poutine.enum(model, first_available_dim=-2), tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()                       

        with time_it() as t_reduce:
            log_prob_M_0 = tr_model.nodes["M_0"]["log_prob"]  # (M_0 | nY)
            log_prob_M_1_12 = torch.stack([tr_model.nodes[f"M_{j}"]["log_prob"].permute([0, 1, 2] if j % 2 == 0 else [1, 0, 2])
                for j in range(1, nM)]).permute([1, 2, 0, 3])  # (_M, M | nM-1, nY)
            log_prob_D_0 = torch.stack([tr_model.nodes[f"D_{j}_0"]["log_prob"].squeeze()
                for j in range(nM)]).permute([1, 2, 0, 3])  # (D_0, M | nM, nY)
            log_prob_D_1_31 = torch.stack([torch.stack([tr_model.nodes[f"D_{j}_{k}"]["log_prob"].squeeze().permute([0, 1, 2, 3] if k % 2 == 0 else [1, 0, 2, 3])
                for k in range(1, nD)])
                for j in range(nM)]).permute([2, 3, 4, 1, 0, 5])  # (_D, D, M | nD-1, nM, nY)
            log_prob_H_0 = torch.stack([torch.stack([tr_model.nodes[f"H_{j}_{k}_0"]["log_prob"].squeeze()
                for k in range(nD)])
                for j in range(nM)]).permute([2, 3, 1, 0, 4])  # (H, D | nD, nM, nY)
            log_prob_H_1_24 = torch.stack([torch.stack([torch.stack([tr_model.nodes[f"H_{j}_{k}_{l}"]["log_prob"].squeeze().permute([0, 1, 2, 3] if l % 2 == 0 else [1, 0, 2, 3])
                for l in range(1, nH)])
                for k in range(nD)])
                for j in range(nM)]).permute([3, 4, 5, 2, 1, 0, 6]) # (_H, H, D | nH-1, nD, nM, nY)
            log_prob_obs = torch.stack([torch.stack([torch.stack([tr_model.nodes[f"obs_{j}_{k}_{l}"]["log_prob"].squeeze()
                for l in range(nH)])
                for k in range(nD)])
                for j in range(nM)]).permute([3, 4, 5, 2, 1, 0, 6])  # (H, D, M | nH, nD, nM, nY)

                                                                                         # [_M _D _H  H  D  M | nH    nD    nM    nY]
            log_prob_M_0 = log_prob_M_0[:, None, None, None]                             # [                M | 1,    1,     1,   nY]
            log_prob_M_1_12 = log_prob_M_1_12[:, None, None, None, None, :, None, None]  # [_M, 1, 1, 1, 1, M | 1,    1,  nM-1,   nY]
            log_prob_D_0 = log_prob_D_0[:, :, None, None]                                # [             D, M | 1,    1,    nM,   nY]
            log_prob_D_1_31 = log_prob_D_1_31[:, None, None, :, :, None]                 # [   _D, 1, 1, D, M | 1,    nD-1, nM,   nY]
            log_prob_H_0 = log_prob_H_0[:, :, None, None]                                # [          H, D, 1 | 1,    nD,   nM,   nY]
            log_prob_H_1_24 = log_prob_H_1_24[:, :, :, None]                             # [      _H, H, D, 1 | nH-1, nD,   nM,   nY]

            log_prob_M = cat([log_prob_M_0, log_prob_M_1_12], dim=-2)   # [_M, 1, 1, 1, 1, M |  1,  1, nM, nY]
            log_prob_D = cat([log_prob_D_0, log_prob_D_1_31], dim=-3)   # [   _D, 1  1, D, M |  1, nD, nM, nY]
            log_prob_H = cat([log_prob_H_0, log_prob_H_1_24], dim=-4)   # [      _H, H, D, 1 | nH, nD, nM, nY]
            log_prob_obs = log_prob_obs                                 # [          H, D, M | nH, nD, nM, nY)

            log_prob_probsMM = tr_model.nodes["probsMM"]["log_prob"]
            log_prob_probsMDD = tr_model.nodes["probsMDD"]["log_prob"]
            log_prob_probsDHH = tr_model.nodes["probsDHH"]["log_prob"]
            log_prob_locs = tr_model.nodes["locs"]["log_prob"]
            log_prob_scale = tr_model.nodes["scale"]["log_prob"]

            loss = marginalize(log_prob_M, log_prob_D, log_prob_H, log_prob_obs, log_prob_probsMM, log_prob_probsMDD, log_prob_probsDHH, log_prob_locs, log_prob_scale)
    
        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_plate(data, hidden_dims, is_guide=False):
    hM, hD, hH = hidden_dims
    nY, nM, nD, nH = data.shape

    probsMM = pyro.sample("probsMM", dist.Dirichlet(torch.ones(hM)).expand((hM+1,)).to_event(1))  # (hM+1, hM)
    probsMDD = pyro.sample("probsMDD", dist.Dirichlet(torch.ones(hD)).expand((hM, hD+1)).to_event(2))  # (hM, hD+1, hD)
    probsDHH = pyro.sample("probsDHH", dist.Dirichlet(torch.ones(hH)).expand((hD, hH+1)).to_event(2))  # (hD, hH+1, hH)
    locs = pyro.sample("locs", dist.Normal(2, 1).expand((hM, hD, hH)).to_event(3))  # (hM, hD, hH)
    scale = pyro.sample("scale", dist.LogNormal(0, 1))  # ()

    if not is_guide:

        with pyro.plate("years", nY, dim=-1) as i:
            M = -1
            for j in pyro.markov(range(nM)):
                M = pyro.sample(
                    f"M_{j}",
                    Categorical(logits=probsMM[M].log()),
                    infer={"enumerate": "parallel"},
                )
                D = -1
                for k in pyro.markov(range(nD)):
                    D = pyro.sample(
                        f"D_{j}_{k}",
                        Categorical(logits=probsMDD[M, D].log()),
                        infer={"enumerate": "parallel"},
                    )
                    H = -1
                    for l in pyro.markov(range(nH)):
                        H = pyro.sample(
                            f"H_{j}_{k}_{l}",
                            Categorical(logits=probsDHH[D, H].log()),
                            infer={"enumerate": "parallel"}
                        )
                        pyro.sample(
                            f"obs_{j}_{k}_{l}",
                            dist.Normal(locs[M, D, H], scale),
                            obs=data[i, j, k, l],
                        )


svi_vmarkov = None
model_vmarkov = None


svi_discHMM = None
model_discHMM = None


def svi_ours(model, guide, optim, *args, **kwargs):
    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = vec.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = vec.trace(vec.replay(vec.enum(model), tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()                       

        with time_it() as t_reduce:
                                                              # [_M _D _H  H  D  M  nH  nD  nM  nY]
            log_prob_M = tr_model.nodes["M"]["log_prob"]      # [_M, 1, 1, 1, 1, M,  1,  1, nM, nY]
            log_prob_D = tr_model.nodes["D"]["log_prob"]      #    [_D, 1, 1, D, M,  1, nD, nM, nY]
            log_prob_H = tr_model.nodes["H"]["log_prob"]      #       [_H, H, D, 1, nH, nD, nM, nY]
            log_prob_obs = tr_model.nodes["obs"]["log_prob"]  #           [H, D, M, nH, nD, nM, nY]

            log_prob_probsMM = tr_model.nodes["probsMM"]["log_prob"]
            log_prob_probsMDD = tr_model.nodes["probsMDD"]["log_prob"]
            log_prob_probsDHH = tr_model.nodes["probsDHH"]["log_prob"]
            log_prob_locs = tr_model.nodes["locs"]["log_prob"]
            log_prob_scale = tr_model.nodes["scale"]["log_prob"]

            loss = marginalize(log_prob_M, log_prob_D, log_prob_H, log_prob_obs, log_prob_probsMM, log_prob_probsMDD, log_prob_probsDHH, log_prob_locs, log_prob_scale)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


@vec.vectorize
def model_ours(s: vec.State, data, hidden_dims, is_guide=False):
    hM, hD, hH = hidden_dims
    nY, nM, nD, nH = data.shape

    probsMM = pyro.sample("probsMM", dist.Dirichlet(torch.ones(hM)).expand((hM+1,)).to_event(1))  # (hM+1, hM)
    probsMDD = pyro.sample("probsMDD", dist.Dirichlet(torch.ones(hD)).expand((hM, hD+1)).to_event(2))  # (hM, hD+1, hD)
    probsDHH = pyro.sample("probsDHH", dist.Dirichlet(torch.ones(hH)).expand((hD, hH+1)).to_event(2))  # (hD, hH+1, hH)
    locs = pyro.sample("locs", dist.Normal(2, 1).expand((hM, hD, hH)).to_event(3))  # (hM, hD, hH)
    scale = pyro.sample("scale", dist.LogNormal(0, 1))  # ()

    if not is_guide:

        for s.i in vec.range("years", nY, vectorized=True):
            s.M = -1
            for s.j in vec.range("months", nM, vectorized=True):
                s.M = pyro.sample(
                    "M",
                    Categorical(logits=Index(probsMM)[s.M].log()),
                    infer={"enumerate": "parallel"},
                )
                s.D = -1
                for s.k in vec.range("days", nD, vectorized=True):
                    s.D = pyro.sample(
                        "D",
                        Categorical(logits=Index(probsMDD)[s.M, s.D].log()),
                        infer={"enumerate": "parallel"},
                    )
                    s.H = -1
                    for s.l in vec.range("hours", nH, vectorized=True):
                        s.H = pyro.sample(
                            "H",
                            Categorical(logits=Index(probsDHH)[s.D, s.H].log()),
                            infer={"enumerate": "parallel"}
                        )
                        pyro.sample(
                            "obs",
                            dist.Normal(Index(locs)[s.M, s.D, s.H], scale),
                            obs=Index(data)[s.i, s.j, s.k, s.l],
                        )


def guide(model, data, hidden_dims):
    hM, hD, hH = hidden_dims
    init_locs = {
        "probsMM": dist.Dirichlet(torch.ones(hM)).expand((hM+1,)).sample(),
        "probsMDD": dist.Dirichlet(torch.ones(hD)).expand((hM, hD+1)).sample(),
        "probsDHH": dist.Dirichlet(torch.ones(hH)).expand((hD, hH+1)).sample(),
        "locs": dist.Normal(2, 1).expand((hM, hD, hH)).sample(),
        "scale": dist.LogNormal(0, 1).sample(),
    }

    _guide = AutoDelta(
        partial(model, is_guide=True),
        init_loc_fn=lambda msg: init_locs[msg["name"]]
    )

    return _guide


def data(args):
    df = pd.read_csv(os.path.join(path, "dataset/train_to_UCTY.txt"), index_col="Date", parse_dates=True)
    df = df[df.index.year != 2020]
    df = df[df.index.year != 2019]
    df = df[df.index.year != 2018]
    df = df[df.index.year != 2017]
    df = df[df.index.year != 2016]

    data_list = []
    for _, group in df.groupby([df.index.year, df.index.month]):
        data_list.append(group["Trip Count"].to_numpy())

    max_hours_in_a_month = max([len(month) for month in data_list])
    data = np.array([np.pad(month, (0, max_hours_in_a_month - len(month))) for month in data_list])
    data = torch.as_tensor(data, device=torch.get_default_device()).reshape(5, 12, 31, 24).log1p().to(torch.float32)
    data = data[:args.num_batch]
    hidden_dims = (2, 4, 16)
    return data, hidden_dims


def optim(args):
    return Adam({"lr": args.lr})