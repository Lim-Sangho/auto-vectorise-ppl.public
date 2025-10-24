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

import vectorized_loop as vec
from vectorized_loop.distributions import NoClampCategorical as Categorical
from vectorized_loop.ops import Index, cat

from util import time_it

from pyro.distributions.hmm import _sequential_logmatmulexp

def add_node(self, site_name, **kwargs):
    self.nodes[site_name] = kwargs
    self._pred[site_name] = set()
    self._succ[site_name] = set()

def _pyro_post_sample(self, msg):
    if self.param_only:
        return
    if msg["infer"].get("_do_not_trace"):
        return
    self.trace.add_node(msg["name"], **msg.copy())

pyro.poutine.trace_struct.Trace.add_node = add_node
pyro.poutine.trace_messenger.TraceMessenger._pyro_post_sample = _pyro_post_sample


def marginalize(log_prob_Y, log_prob_Q, log_prob_log_h, log_prob_obs, log_prob_probsYY, log_prob_probsYQQ, log_prob_alpha, log_prob_phi, log_prob_scale):
    log_prob_Y, log_prob_Q, log_prob_log_h, log_prob_obs, log_prob_probsYY, log_prob_probsYQQ, log_prob_alpha, log_prob_phi, log_prob_scale = \
        map(torch.Tensor.contiguous, (log_prob_Y, log_prob_Q, log_prob_log_h, log_prob_obs, log_prob_probsYY, log_prob_probsYQQ, log_prob_alpha, log_prob_phi, log_prob_scale))

    log_prob_log_h_obs = log_prob_log_h + log_prob_obs  # [Q, Y, nD, nQ, nY, nB]
    log_prob_log_h_obs = log_prob_log_h_obs.sum(-4)     # [Q, Y, nQ, nY, nB]

    log_prob_Q_log_h_obs = log_prob_Q.squeeze() + log_prob_log_h_obs         # [_Q, Q, Y, nQ, nY, nB]
    log_prob_Q_log_h_obs = log_prob_Q_log_h_obs.permute([2, 4, 5, 3, 0, 1])  # [Y, nY, nB, nQ, _Q, Q]
    log_prob_Q_log_h_obs = _sequential_logmatmulexp(log_prob_Q_log_h_obs)    # [Y, nY, nB, _Q, Q]
    log_prob_Q_log_h_obs = log_prob_Q_log_h_obs[..., 0, :]                   # [Y, nY, nB, Q]
    log_prob_Q_log_h_obs = log_prob_Q_log_h_obs.logsumexp(-1)                # [Y, nY, nB]

    log_prob_YQ_log_h_obs = log_prob_Y.squeeze() + log_prob_Q_log_h_obs      # [_Y, Y, nY, nB]
    log_prob_YQ_log_h_obs = log_prob_YQ_log_h_obs.permute([3, 2, 0, 1])      # [nB, nY, _Y, Y]
    log_prob_YQ_log_h_obs = _sequential_logmatmulexp(log_prob_YQ_log_h_obs)  # [nB, _Y, Y]
    log_prob_YQ_log_h_obs = log_prob_YQ_log_h_obs[..., 0, :]                 # [nB, Y]
    log_prob_YQ_log_h_obs = log_prob_YQ_log_h_obs.logsumexp(-1)              # [nB]
    log_prob_YQ_log_h_obs = log_prob_YQ_log_h_obs.sum(-1)                    # []

    loss = -(
        log_prob_YQ_log_h_obs
        + log_prob_probsYY
        + log_prob_probsYQQ
        + log_prob_alpha
        + log_prob_phi
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
                                                                                                       # [_Y, _Q,  Q,  Y | nD, nQ, nY, nB]
            log_prob_Y = tr_model.nodes["Y"]["log_prob"][(None,) * 2].transpose(0, 3).transpose(2, 3)  # [ 4,  1,  1,  4 |  1,  1, 10, 10]
            log_prob_Q = tr_model.nodes["Q"]["log_prob"].transpose(2, 3).transpose(0, 2)[0]            # [     4,  4,  4 |  1,  4, 10, 10]
            log_prob_log_h = tr_model.nodes["log_h"]["log_prob"]                                       # [               | 64,  4, 10, 10]
            log_prob_obs = tr_model.nodes["obs"]["log_prob"].transpose(2, 3).transpose(0, 2)[0, 0]     # [         4,  4 | 64,  4, 10, 10]

            log_prob_probsYY = tr_model.nodes["probsYY"]["log_prob"]    # []
            log_prob_probsYQQ = tr_model.nodes["probsYQQ"]["log_prob"]  # []
            log_prob_alpha = tr_model.nodes["alpha"]["log_prob"]        # []
            log_prob_phi = tr_model.nodes["phi"]["log_prob"]            # []
            log_prob_scale = tr_model.nodes["scale"]["log_prob"]        # []

            loss = marginalize(log_prob_Y, log_prob_Q, log_prob_log_h, log_prob_obs, log_prob_probsYY, log_prob_probsYQQ, log_prob_alpha, log_prob_phi, log_prob_scale)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_manual(data, hidden_dims):
    hY, hQ = hidden_dims
    nB, nY, nQ, nD = data.shape

    probsYY = pyro.sample("probsYY", dist.Dirichlet(torch.ones(hY)).expand((hY+1,)).to_event(1))  # (hY+1, hY)
    probsYQQ = pyro.sample("probsYQQ", dist.Dirichlet(torch.ones(hQ)).expand((hY, hQ+1)).to_event(2))  # (hY, hQ+1, hQ)

    avg_return = 8 ** (1 / (nY * nQ * nD)) - 1
    alpha = pyro.sample("alpha", dist.Normal(0, 1e-2).expand((hY, hQ)).to_event(2))  # (hY, hQ)
    phi = pyro.sample("phi", dist.Normal(1, 1e-2))  # ()
    scale = pyro.sample("scale", dist.LogNormal(-1, 1e-2))  # ()

    Y_prev = pyro.sample( # (_Y | 1, 1, 1, 1)
        "Y",
        Categorical(logits=torch.zeros(hY)), # (hY,)
        infer={"enumerate": "parallel", "_do_not_trace": True, "is_auxiliary": True}
    )
    Y_prev = cat([torch.ones(1, 1, dtype=torch.long) * -1, Y_prev.repeat(1, 1, 1, nY-1, nB)], dim=-2) # (_Y | 1, 1, nY, nB)
    Y_curr = pyro.sample( # (Y, 1 | 1, 1, 1, 1)
        "Y",
        Categorical(logits=probsYY[Y_prev].log()), # (_Y | 1, 1, nY, nB | hY)
        infer={"enumerate": "parallel"},
    ) # log_prob: (Y, _Y | 1, 1, nY, nB)

    Q_prev = pyro.sample( # (_Q, 1, 1 | 1, 1, 1, 1)
        "Q",
        Categorical(logits=torch.zeros(hQ)), # (hQ,)
        infer={"enumerate": "parallel", "_do_not_trace": True, "is_auxiliary": True}
    )
    Q_prev = cat([torch.ones(1, 1, 1, dtype=torch.long) * -1, Q_prev.repeat(1, 1, 1, 1, nQ-1, nY, nB)], dim=-3) # (_Q, 1, 1 | 1, nQ, nY, nB)
    Q_curr = pyro.sample( # (Q, 1, 1, 1 | 1, 1, 1, 1)
        "Q",
        Categorical(logits=probsYQQ[Y_curr, Q_prev].log()), # (_Q, Y, 1 | 1, nQ, nY, nB | hQ)
        infer={"enumerate": "parallel"},
    ) # log_prob: (Q, _Q, Y, 1 | 1, nQ, nY, nB)

    log_h_prev = pyro.sample("log_h", dist.Normal(0, 1),
                             infer={"_do_not_trace": True, "is_auxiliary": True}) # (nD, nQ, nY, nB)
    log_h_prev = cat([torch.ones(1, 1, 1, 1, dtype=torch.long) * -8, log_h_prev[:-1]], dim=-4) # (nD, nQ, nY, nB)
    log_h_curr = pyro.sample("log_h", dist.Normal(phi * log_h_prev, scale)) # (nD, nQ, nY, nB)
    sqrt_h = (log_h_curr * 0.5).exp().clamp(1e-2, 1e2) # (nD, nQ, nY, nB)

    pyro.sample(
        "obs", 
        dist.Normal((alpha[Y_curr, Q_curr] + avg_return) * sqrt_h, sqrt_h), # (Q, 1, Y, 1 | 1, 1, 1, 1)
        obs=data.permute([3, 2, 1, 0]), # (nD, nQ, nY, nB)
    ) # log_prob = (Q, 1, Y, 1 | nD, nQ, nY, nB)


def guide_manual(data, hidden_dims):
    hY, hQ = hidden_dims
    nB, nY, nQ, nD = data.shape

    probsYY = pyro.param("_probsYY", torch.rand(hY+1, hY), constraint=dist.constraints.simplex)
    pyro.sample("probsYY", dist.Delta(probsYY).to_event(2))
    probsYQQ = pyro.param("_probsYQQ", torch.rand(hY, hQ+1, hQ), constraint=dist.constraints.simplex)
    pyro.sample("probsYQQ", dist.Delta(probsYQQ).to_event(3))
    alpha = pyro.param("_alpha", torch.ones(hY, hQ))
    pyro.sample("alpha", dist.Delta(alpha).to_event(2))
    phi = pyro.param("_phi", torch.tensor(1.0))
    pyro.sample("phi", dist.Delta(phi))
    scale = pyro.param("_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
    pyro.sample("scale", dist.Delta(scale))

    log_h = pyro.param("_log_h", torch.full((nB, nY, nQ, nD), -5.0))
    with pyro.plate("batch", nB):
        with pyro.plate("years", nY):
            with pyro.plate("quarters", nQ):
                with pyro.plate("days", nD):
                    pyro.sample("log_h", dist.Delta(log_h.permute([3, 2, 1, 0]))) # (nD, nQ, nY, nB)


def svi_seq(model, guide, optim, *args, **kwargs):
    nB, nY, nQ, nD = args[0].shape

    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = poutine.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = poutine.trace(poutine.replay(poutine.enum(model, first_available_dim=-1), tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()                       
    
        with time_it() as t_reduce:
            log_prob_Y_0 = torch.stack([tr_model.nodes[f"Y_{b}_0"]["log_prob"]
                                        for b in range(nB)], dim=-1)  # (Y_0 | nB)
            log_prob_Y_1_10 = torch.stack([torch.stack([tr_model.nodes[f"Y_{b}_{i}"]["log_prob"].permute([0, 1] if i % 2 == 0 else [1, 0])
                                        for i in range(1, nY)], dim=-1)
                                        for b in range(nB)], dim=-1)  # (_Y, Y | nY-1, nB)
            log_prob_Q_0 = torch.stack([torch.stack([tr_model.nodes[f"Q_{b}_{i}_0"]["log_prob"].squeeze()
                                        for i in range(nY)], dim=-1)
                                        for b in range(nB)], dim=-1)  # (Q_0, Y | nY, nB)
            log_prob_Q_1_4 = torch.stack([torch.stack([torch.stack([tr_model.nodes[f"Q_{b}_{i}_{j}"]["log_prob"].squeeze().permute([0, 1, 2] if j % 2 == 0 else [1, 0, 2])
                                        for j in range(1, nQ)], dim=-1)
                                        for i in range(nY)], dim=-1)
                                        for b in range(nB)], dim=-1)  # (_Q, Q, Y | nQ-1, nY, nB)
            log_prob_log_h = torch.stack([torch.stack([torch.stack([torch.stack([tr_model.nodes[f"log_h_{b}_{i}_{j}_{k}"]["log_prob"].squeeze()
                                        for k in range(nD)], dim=-1)
                                        for j in range(nQ)], dim=-1)
                                        for i in range(nY)], dim=-1)
                                        for b in range(nB)], dim=-1)  # (nD, nQ, nY, nB)
            log_prob_obs = torch.stack([torch.stack([torch.stack([torch.stack([tr_model.nodes[f"obs_{b}_{i}_{j}_{k}"]["log_prob"].squeeze()
                                        for k in range(nD)], dim=-1)
                                        for j in range(nQ)], dim=-1)
                                        for i in range(nY)], dim=-1)
                                        for b in range(nB)], dim=-1)  # (Q, Y, nD, nQ, nY, nB)

                                                                             # [_Y _Q  Q  Y |  nD,   nQ,  nY,  nB]
            log_prob_Y_0 = log_prob_Y_0[:, None, None, None]                 # [          Y |  1,    1,    1,  nB]
            log_prob_Y_1_10 = log_prob_Y_1_10[:, None, None, :, None, None]  # [_Y, 1, 1, Y |  1,    1, nY-1,  nB]
            log_prob_Q_0 = log_prob_Q_0[:, :, None, None]                    # [       Q, Y |  1,    1,   nY,  nB]
            log_prob_Q_1_4 = log_prob_Q_1_4[:, :, :, None]                   # [   _Q, Q, Y |  1, nQ-1,   nY,  nB]
            log_prob_log_h = log_prob_log_h                                  # [            | nD,   nQ,   nY,  nB]
            log_prob_obs = log_prob_obs                                      # [       Q, Y | nD,   nQ,   nY,  nB]
            
            log_prob_Y = cat([log_prob_Y_0, log_prob_Y_1_10], dim=-2)  # [_Y, 1, 1, Y |  1,  1, nY, nB]
            log_prob_Q = cat([log_prob_Q_0, log_prob_Q_1_4], dim=-3)   # [   _Q, Q, Y |  1, nQ, nY, nB]
            log_prob_log_h = log_prob_log_h                            # [            | nD, nQ, nY, nB]
            log_prob_obs = log_prob_obs                                # [       Q, Y | nD, nQ, nY, nB]

            log_prob_probsYY = tr_model.nodes["probsYY"]["log_prob"]     # []
            log_prob_probsYQQ = tr_model.nodes["probsYQQ"]["log_prob"]   # []
            log_prob_alpha = tr_model.nodes["alpha"]["log_prob"]         # []
            log_prob_phi = tr_model.nodes["phi"]["log_prob"]             # []
            log_prob_scale = tr_model.nodes["scale"]["log_prob"]         # []

            loss = marginalize(log_prob_Y, log_prob_Q, log_prob_log_h, log_prob_obs, log_prob_probsYY, log_prob_probsYQQ, log_prob_alpha, log_prob_phi, log_prob_scale,)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_seq(data, hidden_dims):
    hY, hQ = hidden_dims
    nB, nY, nQ, nD = data.shape

    probsYY = pyro.sample("probsYY", dist.Dirichlet(torch.ones(hY)).expand((hY+1,)).to_event(1))  # (hY+1, hY)
    probsYQQ = pyro.sample("probsYQQ", dist.Dirichlet(torch.ones(hQ)).expand((hY, hQ+1)).to_event(2))  # (hY, hQ+1, hQ)

    avg_return = 8 ** (1 / (nY * nQ * nD)) - 1
    alpha = pyro.sample("alpha", dist.Normal(0, 1e-2).expand((hY, hQ)).to_event(2))  # (hY, hQ)
    phi = pyro.sample("phi", dist.Normal(1, 1e-2))  # ()
    scale = pyro.sample("scale", dist.LogNormal(-1, 1e-2))  # ()

    for b in range(nB):
        log_h = -8
        Y = -1
        for i in pyro.markov(range(nY)):
            Y = pyro.sample(
                f"Y_{b}_{i}",
                Categorical(logits=probsYY[Y].log()),
                infer={"enumerate": "parallel"},
            )
            Q = -1
            for j in pyro.markov(range(nQ)):
                Q = pyro.sample(
                    f"Q_{b}_{i}_{j}",
                    Categorical(logits=probsYQQ[Y, Q].log()),
                    infer={"enumerate": "parallel"},
                )
                for k in pyro.markov(range(nD)):
                    log_h = pyro.sample(f"log_h_{b}_{i}_{j}_{k}", dist.Normal(phi * log_h, scale))
                    sqrt_h = (log_h * 0.5).exp().clamp(1e-2, 1e2)
                    pyro.sample(
                        f"obs_{b}_{i}_{j}_{k}", 
                        dist.Normal((alpha[Y, Q] + avg_return) * sqrt_h, sqrt_h),
                        obs=data[b, i, j, k],
                    )


def guide_seq(data, hidden_dims):
    hY, hQ = hidden_dims
    nB, nY, nQ, nD = data.shape

    probsYY = pyro.param("_probsYY", torch.rand(hY+1, hY), constraint=dist.constraints.simplex)
    pyro.sample("probsYY", dist.Delta(probsYY).to_event(2))
    probsYQQ = pyro.param("_probsYQQ", torch.rand(hY, hQ+1, hQ), constraint=dist.constraints.simplex)
    pyro.sample("probsYQQ", dist.Delta(probsYQQ).to_event(3))
    alpha = pyro.param("_alpha", torch.ones(hY, hQ))
    pyro.sample("alpha", dist.Delta(alpha).to_event(2))
    phi = pyro.param("_phi", torch.tensor(1.0))
    pyro.sample("phi", dist.Delta(phi))
    scale = pyro.param("_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
    pyro.sample("scale", dist.Delta(scale))

    log_h = pyro.param("_log_h", torch.full((nB, nY, nQ, nD), -5.0))
    for b in range(nB):
        for i in pyro.markov(range(nY)):
            for j in pyro.markov(range(nQ)):
                for k in pyro.markov(range(nD)):
                    pyro.sample(f"log_h_{b}_{i}_{j}_{k}", dist.Delta(log_h[b, i, j, k]))


def svi_plate(model, guide, optim, *args, **kwargs):
    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = poutine.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = poutine.trace(poutine.replay(poutine.enum(model, first_available_dim=-2), tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()                       
    
        with time_it() as t_reduce:
            log_prob_Y_0 = tr_model.nodes["Y_0"]["log_prob"]  # (Y_0 | nB)
            log_prob_Y_1_10 = torch.stack([tr_model.nodes[f"Y_{i}"]["log_prob"]
                .permute([0, 1, 2] if i % 2 == 0 else [1, 0, 2])
                for i in range(1, 10)]).permute([1, 2, 0, 3])  # (_Y, Y | nY-1, nB)
            log_prob_Q_0 = torch.stack([tr_model.nodes[f"Q_{i}_0"]["log_prob"].squeeze()
                for i in range(10)]).permute([1, 2, 0, 3])  # (Q_0, Y | nY, nB)
            log_prob_Q_1_4 = torch.stack([torch.stack([tr_model.nodes[f"Q_{i}_{j}"]["log_prob"].squeeze()
                .permute([0, 1, 2, 3] if j % 2 == 0 else [1, 0, 2, 3])
                for j in range(1, 4)])
                for i in range(10)]).permute([2, 3, 4, 1, 0, 5])  # (_Q, Q, Y | nQ-1, nY, nB)
            log_prob_log_h = torch.stack([torch.stack([torch.stack([tr_model.nodes[f"log_h_{i}_{j}_{k}"]["log_prob"].squeeze()
                for k in range(64)])
                for j in range(4)])
                for i in range(10)]).permute([2, 1, 0, 3])  # (nD, nQ, nY, nB)
            log_prob_obs = torch.stack([torch.stack([torch.stack([tr_model.nodes[f"obs_{i}_{j}_{k}"]["log_prob"].squeeze()
                for k in range(64)])
                for j in range(4)])
                for i in range(10)]).permute([3, 4, 2, 1, 0, 5])  # (Q, Y, nD, nQ, nY, nB)

                                                                             # [_Y _Q  Q  Y |  nD,   nQ,  nY,  nB]
            log_prob_Y_0 = log_prob_Y_0[:, None, None, None]                 # [          Y |  1,    1,    1,  nB]
            log_prob_Y_1_10 = log_prob_Y_1_10[:, None, None, :, None, None]  # [_Y, 1, 1, Y |  1,    1, nY-1,  nB]
            log_prob_Q_0 = log_prob_Q_0[:, :, None, None]                    # [       Q, Y |  1,    1,   nY,  nB]
            log_prob_Q_1_4 = log_prob_Q_1_4[:, :, :, None]                   # [   _Q, Q, Y |  1, nQ-1,   nY,  nB]
            log_prob_log_h = log_prob_log_h                                  # [            | nD,   nQ,   nY,  nB]
            log_prob_obs = log_prob_obs                                      # [       Q, Y | nD,   nQ,   nY,  nB]
            
            log_prob_Y = cat([log_prob_Y_0, log_prob_Y_1_10], dim=-2)  # [_Y, 1, 1, Y |  1,  1, nY, nB]
            log_prob_Q = cat([log_prob_Q_0, log_prob_Q_1_4], dim=-3)   # [   _Q, Q, Y |  1, nQ, nY, nB]
            log_prob_log_h = log_prob_log_h                            # [            | nD, nQ, nY, nB]
            log_prob_obs = log_prob_obs                                # [       Q, Y | nD, nQ, nY, nB]

            log_prob_probsYY = tr_model.nodes["probsYY"]["log_prob"]     # []
            log_prob_probsYQQ = tr_model.nodes["probsYQQ"]["log_prob"]   # []
            log_prob_alpha = tr_model.nodes["alpha"]["log_prob"]         # []
            log_prob_phi = tr_model.nodes["phi"]["log_prob"]             # []
            log_prob_scale = tr_model.nodes["scale"]["log_prob"]         # []

            loss = marginalize(log_prob_Y, log_prob_Q, log_prob_log_h, log_prob_obs, log_prob_probsYY, log_prob_probsYQQ, log_prob_alpha, log_prob_phi, log_prob_scale,)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_plate(data, hidden_dims):
    hY, hQ = hidden_dims
    nB, nY, nQ, nD = data.shape

    probsYY = pyro.sample("probsYY", dist.Dirichlet(torch.ones(hY)).expand((hY+1,)).to_event(1))  # (hY+1, hY)
    probsYQQ = pyro.sample("probsYQQ", dist.Dirichlet(torch.ones(hQ)).expand((hY, hQ+1)).to_event(2))  # (hY, hQ+1, hQ)

    avg_return = 8 ** (1 / (nY * nQ * nD)) - 1
    alpha = pyro.sample("alpha", dist.Normal(0, 1e-2).expand((hY, hQ)).to_event(2))  # (hY, hQ)
    phi = pyro.sample("phi", dist.Normal(1, 1e-2))  # ()
    scale = pyro.sample("scale", dist.LogNormal(-1, 1e-2))  # ()

    with pyro.plate("batch", nB) as b:
        log_h = -8
        Y = -1
        for i in pyro.markov(range(nY)):
            Y = pyro.sample(
                f"Y_{i}",
                Categorical(logits=probsYY[Y].log()),
                infer={"enumerate": "parallel"},
            )
            Q = -1
            for j in pyro.markov(range(nQ)):
                Q = pyro.sample(
                    f"Q_{i}_{j}",
                    Categorical(logits=probsYQQ[Y, Q].log()),
                    infer={"enumerate": "parallel"},
                )
                for k in pyro.markov(range(nD)):
                    log_h = pyro.sample(f"log_h_{i}_{j}_{k}", dist.Normal(phi * log_h, scale))
                    sqrt_h = (log_h * 0.5).exp().clamp(1e-2, 1e2)
                    pyro.sample(
                        f"obs_{i}_{j}_{k}", 
                        dist.Normal((alpha[Y, Q] + avg_return) * sqrt_h, sqrt_h),
                        obs=data[b, i, j, k],
                    )


def guide_plate(data, hidden_dims):
    hY, hQ = hidden_dims
    nB, nY, nQ, nD = data.shape

    probsYY = pyro.param("_probsYY", torch.rand(hY+1, hY), constraint=dist.constraints.simplex)
    pyro.sample("probsYY", dist.Delta(probsYY).to_event(2))
    probsYQQ = pyro.param("_probsYQQ", torch.rand(hY, hQ+1, hQ), constraint=dist.constraints.simplex)
    pyro.sample("probsYQQ", dist.Delta(probsYQQ).to_event(3))
    alpha = pyro.param("_alpha", torch.ones(hY, hQ))
    pyro.sample("alpha", dist.Delta(alpha).to_event(2))
    phi = pyro.param("_phi", torch.tensor(1.0))
    pyro.sample("phi", dist.Delta(phi))
    scale = pyro.param("_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
    pyro.sample("scale", dist.Delta(scale))

    log_h = pyro.param("_log_h", torch.full((nB, nY, nQ, nD), -5.0))
    with pyro.plate("batch", nB) as b:
        for i in pyro.markov(range(nY)):
            for j in pyro.markov(range(nQ)):
                for k in pyro.markov(range(nD)):
                    pyro.sample(f"log_h_{i}_{j}_{k}", dist.Delta(log_h[b, i, j, k]))


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
                                                                  # [_Y, _Q,  Q,  Y, nD, nQ, nY, nB]
            log_prob_Y = tr_model.nodes["Y"]["log_prob"]          # [ 4,  1,  1,  4,  1,  1, 10, 10]
            log_prob_Q = tr_model.nodes["Q"]["log_prob"]          # [     4,  4,  4,  1,  4, 10, 10]
            log_prob_log_h = tr_model.nodes["log_h"]["log_prob"]  # [                64,  4, 10, 10]
            log_prob_obs = tr_model.nodes["obs"]["log_prob"]      # [         4,  4, 64,  4, 10, 10]

            log_prob_probsYY = tr_model.nodes["probsYY"]["log_prob"]    # []
            log_prob_probsYQQ = tr_model.nodes["probsYQQ"]["log_prob"]  # []
            log_prob_alpha = tr_model.nodes["alpha"]["log_prob"]        # []
            log_prob_phi = tr_model.nodes["phi"]["log_prob"]            # []
            log_prob_scale = tr_model.nodes["scale"]["log_prob"]        # []

            loss = marginalize(log_prob_Y, log_prob_Q, log_prob_log_h, log_prob_obs, log_prob_probsYY, log_prob_probsYQQ, log_prob_alpha, log_prob_phi, log_prob_scale)

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
def model_ours(s: vec.State, data, hidden_dims):
    hY, hQ = hidden_dims
    nB, nY, nQ, nD = data.shape

    probsYY = pyro.sample("probsYY", dist.Dirichlet(torch.ones(hY)).expand((hY+1,)).to_event(1))  # (hY+1, hY)
    probsYQQ = pyro.sample("probsYQQ", dist.Dirichlet(torch.ones(hQ)).expand((hY, hQ+1)).to_event(2))  # (hY, hQ+1, hQ)

    avg_return = 8 ** (1 / (nY * nQ * nD)) - 1
    alpha = pyro.sample("alpha", dist.Normal(0, 1e-2).expand((hY, hQ)).to_event(2))  # (hY, hQ)
    phi = pyro.sample("phi", dist.Normal(1, 1e-2))  # ()
    scale = pyro.sample("scale", dist.LogNormal(-1, 1e-2))  # ()

    for s.b in vec.range("batch", nB, vectorized=True):
        s.log_h = -8
        s.Y = -1
        for s.i in vec.range("years", nY, vectorized=True):
            s.Y = pyro.sample(
                "Y",
                Categorical(logits=Index(probsYY)[s.Y].log()),
                infer={"enumerate": "parallel"},
            )
            s.Q = -1
            for s.j in vec.range("quarters", nQ, vectorized=True):
                s.Q = pyro.sample(
                    "Q",
                    Categorical(logits=Index(probsYQQ)[s.Y, s.Q].log()),
                    infer={"enumerate": "parallel"},
                )
                for s.k in vec.range("days", nD, vectorized=True):
                    s.log_h = pyro.sample("log_h", dist.Normal(phi * s.log_h, scale))
                    s.sqrt_h = (s.log_h * 0.5).exp().clamp(1e-2, 1e2)
                    pyro.sample(
                        "obs", 
                        dist.Normal((Index(alpha)[s.Y, s.Q] + avg_return) * s.sqrt_h, s.sqrt_h),
                        obs=Index(data)[s.b, s.i, s.j, s.k]
                    )


@vec.vectorize
def guide_ours(s: vec.State, data, hidden_dims):
    hY, hQ = hidden_dims
    nB, nY, nQ, nD = data.shape

    probsYY = pyro.param("_probsYY", torch.rand(hY+1, hY), constraint=dist.constraints.simplex)
    pyro.sample("probsYY", dist.Delta(probsYY).to_event(2))
    probsYQQ = pyro.param("_probsYQQ", torch.rand(hY, hQ+1, hQ), constraint=dist.constraints.simplex)
    pyro.sample("probsYQQ", dist.Delta(probsYQQ).to_event(3))
    alpha = pyro.param("_alpha", torch.ones(hY, hQ))
    pyro.sample("alpha", dist.Delta(alpha).to_event(2))
    phi = pyro.param("_phi", torch.tensor(1.0))
    pyro.sample("phi", dist.Delta(phi))
    scale = pyro.param("_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
    pyro.sample("scale", dist.Delta(scale))

    s.log_h = pyro.param("_log_h", torch.full((nB, nY, nQ, nD), -5.0))
    for s.b in vec.range("batch", nB, vectorized=True):
        for s.i in vec.range("years", nY, vectorized=True):
            for s.j in vec.range("quarters", nQ, vectorized=True):
                for s.k in vec.range("days", nD, vectorized=True):
                    pyro.sample("log_h", dist.Delta(Index(s.log_h)[s.b, s.i, s.j, s.k]))


def guide(model, data, hidden_dims):
    if model is model_manual:
        return guide_manual
    elif model is model_seq:
        return guide_seq
    elif model is model_plate:
        return guide_plate
    elif model is model_ours:
        return guide_ours
    else:
        return None


def data(args):
    df = pd.read_csv(os.path.join(path, "dataset/stock.txt"), parse_dates=True, index_col="Date")
    df.index = pd.to_datetime(df.index, utc=True)
    dfs_month = [group for _, group in df.groupby([df.index.year, df.index.month])]
    dfs_quarter = [pd.concat(dfs_month[i * 3 : i * 3 + 3], axis=0) for i in range(len(dfs_month) // 3)]
    max_days = max([len(quarter) for quarter in dfs_quarter])
    data = np.array([np.pad(df_quarter.to_numpy(), ((0, max_days - len(df_quarter)), (0, 0)), "edge") for df_quarter in dfs_quarter])
    data = torch.as_tensor(data, device=torch.get_default_device()).reshape(10, 4, 64, 10).permute([3, 0, 1, 2]).to(torch.float32)
    data = data[:args.num_batch]
    hidden_dims = (4, 4)
    return (data, hidden_dims)


def optim(args):
    return Adam({"lr": args.lr})