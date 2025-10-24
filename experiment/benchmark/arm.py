import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, "../"))
sys.path.append(os.path.join(path, "../../"))
sys.dont_write_bytecode = True

import pyro.poutine.replay_messenger
import torch
from torch import nn

import pyro
import numpy as np
import pyro.distributions as dist

from pyro import poutine
from pyro.optim import Adam

from pyro.contrib.funsor import vectorized_markov, plate, condition
from pyro.contrib.funsor.handlers import trace

import vectorized_loop as vec
from vectorized_loop.ops import Index, cat

from util import time_it


def _pyro_sample(self, msg):
    assert msg["name"] is not None
    name = msg["name"]
    if self.trace is not None and name in self.trace:
        guide_msg = self.trace.nodes[name]
        if msg["is_observed"]:
            return None
        msg["done"] = True
        msg["value"] = guide_msg["value"]
        msg["infer"] = guide_msg["infer"]

def _pyro_post_sample(self, msg):
    if self.param_only:
        return
    if msg["infer"].get("_do_not_trace"):
        return
    self.trace.add_node(msg["name"], **msg.copy())
    
def add_node(self, site_name, **kwargs):
    self.nodes[site_name] = kwargs
    self._pred[site_name] = set()
    self._succ[site_name] = set()

pyro.poutine.replay_messenger.ReplayMessenger._pyro_sample = _pyro_sample
pyro.poutine.trace_messenger.TraceMessenger._pyro_post_sample = _pyro_post_sample
pyro.poutine.trace_struct.Trace.add_node = add_node


def marginalize(log_prob_K_tilde_guide, log_prob_coeff_guide, log_prob_x_guide,
                log_prob_K_tilde_model, log_prob_coeff_model, log_prob_x_model, log_prob_obs_model):
    log_prob_K_tilde_guide, log_prob_coeff_guide, log_prob_x_guide, \
    log_prob_K_tilde_model, log_prob_coeff_model, log_prob_x_model, log_prob_obs_model = \
        map(torch.Tensor.contiguous, (log_prob_K_tilde_guide, log_prob_coeff_guide, log_prob_x_guide,
                                      log_prob_K_tilde_model, log_prob_coeff_model, log_prob_x_model, log_prob_obs_model))

    f_guide = log_prob_K_tilde_guide  # only K_tilde depends on K_tilde in guide
    f_model = log_prob_K_tilde_model + log_prob_x_model.sum() + log_prob_obs_model.sum()  # K_tilde, x, obs depend on K_tilde in model
    surrogate_K_tilde = log_prob_K_tilde_guide * (f_guide - f_model).detach()

    log_prob_guide = log_prob_coeff_guide + log_prob_x_guide.sum()
    log_prob_model = log_prob_K_tilde_model + log_prob_coeff_model + log_prob_x_model.sum() + log_prob_obs_model.sum()
    loss = log_prob_guide - log_prob_model + surrogate_K_tilde
    return loss


def svi_manual(model, guide, optim, *args, **kwargs):
    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = poutine.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = poutine.trace(poutine.replay(model, tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()

        with time_it() as t_reduce:
            log_prob_K_tilde_guide = tr_guide.nodes["K_tilde"]["log_prob"]
            log_prob_coeff_guide = tr_guide.nodes["coeff"]["log_prob"]
            log_prob_x_guide = tr_guide.nodes["x"]["log_prob"]

            log_prob_K_tilde_model = tr_model.nodes["K_tilde"]["log_prob"]
            log_prob_coeff_model = tr_model.nodes["coeff"]["log_prob"]
            log_prob_x_model = tr_model.nodes["x"]["log_prob"]
            log_prob_obs_model = tr_model.nodes["obs"]["log_prob"]

            loss = marginalize(log_prob_K_tilde_guide, log_prob_coeff_guide, log_prob_x_guide,
                               log_prob_K_tilde_model, log_prob_coeff_model, log_prob_x_model, log_prob_obs_model)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_manual(data, K_max):
    N = len(data)
    K_tilde = pyro.sample("K_tilde", dist.Poisson(10.0))
    K = min(int(K_tilde), K_max)
    coeff = pyro.sample("coeff", dist.Normal(0, 1).expand((K_max,)).to_event(1)) # (K_max,)

    xs = [pyro.sample("x", dist.Normal(torch.zeros(N), 1), infer={"_do_not_trace": True, "is_auxiliary": True}) for _ in range(K)] # K x (N,)
    for k in range(K):
        xs[k] = torch.cat([torch.zeros(K-k), xs[k][:N-(K-k)]]) # (N,)
    xs = torch.stack(xs, dim=-1) if K > 0 else torch.zeros((N, K))  # (N, K)
    x = pyro.sample("x", dist.Normal((xs * coeff[:K]).sum(-1), 1)) # (N,)
    pyro.sample("obs", dist.Normal(x, 1.0), obs=data) # (N,)


def guide_manual(data, K_max):
    N = len(data)
    rate_K_tilde = pyro.param("rate_K_tilde", torch.tensor(10.0), constraint=dist.constraints.positive)
    K_tilde = pyro.sample("K_tilde", dist.Poisson(rate_K_tilde))

    loc_coeff = pyro.param("loc_coeff", torch.ones(K_max))
    scale_coeff = pyro.param("scale_coeff", torch.tensor(1e-2), constraint=dist.constraints.positive)
    coeff = pyro.sample("coeff", dist.Normal(loc_coeff, scale_coeff).to_event(1))

    loc_x = pyro.param("loc_x", torch.zeros(N))
    scale_x = pyro.param("scale_x", torch.tensor(1e-2), constraint=dist.constraints.positive)
    x = dist.Normal(loc_x, scale_x).sample() # (N,)
    pyro.sample("x", dist.Normal(loc_x, scale_x), obs=x) # (N,)


def svi_seq(model, guide, optim, *args, **kwargs):
    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = poutine.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = poutine.trace(poutine.replay(model, tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()

        with time_it() as t_reduce:
            log_prob_K_tilde_guide = tr_guide.nodes["K_tilde"]["log_prob"]
            log_prob_coeff_guide = tr_guide.nodes["coeff"]["log_prob"]
            log_prob_x_guide = torch.stack([tr_guide.nodes[f"x_{i}"]["log_prob"] for i in range(len(args[0]))])

            log_prob_K_tilde_model = tr_model.nodes["K_tilde"]["log_prob"]
            log_prob_coeff_model = tr_model.nodes["coeff"]["log_prob"]
            log_prob_x_model = torch.stack([tr_model.nodes[f"x_{i}"]["log_prob"] for i in range(len(args[0]))])
            log_prob_obs_model = torch.stack([tr_model.nodes[f"obs_{i}"]["log_prob"] for i in range(len(args[0]))])

            loss = marginalize(log_prob_K_tilde_guide, log_prob_coeff_guide, log_prob_x_guide,
                               log_prob_K_tilde_model, log_prob_coeff_model, log_prob_x_model, log_prob_obs_model)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_seq(data, K_max):
    N = len(data)
    K_tilde = pyro.sample("K_tilde", dist.Poisson(10.0))
    K = min(int(K_tilde), K_max)
    coeff = pyro.sample("coeff", dist.Normal(0, 1).expand((K_max,)).to_event(1))

    x_prev = torch.zeros(K)
    for i in range(N):
        x = pyro.sample(f"x_{i}", dist.Normal((x_prev * coeff[:K]).sum(-1), 1))
        x_prev = cat([x.unsqueeze(-1), x_prev], dim=-1)[..., :K]
        pyro.sample(f"obs_{i}", dist.Normal(x, 1.0), obs=data[i])


def guide_seq(data, K_max):
    N = len(data)
    rate_K_tilde = pyro.param("rate_K_tilde", torch.tensor(10.0), constraint=dist.constraints.positive)
    K_tilde = pyro.sample("K_tilde", dist.Poisson(rate_K_tilde))

    loc_coeff = pyro.param("loc_coeff", torch.ones(K_max))
    scale_coeff = pyro.param("scale_coeff", torch.tensor(1e-2), constraint=dist.constraints.positive)
    coeff = pyro.sample("coeff", dist.Normal(loc_coeff, scale_coeff).to_event(1))

    loc_x = pyro.param("loc_x", torch.zeros(N))
    scale_x = pyro.param("scale_x", torch.tensor(1e-2), constraint=dist.constraints.positive)
    x = dist.Normal(loc_x, scale_x).sample()  # (N,)
    for i in range(N):
        pyro.sample(f"x_{i}", dist.Normal(loc_x[i], scale_x), obs=x[i])


svi_plate = None
model_plate = None


def svi_vmarkov(model, guide, optim, *args, **kwargs):
    N = len(args[0])
    K_max = args[1]

    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()

        with time_it() as t_model:
            K_tilde_guide = tr_guide.nodes["K_tilde"]["value"]
            coeff_guide = tr_guide.nodes["coeff"]["value"]
            x_guide = tr_guide.nodes["x"]["value"]
            K = min(int(K_tilde_guide), K_max)
            
            cond = {"K_tilde": K_tilde_guide, "coeff": coeff_guide}
            cond.update({f"x_{i}": x_guide[i] for i in range(K)})
            cond.update({f"x_slice({i}, {N-K+i}, None)": x_guide[i:N-K+i] for i in range(K+1)})

            tr_model = trace(condition(model, cond)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()

        with time_it() as t_reduce:
            log_prob_K_tilde_guide = tr_guide.nodes["K_tilde"]["log_prob"]
            log_prob_coeff_guide = tr_guide.nodes["coeff"]["log_prob"]
            log_prob_x_guide = tr_guide.nodes["x"]["log_prob"]

            log_prob_K_tilde_model = tr_model.nodes["K_tilde"]["log_prob"]
            log_prob_coeff_model = tr_model.nodes["coeff"]["log_prob"]

            log_prob_x_K_model = torch.stack([tr_model.nodes[f"x_{i}"]["log_prob"] for i in range(K)])
            log_prob_x_N_model = tr_model.nodes[f"x_slice({K}, {N}, None)"]["log_prob"]
            log_prob_x_model = torch.cat([log_prob_x_K_model, log_prob_x_N_model], dim=-1)

            log_prob_obs_K_model = torch.stack([tr_model.nodes[f"obs_{i}"]["log_prob"] for i in range(K)])
            log_prob_obs_N_model = tr_model.nodes[f"obs_slice({K}, {N}, None)"]["log_prob"]
            log_prob_obs_model = torch.cat([log_prob_obs_K_model, log_prob_obs_N_model], dim=-1)

            loss = marginalize(log_prob_K_tilde_guide, log_prob_coeff_guide, log_prob_x_guide,
                               log_prob_K_tilde_model, log_prob_coeff_model, log_prob_x_model, log_prob_obs_model)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_vmarkov(data, K_max):
    N = len(data)
    K_tilde = pyro.sample("K_tilde", dist.Poisson(10.0))
    K = min(int(K_tilde), K_max)
    coeff = pyro.sample("coeff", dist.Normal(0, 1).expand((K_max,)).to_event(1))

    x_prev_int = torch.zeros(K)
    x_prev_tensor = torch.zeros(N-K, K)
    for i in vectorized_markov(None, "length", N, dim=-1, history=K):
        if isinstance(i, int):
            x = pyro.sample(f"x_{i}", dist.Normal((x_prev_int * coeff[:K]).sum(-1), 1))  # ()
            x_prev_int = cat([x.unsqueeze(-1), x_prev_int], dim=-1)[..., :K]  # (K,)
        else:
            x = pyro.sample(f"x_{i}", dist.Normal((x_prev_tensor * coeff[:K]).sum(-1), 1))  # (N-K,)
            x_prev_tensor = cat([x.unsqueeze(-1), x_prev_tensor], dim=-1)[..., :K]  # (N-K, K)
        pyro.sample(f"obs_{i}", dist.Normal(x, 1.0), obs=data[i])  # (N,)


def guide_vmarkov(data, K_max):
    N = len(data)
    rate_K_tilde = pyro.param("rate_K_tilde", torch.tensor(10.0), constraint=dist.constraints.positive)
    K_tilde = pyro.sample("K_tilde", dist.Poisson(rate_K_tilde))

    loc_coeff = pyro.param("loc_coeff", torch.ones(K_max))
    scale_coeff = pyro.param("scale_coeff", torch.tensor(1e-2), constraint=dist.constraints.positive)
    coeff = pyro.sample("coeff", dist.Normal(loc_coeff, scale_coeff).to_event(1))

    loc_x = pyro.param("loc_x", torch.zeros(N))
    scale_x = pyro.param("scale_x", torch.tensor(1e-2), constraint=dist.constraints.positive)
    x = dist.Normal(loc_x, scale_x).sample()  # (N,)
    with plate("length", N, dim=-1) as i:
        pyro.sample("x", dist.Normal(loc_x[i], scale_x), obs=x[i])


svi_discHMM = None
model_discHMM = None


def svi_ours(model, guide, optim, *args, **kwargs):
    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = vec.trace(vec.vectorize(guide)).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = vec.trace(vec.vectorize(vec.replay(model, tr_guide))).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()

        with time_it() as t_reduce:
            log_prob_K_tilde_guide = tr_guide.nodes["K_tilde"]["log_prob"]
            log_prob_coeff_guide = tr_guide.nodes["coeff"]["log_prob"]
            log_prob_x_guide = tr_guide.nodes["x"]["log_prob"]

            log_prob_K_tilde_model = tr_model.nodes["K_tilde"]["log_prob"]
            log_prob_coeff_model = tr_model.nodes["coeff"]["log_prob"]
            log_prob_x_model = tr_model.nodes["x"]["log_prob"]
            log_prob_obs_model = tr_model.nodes["obs"]["log_prob"]

            loss = marginalize(log_prob_K_tilde_guide, log_prob_coeff_guide, log_prob_x_guide,
                               log_prob_K_tilde_model, log_prob_coeff_model, log_prob_x_model, log_prob_obs_model)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_ours(s: vec.State, data, K_max):
    N = len(data)
    K_tilde = pyro.sample("K_tilde", dist.Poisson(10.0))  # ()
    K = min(int(K_tilde), K_max)
    coeff = pyro.sample("coeff", dist.Normal(0, 1).expand((K_max,)).to_event(1))  # (K_max,)

    s.x_prev = torch.zeros(K)  # (K,)
    for s.i in vec.range("length", N, vectorized=True):
        s.x = pyro.sample("x", dist.Normal((s.x_prev * coeff[:K]).sum(-1), 1))  # (N,)
        s.x_prev = cat([s.x.unsqueeze(-1), s.x_prev], dim=-1)[..., :K]  # (N, K)
        pyro.sample("obs", dist.Normal(s.x, 1.0), obs=Index(data)[s.i])  # (N,)


def guide_ours(s: vec.State, data, K_max):
    N = len(data)
    rate_K_tilde = pyro.param("rate_K_tilde", torch.tensor(10.0), constraint=dist.constraints.positive)
    K_tilde = pyro.sample("K_tilde", dist.Poisson(rate_K_tilde))

    loc_coeff = pyro.param("loc_coeff", torch.ones(K_max))
    scale_coeff = pyro.param("scale_coeff", torch.tensor(1e-2), constraint=dist.constraints.positive)
    coeff = pyro.sample("coeff", dist.Normal(loc_coeff, scale_coeff).to_event(1))

    loc_x = pyro.param("loc_x", torch.zeros(N))
    scale_x = pyro.param("scale_x", torch.tensor(1e-2), constraint=dist.constraints.positive)
    x = dist.Normal(loc_x, scale_x).sample()  # (N,)
    for s.i in vec.range("length", N, vectorized=True):
        pyro.sample("x", dist.Normal(Index(loc_x)[s.i], scale_x), obs=Index(x)[s.i])


def guide(model, data, K_max):
    if model is model_manual:
        return guide_manual
    elif model is model_seq:
        return guide_seq
    elif model is model_vmarkov:
        return guide_vmarkov
    elif model is model_ours:
        return guide_ours
    else:
        return None


def data(args):
    data = torch.as_tensor(np.loadtxt(os.path.join(path, "dataset/speech.txt")), device=torch.get_default_device(), dtype=torch.float32)
    K_max = 100
    return data, K_max


def optim(args):
    return Adam({"lr": args.lr})