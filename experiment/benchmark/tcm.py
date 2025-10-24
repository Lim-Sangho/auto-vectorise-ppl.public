import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, "../"))
sys.path.append(os.path.join(path, "../../"))
sys.dont_write_bytecode = True

import pyro
import torch
import numpy as np
import pyro.distributions as dist

from pyro import poutine
from pyro.optim import Adam

import vectorized_loop as vec
from vectorized_loop.ops import Index

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

pyro.poutine.replay_messenger.ReplayMessenger._pyro_sample = _pyro_sample


def marginalize(model_theta, model_q_noise, model_obs, guide_theta, guide_q_noise):
    model_theta, model_q_noise, model_obs, guide_theta, guide_q_noise = \
        map(torch.Tensor.contiguous, (model_theta, model_q_noise, model_obs, guide_theta, guide_q_noise))
    
    model_log_prob = model_theta.sum() + model_q_noise.sum() + model_obs.sum()
    guide_log_prob = guide_theta.sum() + guide_q_noise.sum()
    loss = guide_log_prob - model_log_prob
    return loss


def svi_seq(model, guide, optim, *args, **kwargs):
    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = poutine.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = poutine.trace(poutine.replay(model, tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()

        with time_it() as t_reduce:
            model_theta = torch.stack([torch.stack([tr_model.nodes[f"theta_{b}_{i}"]["log_prob"]
                for i in range(200)]) for b in range(10)]).permute([1, 0])  # (N, B)
            model_q_noise = torch.stack([torch.stack([tr_model.nodes[f"q_noise_{b}_{i}"]["log_prob"]
                for i in range(200)]) for b in range(10)]).permute([1, 0])  # (N, B)
            model_obs = torch.stack([torch.stack([tr_model.nodes[f"obs_{b}_{i}"]["log_prob"]
                for i in range(200)]) for b in range(10)]).permute([1, 0])  # (N, B)
            guide_theta = torch.stack([torch.stack([tr_guide.nodes[f"theta_{b}_{i}"]["log_prob"]
                for i in range(200)]) for b in range(10)]).permute([1, 0])  # (N, B)
            guide_q_noise = torch.stack([torch.stack([tr_guide.nodes[f"q_noise_{b}_{i}"]["log_prob"]
                for i in range(200)]) for b in range(10)]).permute([1, 0])  # (N, B)

            loss = marginalize(model_theta, model_q_noise, model_obs, guide_theta, guide_q_noise)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_seq(data):
    B, N = data.shape
    theta_a = 32.0
    theta_lower = 19.5
    theta_upper = 20.5
    invCR = 1/15
    RP_rate = 21.0
    sigma_0 = 0.2
    sigma_1 = 0.22
    sqrt_dt = 0.5

    for b in range(B):
        q = 0.0
        theta = 20.0
        for i in range(N):
            if theta <= theta_lower:
                q_noise = pyro.sample(f"q_noise_{b}_{i}", dist.Laplace(0, 1e-3))
            elif theta >= theta_upper:
                q_noise = pyro.sample(f"q_noise_{b}_{i}", dist.Laplace(1, 1e-3))
            else:
                q_noise = pyro.sample(f"q_noise_{b}_{i}", dist.Normal(q, 1e-1))

            q = torch.as_tensor(q_noise > 0.5, dtype=torch.float)
            d_theta = invCR * (theta_a - theta - q * RP_rate)
            sigma = torch.where(q.bool(), sigma_1, sigma_0)
            theta = pyro.sample(f"theta_{b}_{i}", dist.Normal(theta + d_theta * sqrt_dt ** 2, sigma * sqrt_dt))
            pyro.sample(f"obs_{b}_{i}", dist.Normal(theta, 2e-1), obs=data[b, i])


def guide_seq(data):
    B, N = data.shape

    loc_theta = pyro.param("loc_theta", torch.full((B, N), 20.0))
    scale_theta = pyro.param("scale_theta", torch.full((B, N), 1.0), constraint=dist.constraints.positive)
    theta = dist.Normal(loc_theta, scale_theta).sample()

    loc_q_noise = pyro.param("loc_q_noise", torch.full((B, N), 0.5))
    scale_q_noise = pyro.param("scale_q_noise", torch.full((B, N), 1.0), constraint=dist.constraints.positive)
    q_noise = dist.Normal(loc_q_noise, scale_q_noise).sample()

    for b in range(B):
        for i in range(N):
            pyro.sample(f"theta_{b}_{i}", dist.Normal(loc_theta[b, i], scale_theta[b, i]), obs=theta[b, i])
            pyro.sample(f"q_noise_{b}_{i}", dist.Normal(loc_q_noise[b, i], scale_q_noise[b, i]), obs=q_noise[b, i])


svi_plate = None
model_plate = None


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
            tr_model = vec.trace(vec.replay(model, tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()

        with time_it() as t_reduce:
            model_theta = tr_model.nodes["theta"]["log_prob"]  # (N, B)
            model_q_noise = tr_model.nodes["q_noise"]["log_prob"]  # (N, B)
            model_obs = tr_model.nodes["obs"]["log_prob"]  # (N, B)
            guide_theta = tr_guide.nodes["theta"]["log_prob"]  # (N, B)
            guide_q_noise = tr_guide.nodes["q_noise"]["log_prob"]  # (N, B)

            loss = marginalize(model_theta, model_q_noise, model_obs, guide_theta, guide_q_noise)
            

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
def model_ours(s: vec.State, data):
    B, N = data.shape
    theta_a = 32.0
    theta_lower = 19.5
    theta_upper = 20.5
    invCR = 1/15
    RP_rate = 21.0
    sigma_0 = 0.2
    sigma_1 = 0.22
    sqrt_dt = 0.5

    for s.b in vec.range("batch", B, vectorized=True):
        s.q = 0.0
        s.theta = 20.0
        for s.i in vec.range("time", N, vectorized=True):
            with vec.branch(s.theta <= theta_lower):
                s.q_noise = pyro.sample("q_noise", dist.Laplace(0, 1e-3))
            with vec.branch(s.theta >= theta_upper):
                s.q_noise = pyro.sample("q_noise", dist.Laplace(1, 1e-3))
            with vec.branch((theta_lower < s.theta) & (s.theta < theta_upper)):
                s.q_noise = pyro.sample("q_noise", dist.Normal(s.q, 1e-1))

            s.q = torch.as_tensor(s.q_noise > 0.5, dtype=torch.float)
            s.d_theta = invCR * (theta_a - s.theta - s.q * RP_rate)
            s.sigma = torch.where(s.q.bool(), sigma_1, sigma_0)
            s.theta = pyro.sample("theta", dist.Normal(s.theta + s.d_theta * sqrt_dt ** 2, s.sigma * sqrt_dt))
            pyro.sample("obs", dist.Normal(s.theta, 2e-1), obs=Index(data)[s.b, s.i])


@vec.vectorize
def guide_ours(s: vec.State, data):
    B, N = data.shape

    loc_theta = pyro.param("loc_theta", torch.full((B, N), 20.0))
    scale_theta = pyro.param("scale_theta", torch.full((B, N), 1.0), constraint=dist.constraints.positive)
    theta = dist.Normal(loc_theta, scale_theta).sample()

    loc_q_noise = pyro.param("loc_q_noise", torch.full((B, N), 0.5))
    scale_q_noise = pyro.param("scale_q_noise", torch.full((B, N), 1.0), constraint=dist.constraints.positive)
    q_noise = dist.Normal(loc_q_noise, scale_q_noise).sample()

    for s.b in vec.range("batch", B, vectorized=True):
        for s.i in vec.range("time", N, vectorized=True):
            pyro.sample(f"theta", dist.Normal(Index(loc_theta)[s.b, s.i], Index(scale_theta)[s.b, s.i]), obs=Index(theta)[s.b, s.i])
            pyro.sample(f"q_noise", dist.Normal(Index(loc_q_noise)[s.b, s.i], Index(scale_q_noise)[s.b, s.i]), obs=Index(q_noise)[s.b, s.i])


def guide(model, data):
    if model is model_seq:
        return guide_seq
    elif model is model_ours:
        return guide_ours
    else:
        return None


def data(args):
    data = torch.as_tensor(np.loadtxt(os.path.join(path, "dataset/thermo/data.txt")), device=torch.get_default_device()).to(torch.float32)
    return (data,)


def optim(args):
    return Adam({"lr": args.lr})