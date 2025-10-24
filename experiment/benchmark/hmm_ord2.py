import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, "../"))
sys.path.append(os.path.join(path, "../../"))
sys.dont_write_bytecode = True

import torch

import pyro
import pyro.distributions as dist

from pyro import poutine
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDelta

from pyro.ops.contract import einsum
from pyro.contrib.funsor import vectorized_markov, plate
from pyro.contrib.funsor.handlers import mask, enum, trace, replay

from functools import partial
from opt_einsum import get_symbol

import vectorized_loop as vec
from vectorized_loop.distributions import \
    NoClampBernoulli as Bernoulli, NoClampCategorical as Categorical
from vectorized_loop.ops import Index, cat

from util import time_it


def marginalize(log_prob_x_0, log_prob_x_1, log_prob_x_2_T, log_prob_y, log_prob_probs_x, log_prob_probs_y):
    log_prob_x_0, log_prob_x_1, log_prob_x_2_T, log_prob_y, log_prob_probs_x, log_prob_probs_y = \
        map(torch.Tensor.contiguous, (log_prob_x_0, log_prob_x_1, log_prob_x_2_T, log_prob_y, log_prob_probs_x, log_prob_probs_y))
    _, data_dim, max_length, num_sequences = log_prob_y.shape

    log_prob_y_sum = log_prob_y.sum(-3)  # (x_curr | max_length, num_sequences)
    log_prob_xy_0 = log_prob_x_0 + log_prob_y_sum[:, 0]  # (x_0 | num_sequences)
    log_prob_xy_1 = log_prob_x_1 + log_prob_y_sum[:, 1]  # (x_0, x_1 | num_sequences)
    log_prob_xy_2_T = log_prob_x_2_T + log_prob_y_sum[:, 2:]  # (x_prev_prev, x_prev, x_curr | max_length-2, num_sequences)

    p = get_symbol(max_length)
    equation = (p + ",").join([get_symbol(t) + get_symbol(t+1) + get_symbol(t+2) for t in range(max_length-2)])
    equation = (p + ",").join([get_symbol(0), get_symbol(0) + get_symbol(1), equation]) + (p + "->")

    log_prob_xy = einsum(equation,
        log_prob_xy_0,
        log_prob_xy_1,
        *log_prob_xy_2_T.permute(3, 0, 1, 2, 4),
        plates=p,
        backend="pyro.ops.einsum.torch_log"
    )[0]

    log_prob = log_prob_xy + log_prob_probs_x + log_prob_probs_y
    loss = -log_prob
    return loss


def svi_manual(model, guide, optim, *args, **kwargs):
    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = poutine.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = poutine.trace(poutine.replay(poutine.enum(model, first_available_dim=-4), tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()

        with time_it() as t_reduce:
            log_prob_x = tr_model.nodes["x"]["log_prob"].permute([2, 1, 0, 5, 4, 3])                       # (x_prev_prev, x_prev, x_curr | 1, max_length, num_sequences)
            log_prob_x_0 = log_prob_x[0, 0, :, 0, 0]                                                       # (x_0 | num_sequences)
            log_prob_x_1 = log_prob_x[0, :, :, 0, 1]                                                       # (x_0, x_1 | num_sequences)
            log_prob_x_2_T = log_prob_x[:, :, :, 0, 2:]                                                    # (x_prev_prev, x_prev, x_curr | max_length-2, num_sequences)
            log_prob_y = tr_model.nodes["y"]["log_prob"].permute([2, 1, 0, 5, 4, 3]).squeeze(0).squeeze(0) # (x_curr | data_dim, max_length, num_sequences)

            log_prob_probs_x = tr_model.nodes["probs_x"]["log_prob"]
            log_prob_probs_y = tr_model.nodes["probs_y"]["log_prob"]

            loss = marginalize(log_prob_x_0, log_prob_x_1, log_prob_x_2_T, log_prob_y, log_prob_probs_x, log_prob_probs_y)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_manual(sequences, lengths, hidden_dim=16, is_guide=False):
    num_sequences, max_length, data_dim = sequences.shape
    probs_x = pyro.sample(
        "probs_x",
        dist.Dirichlet(0.9 * torch.eye(hidden_dim).expand(hidden_dim, hidden_dim, hidden_dim) + 0.1).to_event(2),
    )
    probs_y = pyro.sample(
        "probs_y",
        dist.Beta(0.1, 0.9).expand([hidden_dim, data_dim]).to_event(2),
    )

    if not is_guide:
        mask = torch.arange(max_length)[..., None] < lengths[..., None, None] # (num_sequences, max_length, 1)
        with poutine.mask(mask=mask):
            x_prev_prev = pyro.sample(  # (x_prev_prev | 1, 1, 1)
                "x",
                Categorical(logits=torch.zeros(hidden_dim)),  # (hidden_dim,)
                infer={"enumerate": "parallel", "_do_not_trace": True, "is_auxiliary": True},
            )
            x_prev_prev = cat([torch.zeros(2, 1, dtype=torch.long), x_prev_prev.repeat(1, num_sequences, max_length-2, 1)], dim=-2)  # (x_prev_prev | num_sequences, max_length, 1)

            x_prev = pyro.sample(  # (x_prev, 1 | 1, 1, 1)
                "x",
                Categorical(logits=torch.zeros(hidden_dim)),  # (hidden_dim,)
                infer={"enumerate": "parallel", "_do_not_trace": True, "is_auxiliary": True},
            )
            x_prev = cat([torch.zeros(1, 1, dtype=torch.long), x_prev.repeat(1, 1, num_sequences, max_length-1, 1)], dim=-2)  # (x_prev, 1 | num_sequences, max_length, 1)

            x_curr = pyro.sample(  # (x_curr, 1, 1 | 1, 1, 1)
                "x",
                Categorical(logits=probs_x[x_prev_prev, x_prev]),  # (x_prev, x_prev_prev | num_sequences, max_length, 1, hidden_dim)
                infer={"enumerate": "parallel"},
            )  # log_prob: (x_curr, x_prev, x_prev_prev | num_sequences, max_length, 1)

            pyro.sample(
                "y",
                Bernoulli(logits=probs_y[x_curr.squeeze(-1)].log()),  # (x_curr, 1, 1 | num_sequences, max_length, data_dim)
                obs=sequences, # (num_sequences, max_length, data_dim)
            )  # log_prob: (x_curr, 1, 1 | num_sequences, max_length, data_dim)


def svi_seq(model, guide, optim, *args, **kwargs):
    num_sequences, max_length, data_dim = args[0].shape

    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = poutine.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = poutine.trace(poutine.replay(poutine.enum(model, first_available_dim=-1), tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()

        with time_it() as t_reduce:
            log_prob_x_0 = torch.stack([tr_model.nodes[f"x_{i}_0"]["log_prob"]
                                        for i in range(num_sequences)], dim=-1)  # (x_0, | num_sequences)
            log_prob_x_1 = torch.stack([tr_model.nodes[f"x_{i}_1"]["log_prob"].permute([1, 0])
                                        for i in range(num_sequences)], dim=-1)  # (x_0, x_1 | num_sequences)
            log_prob_x_2_T = torch.stack([torch.stack([tr_model.nodes[f"x_{i}_{j}"]["log_prob"].permute([1, 0, 2] if j % 3 == 0 else [0, 2, 1] if j % 3 == 1 else [2, 1, 0])
                                                       for j in range(2, max_length)], dim=-1)
                                                       for i in range(num_sequences)], dim=-1)  # (x_prev_prev, x_prev, x_curr | max_length-2, num_sequences)
            log_prob_y  = torch.stack([torch.stack([torch.stack([tr_model.nodes[f"y_{i}_{j}_{k}"]["log_prob"].squeeze()
                                                                 for k in range(data_dim)], dim=-1)
                                                                 for j in range(max_length)], dim=-1)
                                                                 for i in range(num_sequences)], dim=-1)  # (x_curr | data_dim, max_length, num_sequences)

            log_prob_probs_x = tr_model.nodes["probs_x"]["log_prob"]
            log_prob_probs_y = tr_model.nodes["probs_y"]["log_prob"]

            loss = marginalize(log_prob_x_0, log_prob_x_1, log_prob_x_2_T, log_prob_y, log_prob_probs_x, log_prob_probs_y)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_seq(sequences, lengths, hidden_dim=16, is_guide=False):
    num_sequences, max_length, data_dim = sequences.shape
    probs_x = pyro.sample(
        "probs_x",
        dist.Dirichlet(0.9 * torch.eye(hidden_dim).expand(hidden_dim, hidden_dim, hidden_dim) + 0.1).to_event(2),
    )
    probs_y = pyro.sample(
        "probs_y",
        dist.Beta(0.1, 0.9).expand([hidden_dim, data_dim]).to_event(2),
    )

    if not is_guide:
        for i in range(num_sequences):
            x_curr, x_prev = 0, 0
            for j in pyro.markov(range(max_length), history=2):
                with poutine.mask(mask=(j < lengths[i])):
                    x_prev, x_curr = x_curr, pyro.sample(
                        "x_{}_{}".format(i, j),
                        Categorical(logits=probs_x[x_prev, x_curr].log()),
                        infer={"enumerate": "parallel"},
                    )
                    for k in range(data_dim):
                        pyro.sample(
                            "y_{}_{}_{}".format(i, j, k),
                            Bernoulli(logits=probs_y[x_curr, k].log()),
                            obs=sequences[i, j, k],
                        )
    

def svi_plate(model, guide, optim, *args, **kwargs):
    num_sequences, max_length, data_dim = args[0].shape

    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = poutine.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = poutine.trace(poutine.replay(poutine.enum(model, first_available_dim=-3), tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()

        with time_it() as t_reduce:
            log_prob_x_0 = tr_model.nodes["x_0"]["log_prob"].permute([0, 2, 1]).squeeze(-2)  # (x_0 | num_sequences)
            log_prob_x_1 = tr_model.nodes["x_1"]["log_prob"].permute([1, 0, 3, 2]).squeeze(-2)  # (x_0, x_1 | num_sequences)
            log_prob_x_2_T = [tr_model.nodes[f"x_{i}"]["log_prob"] for i in range(2, max_length)]

            for i in range(len(log_prob_x_2_T)):
                if i % 3 == 0:
                    log_prob_x_2_T[i] = log_prob_x_2_T[i].permute([2, 1, 0, 3, 4])
                elif i % 3 == 1:
                    log_prob_x_2_T[i] = log_prob_x_2_T[i].permute([1, 0, 2, 3, 4])
                else:
                    log_prob_x_2_T[i] = log_prob_x_2_T[i].permute([0, 2, 1, 3, 4])
            log_prob_x_2_T = torch.stack(log_prob_x_2_T).permute([1, 2, 3, 5, 0, 4]).squeeze(-3)  # (x_prev_prev, x_prev, x_curr | max_length-2, num_sequences)

            log_prob_y = torch.stack([tr_model.nodes[f"y_{i}"]["log_prob"].squeeze() for i in range(129)]).permute([1, 3, 0, 2])  # (x_curr | data_dim, max_length, num_sequences)

            log_prob_probs_x = tr_model.nodes["probs_x"]["log_prob"]
            log_prob_probs_y = tr_model.nodes["probs_y"]["log_prob"]

            loss = marginalize(log_prob_x_0, log_prob_x_1, log_prob_x_2_T, log_prob_y, log_prob_probs_x, log_prob_probs_y)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_plate(sequences, lengths, hidden_dim=16, is_guide=False):
    num_sequences, max_length, data_dim = sequences.shape
    probs_x = pyro.sample(
        "probs_x",
        dist.Dirichlet(0.9 * torch.eye(hidden_dim).expand(hidden_dim, hidden_dim, hidden_dim) + 0.1).to_event(2),
    )
    probs_y = pyro.sample(
        "probs_y",
        dist.Beta(0.1, 0.9).expand([hidden_dim, data_dim]).to_event(2),
    )

    if not is_guide:
        tones_plate = pyro.plate("tones", data_dim, dim=-1)
        with pyro.plate("sequences", num_sequences, dim=-2):
            x_curr, x_prev = 0, 0
            for j in pyro.markov(range(max_length), history=2):
                with poutine.mask(mask=(j < lengths).unsqueeze(-1)):
                    x_prev, x_curr = x_curr, pyro.sample(
                        "x_{}".format(j),
                        Categorical(logits=probs_x[x_prev, x_curr].log()),
                        infer={"enumerate": "parallel"},
                    )
                    with tones_plate:
                        pyro.sample(
                            "y_{}".format(j),
                            Bernoulli(logits=probs_y[x_curr.squeeze(-1)].log()),
                            obs=sequences[:, j]
                        )


def svi_vmarkov(model, guide, optim, *args, **kwargs):
    num_sequences, max_length, data_dim = args[0].shape

    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = trace(guide).get_trace(*args, **kwargs)
        with time_it() as t_model:
            with enum():
                tr_model = trace(replay(model, tr_guide)).get_trace(*args, **kwargs)

        with time_it() as t_reduce:
            log_prob_x_0 = tr_model.nodes["x_0"]["funsor"]["log_prob"].data  # (x_0, | num_sequences)
            log_prob_x_1 = tr_model.nodes["x_1"]["funsor"]["log_prob"].data.permute([1, 0, 2])  # (x_0, x_1 | num_sequences)
            log_prob_x_2_T = tr_model.nodes[f"x_slice(2, {max_length}, None)"]["funsor"]["log_prob"].data.permute([2, 1, 0, 4, 3]) # (x_prev_prev, x_prev, x_curr | max_length-2, num_sequences)

            log_prob_y_0 = tr_model.nodes["y_0"]["funsor"]["log_prob"].data.permute([0, 2, 1])  # (x_0 | data_dim, num_sequences)
            log_prob_y_1 = tr_model.nodes["y_1"]["funsor"]["log_prob"].data.permute([0, 2, 1])  # (x_1 | data_dim, num_sequences)
            log_prob_y_2_T = tr_model.nodes[f"y_slice(2, {max_length}, None)"]["funsor"]["log_prob"].data.permute([0, 3, 2, 1]) # (x_curr | data_dim, max_length-2, num_sequences)
            log_prob_y = torch.cat([log_prob_y_0.unsqueeze(-2), log_prob_y_1.unsqueeze(-2), log_prob_y_2_T], dim=-2)  # (x_curr | data_dim, max_length, num_sequences)

            log_prob_probs_x = tr_model.nodes["probs_x"]["funsor"]["log_prob"].data
            log_prob_probs_y = tr_model.nodes["probs_y"]["funsor"]["log_prob"].data

            loss = marginalize(log_prob_x_0, log_prob_x_1, log_prob_x_2_T, log_prob_y, log_prob_probs_x, log_prob_probs_y)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_vmarkov(sequences, lengths, hidden_dim=16, is_guide=False):
    num_sequences, max_length, data_dim = sequences.shape
    probs_x = pyro.sample(
        "probs_x",
        dist.Dirichlet(0.9 * torch.eye(hidden_dim).expand(hidden_dim, hidden_dim, hidden_dim) + 0.1).to_event(2),
    )
    probs_y = pyro.sample(
        "probs_y",
        dist.Beta(0.1, 0.9).expand([hidden_dim, data_dim]).to_event(2),
    )

    if not is_guide:
        tones_plate = plate("tones", data_dim, dim=-1)
        with plate("sequences", num_sequences, dim=-3) as batch:
            batch = batch.unsqueeze(-1)
            x_curr, x_prev = 0, 0
            for j in vectorized_markov(None, "lengths", max_length, dim=-2, history=2):
                with mask(mask=(j < lengths.unsqueeze(-1)).unsqueeze(-1)):
                    x_prev, x_curr = x_curr, pyro.sample(
                        "x_{}".format(j),
                        Categorical(logits=probs_x[x_prev, x_curr].log()),
                        infer={"enumerate": "parallel"},
                    )
                    with tones_plate:
                        pyro.sample(
                            "y_{}".format(j),
                            Bernoulli(logits=probs_y[x_curr.squeeze(-1)].log()),
                            obs=sequences[batch, j]
                        )


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
            log_prob_x = tr_model.nodes["x"]["log_prob"]  # (x_prev_prev, x_prev, x_curr | 1,        max_length, num_sequences)
            log_prob_x_0 = log_prob_x[0, 0, :, 0, 0]  # (x_0 | num_sequences)
            log_prob_x_1 = log_prob_x[0, :, :, 0, 1]  # (x_0, x_1 | num_sequences)
            log_prob_x_2_T = log_prob_x[:, :, :, 0, 2:]  # (x_prev_prev, x_prev, x_curr | max_length-2, num_sequences)
            log_prob_y = tr_model.nodes["y"]["log_prob"]  # (x_curr | data_dim, max_length, num_sequences)

            log_prob_probs_x = tr_model.nodes["probs_x"]["log_prob"]
            log_prob_probs_y = tr_model.nodes["probs_y"]["log_prob"]

            loss = marginalize(log_prob_x_0, log_prob_x_1, log_prob_x_2_T, log_prob_y, log_prob_probs_x, log_prob_probs_y)

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
def model_ours(s: vec.State, sequences, lengths, hidden_dim=16, is_guide=False):
    num_sequences, max_length, data_dim = sequences.shape
    probs_x = pyro.sample(
        "probs_x",
        dist.Dirichlet(0.9 * torch.eye(hidden_dim).expand(hidden_dim, hidden_dim, hidden_dim) + 0.1).to_event(2),
    )
    probs_y = pyro.sample(
        "probs_y",
        dist.Beta(0.1, 0.9).expand([hidden_dim, data_dim]).to_event(2),
    )

    if not is_guide:
        for i in vec.range("sequences", num_sequences, vectorized=True):
            s.x_curr, s.x_prev = 0, 0
            for j in vec.range("lengths", max_length, vectorized=True):
                with poutine.mask(mask=(j < lengths[i])):
                    s.x_prev, s.x_curr = s.x_curr, pyro.sample(
                        "x",
                        Categorical(logits=Index(probs_x)[s.x_prev, s.x_curr].log()),
                        infer={"enumerate": "parallel"},
                    )
                    for k in vec.range("tones", data_dim, vectorized=True):
                        pyro.sample(
                            "y",
                            Bernoulli(logits=Index(probs_y)[s.x_curr, k].log()),
                            obs=Index(sequences)[i, j, k],
                        )


def guide(model, sequences, lengths, hidden_dim=16):
    _, _, data_dim = sequences.shape

    init_probs_x = dist.Dirichlet(0.9 * torch.eye(hidden_dim).expand(hidden_dim, hidden_dim, hidden_dim) + 0.1).sample()
    init_probs_y = dist.Beta(0.1, 0.9).expand([hidden_dim, data_dim]).sample()
    _guide = AutoDelta(
        partial(model, is_guide=True),
        init_loc_fn=lambda msg: {"probs_x": init_probs_x, "probs_y": init_probs_y}[msg["name"]],
    )
    return _guide


def data(args):
    sequences = torch.load(os.path.join(path, "dataset/polyphony/sequences.pt"), map_location=torch.get_default_device(), weights_only=True)
    lengths = torch.load(os.path.join(path, "dataset/polyphony/lengths.pt"), map_location=torch.get_default_device(), weights_only=True)
    sequences = sequences[:args.num_batch]
    lengths = lengths[:args.num_batch]
    present_notes = (sequences == 1).sum(0).sum(0) > 0
    sequences = sequences[:, :, present_notes]
    hidden_dim = 16
    return sequences, lengths, hidden_dim


def optim(args):
    return Adam({"lr": args.lr})