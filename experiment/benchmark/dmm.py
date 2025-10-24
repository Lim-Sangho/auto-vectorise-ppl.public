import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, "../"))
sys.path.append(os.path.join(path, "../../"))
sys.dont_write_bytecode = True

import torch
from torch import nn

import pyro
import pyro.distributions as dist

from pyro import poutine
from pyro.optim import Adam

from pyro.contrib.funsor import vectorized_markov, plate, condition
from pyro.contrib.funsor.handlers import mask, trace

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


def reverse_sequences(mini_batch, seq_lengths):
    reversed_mini_batch = torch.zeros_like(mini_batch)
    for b in range(mini_batch.size(0)):
        T = seq_lengths[b]
        time_slice = torch.arange(T - 1, -1, -1, device=mini_batch.device)
        reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence
    return reversed_mini_batch


def pad_and_reverse(rnn_output, seq_lengths):
    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    reversed_output = reverse_sequences(rnn_output, seq_lengths)
    return reversed_output


def get_mini_batch_mask(mini_batch, seq_lengths):
    mask = torch.zeros(mini_batch.shape[0:2])
    for b in range(mini_batch.shape[0]):
        mask[b, 0 : seq_lengths[b]] = torch.ones(seq_lengths[b])
    return mask


def get_mini_batch(mini_batch_indices, sequences, seq_lengths, cuda=False):
    seq_lengths = seq_lengths[mini_batch_indices]
    _, sorted_seq_length_indices = torch.sort(seq_lengths)
    sorted_seq_length_indices = sorted_seq_length_indices.flip(0)
    sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]
    sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices]

    max_length = torch.max(seq_lengths)
    mini_batch = sequences[sorted_mini_batch_indices, 0:max_length, :]
    mini_batch_reversed = reverse_sequences(mini_batch, sorted_seq_lengths)
    mini_batch_mask = get_mini_batch_mask(mini_batch, sorted_seq_lengths)

    if cuda:
        mini_batch = mini_batch.cuda()
        mini_batch_mask = mini_batch_mask.cuda()
        mini_batch_reversed = mini_batch_reversed.cuda()

    mini_batch_reversed = nn.utils.rnn.pack_padded_sequence(
        mini_batch_reversed, sorted_seq_lengths, batch_first=True
    )

    return mini_batch, mini_batch_reversed, mini_batch_mask, sorted_seq_lengths


def marginalize(log_prob_z_guide, log_prob_z_model, log_prob_x_model):
    log_prob_z_guide, log_prob_z_model, log_prob_x_model = \
        map(torch.Tensor.contiguous, (log_prob_z_guide, log_prob_z_model, log_prob_x_model))

    log_prob_z_guide = log_prob_z_guide.sum()
    log_prob_z_model = log_prob_z_model.sum()
    log_prob_x_model = log_prob_x_model.sum()
    loss = log_prob_z_guide - log_prob_z_model - log_prob_x_model
    return loss


class Emitter(nn.Module):

    def __init__(self, input_dim, z_dim, emission_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, z_t):
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        ps = torch.sigmoid(self.lin_hidden_to_input(h2))
        return ps


class GatedTransition(nn.Module):

    def __init__(self, z_dim, transition_dim):
        super().__init__()
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        return loc, scale


class Combiner(nn.Module):

    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        loc = self.lin_hidden_to_loc(h_combined)
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        return loc, scale


class Trainer(nn.Module):

    def __init__(self, input_dim=88, z_dim=100, emission_dim=100, transition_dim=200, rnn_dim=600, num_layers=1, use_cuda=False):
        super().__init__()
        self.z_dim = z_dim
        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)
        
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=rnn_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers,
        )

        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def combiner_output(self, rnn_output):
        num_sequences, max_length, rnn_dim = rnn_output.shape
        e_z = torch.randn((num_sequences, max_length, self.z_dim))  # (num_sequences, max_length, z_dim)

        zs, z_locs, z_scales = [], [], []
        z = self.z_q_0
        for t in range(max_length):
            z_loc, z_scale = self.combiner(z, rnn_output[:, t])  # (num_sequences, z_dim)
            z = z_loc + z_scale * e_z[:, t]  # (num_sequences, z_dim)
            zs.append(z)
            z_locs.append(z_loc)
            z_scales.append(z_scale)

        zs = torch.stack(zs, dim=1).contiguous()  # (num_sequences, max_length, z_dim)
        z_locs = torch.stack(z_locs, dim=1).contiguous()  # (num_sequences, max_length, z_dim)
        z_scales = torch.stack(z_scales, dim=1).contiguous()  # (num_sequences, max_length, z_dim)

        return zs, z_locs, z_scales


def svi_manual(model, guide, optim, *args, **kwargs):
    num_sequences, max_length, data_dim = args[0].shape

    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = poutine.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            poutine.replay_messenger.ReplayMessenger._pyro_sample = _pyro_sample
            tr_model = poutine.trace(poutine.replay(model, tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()

        with time_it() as t_reduce:
            log_prob_z_guide = tr_guide.nodes["z"]["log_prob"]
            log_prob_z_model = tr_model.nodes["z"]["log_prob"]
            log_prob_x_model = tr_model.nodes["x"]["log_prob"]

            loss = marginalize(log_prob_z_guide, log_prob_z_model, log_prob_x_model)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_manual(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, trainer):
    num_sequences, max_length, data_dim = mini_batch.shape
    pyro.module("trainer", trainer)

    with poutine.mask(mask=mini_batch_mask.bool()):  # (num_sequences, max_length)
        z_prev = pyro.sample(  # (num_sequences, max_length | z_dim)
            "z",
            dist.Normal(torch.zeros(num_sequences, max_length, trainer.z_dim),
                        torch.ones(num_sequences, max_length, trainer.z_dim)).to_event(1),  # (num_sequences, max_length | z_dim)
            infer={"_do_not_trace": True, "is_auxiliary": True},
        )
        z_prev = torch.cat([trainer.z_0.repeat(num_sequences, 1, 1), z_prev[:, :-1, :]], dim=-2)  # (num_sequences, max_length | z_dim)

        z_curr = pyro.sample(  # (num_sequences, max_length | z_dim)
            "z",
            dist.Normal(*trainer.trans(z_prev)).to_event(1),  # (num_sequences, max_length | z_dim)
        )  # log_prob: (num_sequences, max_length)

        pyro.sample(
            "x",
            dist.Bernoulli(trainer.emitter(z_curr)).to_event(1),  # (num_sequences, max_length | data_dim)
            obs=mini_batch,  # (num_sequences, max_length, data_dim)
        )  # log_prob: (num_sequences, max_length)


def guide_manual(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, trainer):
    num_sequences, max_length, data_dim = mini_batch.shape
    pyro.module("trainer", trainer)

    h_0_contig = trainer.h_0.expand(1, num_sequences, trainer.rnn.hidden_size).contiguous()
    rnn_output, _ = trainer.rnn(mini_batch_reversed, h_0_contig)
    rnn_output = pad_and_reverse(rnn_output, mini_batch_seq_lengths).contiguous()
    z, z_loc, z_scale = trainer.combiner_output(rnn_output)  # (num_sequences, max_length, z_dim)

    with poutine.mask(mask=mini_batch_mask.bool()):  # (num_sequences, max_length)
        pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1), obs=z)  # (num_sequences, max_length | z_dim)


def svi_seq(model, guide, optim, *args, **kwargs):
    num_sequences, max_length, data_dim = args[0].shape

    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = poutine.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = poutine.trace(poutine.replay(model, tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()

        with time_it() as t_reduce:
            log_prob_z_guide = torch.stack([torch.stack([tr_guide.nodes[f"z_{b}_{t}"]["log_prob"]
                for t in range(max_length)], dim=-1)
                for b in range(num_sequences)], dim=-1)
            log_prob_z_model = torch.stack([torch.stack([tr_model.nodes[f"z_{b}_{t}"]["log_prob"]
                for t in range(max_length)], dim=-1)
                for b in range(num_sequences)], dim=-1)
            log_prob_x_model = torch.stack([torch.stack([tr_model.nodes[f"x_{b}_{t}"]["log_prob"]
                for t in range(max_length)], dim=-1)
                for b in range(num_sequences)], dim=-1)

            loss = marginalize(log_prob_z_guide, log_prob_z_model, log_prob_x_model)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_seq(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, trainer):
    num_sequences, max_length, data_dim = mini_batch.shape
    pyro.module("trainer", trainer)

    for b in range(num_sequences):
        z = trainer.z_0
        for t in range(max_length):
            with poutine.mask(mask=mini_batch_mask[b, t].bool()):
                z = pyro.sample(
                    f"z_{b}_{t}",
                    dist.Normal(*trainer.trans(z)).to_event(1),
                )
                pyro.sample(
                    f"x_{b}_{t}",
                    dist.Bernoulli(trainer.emitter(z)).to_event(1),
                    obs=mini_batch[b, t],
                )


def guide_seq(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, trainer):
    num_sequences, max_length, data_dim = mini_batch.shape
    pyro.module("trainer", trainer)

    h_0_contig = trainer.h_0.expand(1, num_sequences, trainer.rnn.hidden_size).contiguous()
    rnn_output, _ = trainer.rnn(mini_batch_reversed, h_0_contig)
    rnn_output = pad_and_reverse(rnn_output, mini_batch_seq_lengths).contiguous()
    z, z_loc, z_scale = trainer.combiner_output(rnn_output)  # (num_sequences, max_length, z_dim)

    for b in range(num_sequences):
        for t in range(max_length):
            with poutine.mask(mask=mini_batch_mask[b, t].bool()):
                pyro.sample(f"z_{b}_{t}", dist.Normal(z_loc[b, t], z_scale[b, t]).to_event(1), obs=z[b, t])  # (z_dim,)


def svi_plate(model, guide, optim, *args, **kwargs):
    num_sequences, max_length, data_dim = args[0].shape

    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = poutine.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = poutine.trace(poutine.replay(model, tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()

        with time_it() as t_reduce:
            log_prob_z_guide = torch.stack([tr_guide.nodes[f"z_{t}"]["log_prob"] for t in range(max_length)])
            log_prob_z_model = torch.stack([tr_model.nodes[f"z_{t}"]["log_prob"] for t in range(max_length)])
            log_prob_x_model = torch.stack([tr_model.nodes[f"x_{t}"]["log_prob"] for t in range(max_length)])

            loss = marginalize(log_prob_z_guide, log_prob_z_model, log_prob_x_model)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_plate(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, trainer):
    num_sequences, max_length, data_dim = mini_batch.shape
    pyro.module("trainer", trainer)

    with pyro.plate("z_minibatch", num_sequences):
        z = trainer.z_0.expand(num_sequences, trainer.z_dim)
        for t in range(max_length):
            with poutine.mask(mask=mini_batch_mask[:, t].bool()):
                z = pyro.sample(
                    f"z_{t}",
                    dist.Normal(*trainer.trans(z)).to_event(1),
                )
                pyro.sample(
                    f"x_{t}",
                    dist.Bernoulli(trainer.emitter(z)).to_event(1),
                    obs=mini_batch[:, t],
                )


def guide_plate(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, trainer):
    num_sequences, max_length, data_dim = mini_batch.shape
    pyro.module("trainer", trainer)

    h_0_contig = trainer.h_0.expand(1, num_sequences, trainer.rnn.hidden_size).contiguous()
    rnn_output, _ = trainer.rnn(mini_batch_reversed, h_0_contig)
    rnn_output = pad_and_reverse(rnn_output, mini_batch_seq_lengths).contiguous()
    z, z_loc, z_scale = trainer.combiner_output(rnn_output)  # (num_sequences, max_length, z_dim)

    with pyro.plate("z_minibatch", num_sequences):
        for t in range(max_length):
            with poutine.mask(mask=mini_batch_mask[:, t].bool()):
                pyro.sample(f"z_{t}", dist.Normal(z_loc[:, t], z_scale[:, t]).to_event(1), obs=z[:, t])  # (num_sequences | z_dim)


def svi_vmarkov(model, guide, optim, *args, **kwargs):
    num_sequences, max_length, data_dim = args[0].shape

    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()

        with time_it() as t_model:
            zs = tr_guide.nodes["z"]["value"]  # (num_sequences, max_length, z_dim)

            data = {
                "z_0": zs[:, 0:1, :],
                f"z_slice(0, {max_length-1}, None)": zs[:, 0:max_length-1, :],
                f"z_slice(1, {max_length}, None)": zs[:, 1:max_length, :],
            }

            tr_model = trace(condition(model, data)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()

        with time_it() as t_reduce:
            log_prob_z_0_model = tr_model.nodes["z_0"]["log_prob"]
            log_prob_z_1_T_model = tr_model.nodes[f"z_slice(1, {max_length}, None)"]["log_prob"]
            log_prob_x_0_model = tr_model.nodes["x_0"]["log_prob"]
            log_prob_x_1_T_model = tr_model.nodes[f"x_slice(1, {max_length}, None)"]["log_prob"]

            log_prob_z_guide = tr_guide.nodes["z"]["log_prob"].T
            log_prob_z_model = torch.cat([log_prob_z_0_model, log_prob_z_1_T_model], dim=-1).T
            log_prob_x_model = torch.cat([log_prob_x_0_model, log_prob_x_1_T_model], dim=-1).T

            loss = marginalize(log_prob_z_guide, log_prob_z_model, log_prob_x_model)

        with time_it() as t_backward:
            loss.backward()
            params = {name: value.unconstrained() for name, value in pyro.get_param_store().items()}
            grads = {name: value.grad for name, value in params.items()}
            optim(params.values())

    memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pyro.infer.util.zero_grads(params.values())

    return (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory)


def model_vmarkov(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, trainer):
    num_sequences, max_length, data_dim = mini_batch.shape
    pyro.module("trainer", trainer)

    with plate("z_minibatch", num_sequences, dim=-2) as batch:
        z = trainer.z_0.expand(num_sequences, 1, trainer.z_dim)  # (num_sequences, 1, z_dim)
        batch = batch.unsqueeze(-1)  # (num_sequences, 1)
        for t in vectorized_markov(None, "lengths", max_length, dim=-1, history=1):
            with mask(mask=mini_batch_mask[batch, t].bool()):
                z = pyro.sample(
                    "z_{}".format(t),
                    dist.Normal(*trainer.trans(z)).to_event(1),
                )
                pyro.sample(
                    "x_{}".format(t),
                    dist.Bernoulli(trainer.emitter(z)).to_event(1),
                    obs=mini_batch[batch, t],
                )


def guide_vmarkov(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, trainer):
    num_sequences, max_length, data_dim = mini_batch.shape
    pyro.module("trainer", trainer)

    h_0_contig = trainer.h_0.expand(1, num_sequences, trainer.rnn.hidden_size).contiguous()
    rnn_output, _ = trainer.rnn(mini_batch_reversed, h_0_contig)
    rnn_output = pad_and_reverse(rnn_output, mini_batch_seq_lengths).contiguous()
    z, z_loc, z_scale = trainer.combiner_output(rnn_output)  # (num_sequences, max_length, z_dim)

    with plate("z_minibatch", num_sequences, dim=-2) as batch:
        batch = batch.unsqueeze(-1)  # (num_sequences, 1)
        with plate("time_length", max_length, dim=-1) as t:
            with mask(mask=mini_batch_mask[batch, t].bool()):
                pyro.sample("z", dist.Normal(z_loc[batch, t], z_scale[batch, t]).to_event(1), obs=z[batch, t])  # (num_sequences, max_length | z_dim)


svi_discHMM = None
model_discHMM = None


def svi_ours(model, guide, optim, *args, **kwargs):
    num_sequences, max_length, data_dim = args[0].shape

    with time_it() as t_total:
        with time_it() as t_guide:
            tr_guide = vec.trace(guide).get_trace(*args, **kwargs)
            tr_guide.compute_score_parts()
        with time_it() as t_model:
            tr_model = vec.trace(vec.replay(model, tr_guide)).get_trace(*args, **kwargs)
            tr_model.compute_log_prob()

        with time_it() as t_reduce:
            log_prob_z_guide = tr_guide.nodes["z"]["log_prob"]
            log_prob_z_model = tr_model.nodes["z"]["log_prob"]
            log_prob_x_model = tr_model.nodes["x"]["log_prob"]

            loss = marginalize(log_prob_z_guide, log_prob_z_model, log_prob_x_model)

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
def model_ours(s: vec.State, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, trainer):
    num_sequences, max_length, data_dim = mini_batch.shape
    pyro.module("trainer", trainer)

    for b in vec.range("z_minibatch", num_sequences, vectorized=True):
        s.z = trainer.z_0
        for t in vec.range("time_length", max_length, vectorized=True):
            with poutine.mask(mask=mini_batch_mask[b, t].bool()):
                s.z = pyro.sample(
                    "z",
                    dist.Normal(*trainer.trans(s.z)).to_event(1)
                )
                pyro.sample(
                    "x",
                    dist.Bernoulli(trainer.emitter(s.z)).to_event(1),
                    obs=mini_batch[b, t],
                )


@vec.vectorize
def guide_ours(s: vec.State, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, trainer):
    num_sequences, max_length, data_dim = mini_batch.shape
    pyro.module("trainer", trainer)

    h_0_contig = trainer.h_0.expand(1, num_sequences, trainer.rnn.hidden_size).contiguous()
    rnn_output, _ = trainer.rnn(mini_batch_reversed, h_0_contig)
    rnn_output = pad_and_reverse(rnn_output, mini_batch_seq_lengths).contiguous()
    z, z_loc, z_scale = trainer.combiner_output(rnn_output)  # (num_sequences, max_length, z_dim)

    for b in vec.range("z_minibatch", num_sequences, vectorized=True):
        for t in vec.range("time_length", max_length, vectorized=True):
            with poutine.mask(mask=mini_batch_mask[b, t].bool()):
                pyro.sample(f"z", dist.Normal(z_loc[b, t], z_scale[b, t]).to_event(1), obs=z[b, t])


def guide(model, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, trainer):
    if model is model_manual:
        return guide_manual
    elif model is model_seq:
        return guide_seq
    elif model is model_plate:
        return guide_plate
    elif model is model_vmarkov:
        return guide_vmarkov
    elif model is model_ours:
        return guide_ours
    else:
        return None


def data(args):
    sequences = torch.load(os.path.join(path, "dataset/polyphony/sequences.pt"), map_location="cpu", weights_only=True)
    lengths = torch.load(os.path.join(path, "dataset/polyphony/lengths.pt"), map_location="cpu", weights_only=True)

    num_batch = len(sequences) if args.num_batch is None else args.num_batch
    batch_indices = torch.arange(num_batch).to("cpu")
    batch, batch_reversed, batch_mask, batch_seq_lengths = \
        get_mini_batch(batch_indices, sequences, lengths, cuda=(torch.get_default_device() != torch.device("cpu")))

    num_sequences, max_length, data_dim = batch.shape
    z_dim = 100
    trainer = Trainer(input_dim=data_dim, z_dim=z_dim, use_cuda=(torch.get_default_device() != torch.device("cpu")))

    return batch, batch_reversed, batch_mask, batch_seq_lengths, trainer


def optim(args):
    return Adam({"lr": args.lr})