import os
import sys
import argparse
import itertools

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.dont_write_bytecode = True

import pyro
import torch
import vectorized_loop as vec
import multiprocessing as mp

from pyro import poutine
from pyro import distributions as dist
from functools import partial
from hmc import run_HMC
from benchmark import hmm_ord1, hmm_ord2, arm, tcm, nhmm_train, nhmm_stock

from pyro.contrib import funsor

from util import time_it_context, setup_logger, suppress_backward, suppress_reset_peak_memory_stats, set_device


def log_prob_fn_hmm(args, params):
    benchmark = {"hmm-ord1": hmm_ord1, "hmm-ord2": hmm_ord2, "nhmm-train": nhmm_train, "nhmm-stock": nhmm_stock}[args.benchmark]

    with suppress_backward(), suppress_reset_peak_memory_stats(), time_it_context(True):
        new_params = {}
        for k, v in params.items():
            if "probs" in k:
                v = torch.nn.functional.softmax(v, dim=-1)
            elif "scale" in k:
                v = torch.exp(v) + 1e-3
            new_params[k] = v

        model = getattr(benchmark, f"model_{args.method}")
        if model is None:
            raise NotImplementedError(f"Method {args.method} not implemented for benchmark {args.benchmark}")

        model = poutine.condition(model, new_params)
        guide = lambda *args, **kwargs: None
        optim = lambda params: None

        (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory) = getattr(benchmark, f"svi_{args.method}")(model, guide, optim, *benchmark.data(args))

    return -loss


def initial_params_hmm(args):
    benchmark = {"hmm-ord1": hmm_ord1, "hmm-ord2": hmm_ord2, "nhmm-train": nhmm_train, "nhmm-stock": nhmm_stock}[args.benchmark]

    data = getattr(benchmark, "data")(args)
    sequences, hidden_dim = data[0], data[-1]

    if args.benchmark == "hmm-ord1":
        data_dim = sequences.shape[-1]
        probs_x = torch.randn([hidden_dim, hidden_dim])
        probs_y = torch.randn([hidden_dim, data_dim])
        params = {"probs_x": probs_x, "probs_y": probs_y}

    elif args.benchmark == "hmm-ord2":
        data_dim = sequences.shape[-1]
        probs_x = torch.randn([hidden_dim, hidden_dim, hidden_dim])
        probs_y = torch.randn([hidden_dim, data_dim])
        params = {"probs_x": probs_x, "probs_y": probs_y}
        
    elif args.benchmark == "nhmm-train":
        hM, hD, hH = hidden_dim
        probsMM = torch.randn([hM+1, hM])
        probsMDD = torch.randn([hM, hD+1, hD])
        probsDHH = torch.randn([hD, hH+1, hH])
        locs = torch.randn([hM, hD, hH])
        scale = torch.randn([])
        params = {"probsMM": probsMM, "probsMDD": probsMDD, "probsDHH": probsDHH, "locs": locs, "scale": scale}

    elif args.benchmark == "nhmm-stock":
        nB, nY, nQ, nD = sequences.shape
        hY, hQ = hidden_dim
        probsYY = torch.randn([hY+1, hY])
        probsYQ = torch.randn([hY, hQ+1, hQ])
        alpha = torch.randn([hY, hQ]) * 1e-2
        phi = torch.randn([]) * 1e-2 + 1.0
        scale = torch.randn([]) * 1e-2 - 1.0
        log_h = torch.randn([nD, nQ, nY, nB]) * torch.exp(scale)
        if args.method == "seq":
            params = {"probsYY": probsYY, "probsYQ": probsYQ, "alpha": alpha, "phi": phi, "scale": scale, **{f"log_h_{k}_{j}_{i}_{b}": log_h[k, j, i, b] for k in range(nD) for j in range(nQ) for i in range(nY) for b in range(nB)}}
        elif args.method == "plate":
            params = {"probsYY": probsYY, "probsYQ": probsYQ, "alpha": alpha, "phi": phi, "scale": scale, **{f"log_h_{i}_{j}_{k}": log_h[k, j, i] for k in range(nD) for j in range(nQ) for i in range(nY)}}
        elif args.method in {"manual", "ours"}:
            params = {"probsYY": probsYY, "probsYQ": probsYQ, "alpha": alpha, "phi": phi, "scale": scale, "log_h": log_h}

    return params


def log_prob_fn_tcm(args, params):
    model = getattr(tcm, f"model_{args.method}")
    data, = tcm.data(args)

    new_params = params
    handlers = {"seq": poutine, "ours": vec}[args.method]
    tr_model = handlers.trace(handlers.condition(model, new_params)).get_trace(data)
    return tr_model.log_prob_sum()


def initial_params_tcm(args):
    data, = tcm.data(args)
    B, N = data.shape
    theta = torch.full((N, B), 20.0) + torch.randn((N, B))
    q_noise = torch.full((N, B), 0.5) + torch.randn((N, B)) * 1e-1

    if args.method == "seq":
        initial_params = {
            **{f"theta_{b}_{i}": theta[i, b] for b in range(B) for i in range(N)},
            **{f"q_noise_{b}_{i}": q_noise[i, b] for b in range(B) for i in range(N)},
        }

    elif args.method == "ours":
        initial_params = {
            "theta": theta,
            "q_noise": q_noise,
        }
        
    return initial_params


def log_prob_fn_arm(args, params):
    model = getattr(arm, f"model_{args.method}")
    data, K_max = arm.data(args)

    if args.method in {"manual", "seq"}:
        tr_model = poutine.trace(poutine.condition(model, params)).get_trace(data, K_max)

    elif args.method == "vmarkov":
        N = len(data)
        K = min(int(params["K_tilde"]), K_max)
        x = params["x"]
        new_params = {"K_tilde": params["K_tilde"], "coeff": params["coeff"]}
        new_params.update({f"x_{i}": x[i] for i in range(K)})
        new_params.update({f"x_slice({i}, {N-K+i}, None)": x[i:N-K+i] for i in range(K+1)})
        tr_model = funsor.handlers.trace(funsor.handlers.condition(model, new_params)).get_trace(data, K_max)

    elif args.method == "ours":
        tr_model = vec.trace(vec.vectorize(vec.condition(model, params))).get_trace(data, K_max)

    return tr_model.log_prob_sum()


def initial_params_arm(args):
    data, K_max = arm.data(None)
    N = len(data)

    K_tilde = dist.Poisson(10.0).sample()
    coeff = torch.randn(K_max)
    x = torch.randn(N)
    K = min(int(K_tilde), K_max)

    if args.method == "seq":
        initial_params = {
            "K_tilde": K_tilde,
            "coeff": coeff,
            **{f"x_{i}": x[i] for i in range(N)},
        }        

    elif args.method in {"manual", "vmarkov", "ours"}:
        initial_params = {
            "K_tilde": K_tilde,
            "coeff": coeff,
            "x": x,
        }
        
    return initial_params


def gibbs_step_arm(log_prob_fn, hmc_params, gibbs_params):
    K_tilde = gibbs_params["K_tilde"]
    new_K_tilde = dist.Poisson(K_tilde).sample()

    lp = log_prob_fn({**hmc_params, "K_tilde": K_tilde})
    new_lp = log_prob_fn({**hmc_params, "K_tilde": new_K_tilde})
    lq = dist.Poisson(new_K_tilde).log_prob(K_tilde)
    new_lq = dist.Poisson(K_tilde).log_prob(new_K_tilde)

    log_acc_rate = (new_lp - lp) + (lq - new_lq)
    acc_rate = torch.minimum(torch.tensor(1.0), log_acc_rate.exp())
    is_accept = torch.rand_like(log_acc_rate).log() < log_acc_rate

    new_K_tilde = torch.where(is_accept, new_K_tilde, K_tilde)
    return {"K_tilde": new_K_tilde}, acc_rate.item()


def gibbs_step_skip(log_prob_fn, hmc_params, gibbs_params, *args, **kwargs):
    return gibbs_params, 1.0


def run(args):
    if not args.dry:
        log_dir = os.path.join(path, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger = setup_logger(os.path.join(log_dir, f"samples-{args.benchmark}-{args.method}-seed{args.seed}.log"))

    if args.benchmark in ["hmm-ord1", "hmm-ord2", "nhmm-train", "nhmm-stock"]:
        log_prob_fn = partial(log_prob_fn_hmm, args)
        initial_params = initial_params_hmm(args)
        gibbs_step = gibbs_step_skip
        gibbs_vars = []

    elif args.benchmark == "tcm":
        log_prob_fn = partial(log_prob_fn_tcm, args)
        initial_params = initial_params_tcm(args)
        gibbs_step = gibbs_step_skip
        gibbs_vars = []

    elif args.benchmark == "arm":
        log_prob_fn = partial(log_prob_fn_arm, args)
        initial_params = initial_params_arm(args)
        gibbs_step = gibbs_step_arm
        gibbs_vars = ["K_tilde"]

    hmc_vars = [var for var in initial_params if var not in gibbs_vars]

    if not args.dry:
        header = ["time", "chain", "memory"]
        for var in hmc_vars + gibbs_vars:
            shape = initial_params[var].shape
            indices = itertools.product(*(range(s) for s in shape))
            for index in indices:
                index_str = str(list(index))
                header.append(f"{var}{index_str}".replace(", ", "]["))
        logger.info(",".join(header))

    with mp.Pool(processes=args.num_chains) as pool:
        pool.map(partial(run_HMC, log_prob_fn,
                                  initial_params,
                                  gibbs_step,
                                  gibbs_vars,
                                  hmc_vars,
                                  args),
                 range(args.num_chains))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="hmm-ord1", choices=["hmm-ord1", "hmm-ord2", "nhmm-train", "nhmm-stock", "tcm", "arm"])
    parser.add_argument("--method", type=str, default="ours", choices=["seq", "plate", "vmarkov", "discHMM", "manual", "ours"])
    parser.add_argument("--num_batch", type=int, default=None)
    parser.add_argument("--num_chains", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=10000000)
    parser.add_argument("--step_size", type=float, default=1e-2)
    parser.add_argument("--num_leapfrog_steps", type=int, default=10)
    parser.add_argument("--num_hmc_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--dry", action="store_true", default=False)
    args = parser.parse_args()

    print(f">> Run {args.method} for {args.benchmark}")
    print(f">> Device: {args.cuda if args.cuda >= 0 else 'cpu'}")
    print(f">> Seed: {args.seed}")
    print(f">> Number of chains: {args.num_chains}")
    print(f">> Number of samples: {args.num_samples}")

    mp.set_start_method("spawn", force=True)

    set_device(args.cuda)
    pyro.clear_param_store()
    vec.clear_allocators()
    pyro.set_rng_seed(args.seed)

    run(args)