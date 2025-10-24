import os
import sys
import argparse

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, "../"))
sys.path.append(os.path.join(path, "../../"))
sys.dont_write_bytecode = True

import pyro
import torch
import numpy as np
import pyro.distributions as dist

from pyro import poutine
from tqdm.auto import tqdm

import vectorized_loop as vec
import util


@vec.vectorize
def model_ours(s: vec.State, N, K, vectorized):
    s.x_prev = torch.zeros(K)
    s.coeff = torch.rand(K)
    for _ in vec.range("loop", N, vectorized=vectorized, device=device):
        s.x = pyro.sample("x", dist.Normal((s.x_prev * s.coeff).sum(-1), 1))
        s.x_prev = util.cat([s.x_prev[..., 1:], s.x[..., None]], dim=-1)


@vec.vectorize
def model_ours_nested(s: vec.State, N, K, vectorized):
    assert N % K == 0
    s.x_prev = torch.zeros(K)
    s.coeff = torch.rand(K)
    for _ in vec.range("outer_loop", N // K, vectorized=vectorized, device=device):
        for _ in vec.range("inner_loop", K, vectorized=False, device=device):
            s.x = pyro.sample("x", dist.Normal((s.x_prev * s.coeff).sum(-1), 1))
            s.x_prev = util.cat([s.x_prev[..., 1:], s.x[..., None]], dim=-1)


def model_seq(N, K, vectorized):
    x_prev = torch.zeros(K)
    coeff = torch.rand(K)
    for i in range(N):
        x = pyro.sample("x_{}".format(i), dist.Normal((x_prev * coeff).sum(-1), 1))
        x_prev = util.cat([x_prev[..., 1:], x[..., None]], dim=-1)


def get_time(model, N, K, module, n_sample):
    util._TIMEIT = True

    try:
        time = []
        xs = torch.randn(N)
        for i in tqdm(range(n_sample), leave=False):
            pyro.clear_param_store()
            vec.clear_allocators()
            tr_guide = poutine.Trace("flat")
            tr_guide.nodes["x"] = {"name": "x", "type": "sample", "fn": dist.Normal(0, 1).expand((N,)), "value": xs, "infer": {}}

            with util.time_it() as total_time:
                tr_model = module.trace(module.replay(model, tr_guide)).get_trace(N, K, vectorized=True)
                tr_model.log_prob_sum()

            time.append(total_time.time)

    except torch.OutOfMemoryError:
        time = [np.nan] * n_sample

    return time  # (n_sample,)


def run(args):
    '''
    Here we fix K and vary N.

    Costs: Both should be linear in N.
        1) Seq: O(Nlog(K)) ~= O(N)
        2) Ours: O(NK^2/p + Klog(p)) ~= O(N/p)

    It shows our method is particularly beneficial when N is large and K is small.
    '''

    log_dir = os.path.join(path, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for k in [1, 10, 30, 50]:
        logger = util.setup_logger(os.path.join(log_dir, f"complexity_ours_K_{k}.log"))
        logger.info("N,time")
        Ns = np.concatenate([(np.arange(1, 21) * 500).astype(int), np.arange(15000, 100001, 5000), np.arange(150000, 1000001, 50000)])
        Ks = (np.ones_like(Ns) * k).astype(int)
        for N, K in tqdm(zip(Ns.tolist(), Ks.tolist()), total=len(Ns)):
            time_ours = get_time(model_ours, N, K, vm, n_sample=100)
            logger.info(f"{N},{','.join(map(str, time_ours))}")

    for k in [1, 10, 30, 50]:
        logger = util.setup_logger(os.path.join(log_dir, f"complexity_seq_K_{k}.log"))
        logger.info("N,time")
        Ns = (np.arange(1, 21) * 500).astype(int)
        Ks = (np.ones_like(Ns) * k).astype(int)
        for N, K in tqdm(zip(Ns.tolist(), Ks.tolist()), total=len(Ns)):
            time_ours = get_time(model_seq, N, K, poutine, n_sample=100)
            logger.info(f"{N},{','.join(map(str, time_ours))}")

    '''
    Here we fix N and vary K.

    Costs:
        1) Seq: O(Nlog(K)) ~= O(log(K))
        2) Ours: O(NK^2/p + Klog(p)) ~= O(K^2/p + Klog(p))

    It would be better to focus on cases where histories (K) are small enough.
    Possible justifications for considering only small K's:
        1) In most real-world applications, K is typically small.
        2) If K is known, we can choose to use the baseline for large K and our method for small K.
        3) If K is unknown, we can use our method to determine K and do 2).
        We should verify whether the time taken to find K is negligible compared to the total inference time.
    '''

    for n in [1000, 2000, 3000, 4000]:
        logger = util.setup_logger(os.path.join(log_dir, f"complexity_ours_N_{n}.log"))
        logger.info("K,time")
        Ks = (np.arange(1, 21) * (n // 20)).astype(int)
        Ns = (np.ones_like(Ks) * n).astype(int)
        for N, K in tqdm(zip(Ns.tolist(), Ks.tolist()), total=len(Ns)):
            time_ours = get_time(model_ours, N, K, vm, n_sample=100)
            logger.info(f"{K},{','.join(map(str, time_ours))}")

    for n in [1000, 2000, 3000, 4000]:
        logger = util.setup_logger(os.path.join(log_dir, f"complexity_seq_N_{n}.log"))
        logger.info("K,time")
        Ks = (np.arange(1, 21) * (n // 20)).astype(int)
        Ns = (np.ones_like(Ks) * n).astype(int)
        for N, K in tqdm(zip(Ns.tolist(), Ks.tolist()), total=len(Ns)):
            time_ours = get_time(model_seq, N, K, poutine, n_sample=100)
            logger.info(f"{K},{','.join(map(str, time_ours))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    if args.cuda >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = "cpu" if (not torch.cuda.is_available() or args.cuda == -1) else "cuda"
    torch.set_default_device(device)

    print(f">> Run scalability experiment on device {args.cuda}.")
    run(args)