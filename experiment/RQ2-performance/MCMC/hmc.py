# Reference: https://github.com/Wenlin-Chen/DiGS/blob/master/MoG-40/HMC.ipynb

import os
import sys
import copy
import time

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.dont_write_bytecode = True

import pyro
import torch
import vectorized_loop as vec

from tqdm.auto import tqdm
from functools import partial
from util import set_device, setup_logger


def target_density_and_grad_fn_full(x, inv_temperature, target_log_prob_fn):
    x = x.clone().detach().requires_grad_(True)
    log_prob = target_log_prob_fn(x) * inv_temperature
    log_prob_sum = log_prob.sum()
    log_prob_sum.backward()
    grad = x.grad.clone().detach()
    return log_prob.detach(), grad


class HamiltonianMonteCarlo(object):

    def __init__(self,
                 x,
                 log_prob_fn: callable,
                 step_size: float,
                 num_leapfrog_steps_per_hmc_step: int,
                 inv_temperature: float = 1.0):
        super(HamiltonianMonteCarlo, self).__init__()

        self.x = x
        self.step_size = step_size
        self.inv_temperature = inv_temperature
        self.num_leapfrog_steps_per_hmc_step = num_leapfrog_steps_per_hmc_step
        self.set_target_density_and_grad_fn(log_prob_fn)
        self.current_log_prob, self.current_grad = self.target_density_and_grad_fn(x, self.inv_temperature)

    def set_target_density_and_grad_fn(self, log_prob_fn: callable):
        self.target_density_and_grad_fn = partial(target_density_and_grad_fn_full, target_log_prob_fn=log_prob_fn)

    def leapfrog_integration(self, p):
        """
        Leapfrog integration for simulating Hamiltonian dynamics.
        """
        x = self.x.detach().clone()
        p = p.detach().clone()

        # Half step for momentum
        p += 0.5 * self.step_size * self.current_grad

        # Full steps for position
        for _ in range(self.num_leapfrog_steps_per_hmc_step - 1):
            x += self.step_size * p
            _, grad = self.target_density_and_grad_fn(x, self.inv_temperature)
            p += self.step_size * grad  # this combines two half steps for momentum

        # Final update of position and half step for momentum
        x += self.step_size * p
        new_log_prob, new_grad = self.target_density_and_grad_fn(x, self.inv_temperature)
        p += 0.5 * self.step_size * new_grad

        return x, p, new_log_prob, new_grad


    def sample(self):
        """
        Hamiltonian Monte Carlo step.
        """

        # Sample a new momentum
        p = torch.randn_like(self.x)

        # Simulate Hamiltonian dynamics
        new_x, new_p, new_log_prob, new_grad = self.leapfrog_integration(p)

        # Hamiltonian (log probability + kinetic energy)
        current_hamiltonian = self.current_log_prob - 0.5 * p.pow(2).sum(-1)
        new_hamiltonian = new_log_prob - 0.5 * new_p.pow(2).sum(-1)
        
        log_accept_rate = - current_hamiltonian + new_hamiltonian
        is_accept = torch.rand_like(log_accept_rate).log() < log_accept_rate
        is_accept = is_accept.unsqueeze(-1)

        self.x = torch.where(is_accept, new_x.detach(), self.x)
        self.current_grad = torch.where(is_accept, new_grad.detach(), self.current_grad)
        self.current_log_prob = torch.where(is_accept.squeeze(-1), new_log_prob.detach(), self.current_log_prob)

        acc_rate = torch.minimum(torch.ones_like(log_accept_rate), log_accept_rate.exp()).mean()
        
        return copy.deepcopy(self.x.detach()), acc_rate.item()


def sample_to_params(sample, vars, shapes):
    params = {}
    idx = 0
    for v in vars:
        size = int(torch.tensor(shapes[v]).prod().item())
        params[v] = sample[idx:idx+size].reshape(shapes[v])
        idx += size
    return params


def run_HMC(log_prob_fn, initial_params, gibbs_step, gibbs_vars, hmc_vars, args, chain_id):
    print("\r", end="", flush=True)

    set_device(args.cuda)
    pyro.clear_param_store()
    vec.clear_allocators()
    pyro.set_rng_seed(args.seed * (2 ** 10) + chain_id)

    if not args.dry:
        log_dir = os.path.join(path, "logs")
        logger = setup_logger(os.path.join(log_dir, f"samples-{args.benchmark}-{args.method}-seed{args.seed}.log"))

    gibbs_params = {v: initial_params[v] for v in gibbs_vars}
    hmc_shapes = {v: initial_params[v].shape for v in hmc_vars}
    hmc_sample = torch.cat([initial_params[v].flatten() for v in hmc_vars], dim=0)

    def log_prob_fn_hmc(gibbs_params):
        def wrapper(hmc_sample):
            return log_prob_fn({**sample_to_params(hmc_sample, hmc_vars, hmc_shapes), **gibbs_params})
        return wrapper

    hmc = HamiltonianMonteCarlo(hmc_sample, log_prob_fn_hmc(gibbs_params), args.step_size, args.num_leapfrog_steps)
    bar = tqdm(range(args.num_samples), position=chain_id)
    bar.set_description(f"Chain {chain_id}")

    start_time = time.time()
    for _ in bar:
        acc_rate = [] 
        for _ in range(args.num_hmc_steps):
            gibbs_params, _ = gibbs_step(log_prob_fn, sample_to_params(hmc_sample, hmc_vars, hmc_shapes), gibbs_params)
            hmc.set_target_density_and_grad_fn(log_prob_fn_hmc(gibbs_params))
            hmc_sample, _acc_rate = hmc.sample()
            acc_rate.append(_acc_rate)

        elapsed_time = time.time() - start_time
        bar.set_postfix({"acc_rate": f"{sum(acc_rate) / len(acc_rate):.3f}"})

        memory = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        if not args.dry:
            sample = hmc_sample.tolist()
            for v in gibbs_vars:
                sample += gibbs_params[v].flatten().tolist()
            row = [elapsed_time, chain_id, memory] + sample
            logger.info(",".join(map(str, row)))
