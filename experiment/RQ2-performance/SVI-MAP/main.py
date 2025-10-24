import os
import sys
import argparse
import multiprocessing as mp

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, "../"))
sys.path.append(os.path.join(path, "../../"))
sys.dont_write_bytecode = True

import pyro
import torch
import vectorized_loop as vec
from tqdm.auto import tqdm
from benchmark import hmm_ord1, hmm_ord2, hmm_neural, dmm, arm, tcm, nhmm_stock, nhmm_train

import util


def run(args, method, seed):
    print("\r", end="")
    torch.cuda.reset_peak_memory_stats()
    pyro.clear_param_store()
    vec.clear_allocators()
    pyro.set_rng_seed(seed)

    if args.cuda >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = "cpu" if (not torch.cuda.is_available() or args.cuda == -1) else "cuda"
    torch.set_default_device(device)

    if not args.dry:
        log_dir = os.path.join(path, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logger = util.setup_logger(os.path.join(log_dir, f"{args.benchmark}-{method}-seed{seed}-performance.log"))
        logger.info("step,total,guide,model,reduce,backward,memory")

    benchmarks = {"hmm-ord1": hmm_ord1,
                  "hmm-ord2": hmm_ord2,
                  "hmm-neural": hmm_neural,
                  "dmm": dmm,
                  "arm": arm,
                  "tcm": tcm,
                  "nhmm-stock": nhmm_stock,
                  "nhmm-train": nhmm_train,
                 }

    benchmark = benchmarks[args.benchmark]
    model = getattr(benchmark, f"model_{method}", None)
    svi = getattr(benchmark, f"svi_{method}", None)
    if model is None or svi is None:
        msg = f">> Method {method} is not implemented for benchmark {args.benchmark}."
        print(msg) if args.dry else logger.info(msg)
        return
    
    optim = benchmark.optim(args)
    data = benchmark.data(args)
    guide = benchmark.guide(model, *data)

    for step in (steps := tqdm(range(args.num_step), desc="steps", leave=True)):
        (loss, grads), (t_total, t_guide, t_model, t_reduce, t_backward, memory) = svi(model, guide, optim, *data)
        loss = loss / data[0].numel()
        steps.set_description("Seed: %d, Step: %d, Benchmark: %s, Method: %s, Loss: %.4f" % (seed, step, args.benchmark, method, loss))
        if not args.dry:
            logger.info(f"{step},{t_total.time},{t_guide.time},{t_model.time},{t_reduce.time},{t_backward.time},{memory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark",
                        type=str,
                        default="hmm-ord1",
                        choices=["hmm-ord1", "hmm-ord2", "hmm-neural", "dmm", "arm", "tcm", "nhmm-stock", "nhmm-train"],
                        help="choose from hmm-ord1, hmm-ord2, hmm-neural, dmm, arm, tcm, nhmm-stock, nhmm-train")
    parser.add_argument("--method",
                        type=str,
                        default="ours",
                        choices=["seq", "plate", "vmarkov", "discHMM", "manual", "ours", "all"],
                        help="choose from seq, plate, vmarkov, discHMM, manual, ours, all")
    parser.add_argument("--num_seed", type=int, default=5)
    parser.add_argument("--num_step", type=int, default=1000)
    parser.add_argument("--num_batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--dry", action="store_true", default=False)
    args = parser.parse_args()

    print(f">> Run {args.method} for {args.benchmark} on device {args.cuda} with {args.num_seed} seeds and {args.num_step} steps.")
    mp.set_start_method("spawn", force=True)
    util._TIMEIT = True

    methods = ["seq", "plate", "vmarkov", "discHMM", "manual", "ours"] if args.method == "all" else [args.method]
    for method in methods:
        for seed in range(args.num_seed):
            p = mp.Process(target=run, args=(args, method, seed))
            p.start()
            p.join()