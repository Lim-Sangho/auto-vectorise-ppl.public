import os
import sys
import argparse

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


def run(args):
    benchmark = {"hmm-ord1": hmm_ord1,
                 "hmm-ord2": hmm_ord2,
                 "hmm-neural": hmm_neural,
                 "dmm": dmm,
                 "arm": arm,
                 "tcm": tcm,
                 "nhmm-stock": nhmm_stock,
                 "nhmm-train": nhmm_train,
                }[args.benchmark]
    
    methods = ["plate", "vmarkov", "discHMM", "ours"] if args.method == "all" else [args.method]

    model_baseline = getattr(benchmark, "model_seq", None)
    svi_baseline = getattr(benchmark, "svi_seq", None)
    
    if not args.dry:
        log_dir = os.path.join(path, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        loggers = {}
        for method in methods:
            loggers[method] = util.setup_logger(os.path.join(log_dir, f"{args.benchmark}-{method}-consistency.log"))
            loggers[method].info("seed,error(loss),error(grads)")

    util._TIMEIT = False

    for seed in (seeds := tqdm(range(args.num_seed), desc="seeds", leave=True)):
        pyro.clear_param_store()
        vec.clear_allocators()
        pyro.set_rng_seed(seed)
        data = benchmark.data(args)
        optim = benchmark.optim(args)
        guide_baseline = benchmark.guide(model_baseline, *data)
        (loss_baseline, grads_baseline), _ = svi_baseline(model_baseline, guide_baseline, optim, *data)

        for method in methods:
            pyro.clear_param_store()
            vec.clear_allocators()
            pyro.set_rng_seed(seed)
            model = getattr(benchmark, f"model_{method}", None)
            svi = getattr(benchmark, f"svi_{method}", None)
            if model is None or svi is None:
                if not args.dry:
                    loggers[method].info(f">> Method {method} is not implemented for benchmark {args.benchmark}.")
                continue

            data = benchmark.data(args)
            optim = benchmark.optim(args)
            guide = benchmark.guide(model, *data)
            (loss, grads), _ = svi(model, guide, optim, *data)

            error_loss = util.l2_ratio(loss, loss_baseline)
            error_grads = util.l2_ratio_dict(grads, grads_baseline)
            seeds.set_description("Seed: %d, Method: %s, Error (loss): %f, Error (grads): %f" % (seed, method, error_loss, error_grads))

            if not args.dry:
                loggers[method].info(f"{seed},{error_loss},{error_grads}")


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
                        choices=["plate", "vmarkov", "discHMM", "ours", "all"],
                        help="choose from plate, vmarkov, discHMM, ours, all")
    parser.add_argument("--num_seed", type=int, default=100)
    parser.add_argument("--num_batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--dry", action="store_true", default=False)
    args = parser.parse_args()

    if args.cuda >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = "cpu" if (not torch.cuda.is_available() or args.cuda == -1) else "cuda"
    torch.set_default_device(device)

    print(f">> Run {args.method} for {args.benchmark} on device {args.cuda} with {args.num_seed} seeds.")
    run(args)