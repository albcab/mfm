import argparse

import pandas as pd
import numpy as np

import jax
import jax.numpy as jnp

from distributions import GaussianMixture, PhiFour, LogGaussianCoxPines
from exe_flow_matching import run
from exe_others import run as run_others

from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

import wandb


def main(args):

    if args.example == "gaussian-mixture":
        print("Setting up Gaussian mixture density...")
        args.dim = 2
        args.num_modes = 16
        args.fourier_dim = 64
        args.num_chain = 16
        args.eval_iter = 400
        args.lim = [-16, 16]
        args.levels = 20
        args.num_anneal_temp = 200
        args.step_size = 0.6
        args.hidden_x = args.hidden_t = args.hidden_xt = [64, 64]

        key_mode, key_cov, key_weight = jax.random.split(jax.random.PRNGKey(0), 3)
        modes = jax.random.uniform(key_mode, (args.num_modes, args.dim), minval=args.lim[0] * .8, maxval=args.lim[1] * .8)
        print("Modes=", modes)
        covs = jnp.exp(.5 * jax.random.normal(key_cov, (args.num_modes, args.dim)))
        print("Covs=", covs)
        # covs = jnp.array([cov * jnp.eye(args.dim) for cov in covs])
        weights = jax.random.dirichlet(key_weight, 4. * jnp.ones(args.num_modes))
        print("Weights=", weights)
        dist = GaussianMixture(modes, covs, weights)
        # args.anneal_temp = [i / args.num_anneal_temp for i in range(1, args.num_anneal_temp + 1)]

    elif args.example == "phi-four":
        print("Setting up Phi four example density...")
        dist = PhiFour(args.dim)
        args.lim = [-1, 1]
        args.step_size = 0.001
        dist.sample_model = None
        # args.ref_dist = "phifour"
        # args.cond_flow = True

    elif args.example == "4-mode":
        print("Setting up 4-mode Gaussian mixture density...")
        args.dim = 2
        args.fourier_dim = 64
        args.num_chain = 16
        args.eval_iter = 400
        args.lim = [-16, 16]
        args.levels = 20
        args.num_anneal_temp = 200
        args.step_size = 0.6
        args.hidden_x = args.hidden_t = args.hidden_xt = [64, 64]
        modes = 8. * jnp.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        print("Modes=", modes)
        covs = jnp.ones((4, args.dim))
        print("Covs=", covs)
        weights = jnp.ones(4) / 4
        print("Weights=", weights)
        dist = GaussianMixture(modes, covs, weights)

    elif args.example == "pines":
        print("Setting up Log Gaussian Cox density...")
        args.dim = 900
        args.lim = None
        args.num_chain = 32
        args.hidden_x = args.hidden_xt = [2**10] * 2
        args.hidden_t = [args.fourier_dim, args.fourier_dim]
        dist = LogGaussianCoxPines(args.dim)
        dist.sample_model = None

    else:
        raise Exception("Example not found.") 


    N_PARAM = args.dim
    if args.do_flowmc:
        job_type = "flowMC," + "mcmc_per_flow_steps=" + str(args.mcmc_per_flow_steps)
    elif args.do_pocomc:
        job_type = "pocomc"
    elif args.do_dds:
        job_type = "denoising diffusion sampler"
    elif args.do_smc:
        job_type = "Adaptive tempered SMC"
    else:
        job_type = "mcmc_per_flow_steps=" + str(args.mcmc_per_flow_steps) + ",learning_iter=" + str(args.learning_iter) + (",hutchs" if args.hutchs else "")

    seeds = [args.seed] if args.seed else [i**10 for i in range(10)]
    res = []
    res_ = []
    for seed in seeds:
        args.seed = seed
        wandb.init(project=args.example, config=args, group="dim=" + str(N_PARAM), 
            job_type=job_type)

        print(f"Running algorithm seed {seed}...")
        if args.do_flowmc or args.do_pocomc or args.do_dds or args.do_smc:
            _res, _res_ = run_others(dist, args, dist.sample_model)
        else:
            _res, _res_ = run(dist, args, dist.sample_model)
        res.append(_res)
        res_.append(_res_)

    res = jnp.array(res)
    res_means = res.mean(axis=0)
    res_stds = res.std(axis=0)
    res_ = jnp.array(res_)
    res_means_ = res_.mean(axis=0)
    res_stds_ = res_.std(axis=0)
    print(job_type)
    print("-" * 100)
    print("logprob\t & stein-u\t & stein-v\t & mmd  \t & time \t")
    print(*[f"{m:.2e} \pm {s * 1.96:.2e}" for m, s in zip(res_means, res_stds)], sep="$ & $")
    print(*[f"{m:.2e} \pm {s * 1.96:.2e}" for m, s in zip(res_means_, res_stds_)], sep="$ & $")
    print("-" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    
    parser.add_argument('--dim', type=int, default=64) #2
    parser.add_argument('--num_modes', type=int, default=16)
    parser.add_argument("--example", type=str, default="pines") #gaussian-mixture

    parser.add_argument("--sigma", type=float, default=1e-4)
    parser.add_argument("--fourier_dim", type=int, default=128) #64
    parser.add_argument("--fourier_std", type=float, default=1.0)
    parser.add_argument('--hutchs', dest='hutchs', action='store_true')
    parser.set_defaults(hutchs=False)

    parser.add_argument("--ref_dist", type=str, default='stdgauss')
    parser.add_argument('--cond_flow', dest='cond_flow', action='store_true')
    parser.set_defaults(cond_flow=True) #True
    parser.add_argument('--ot_cond_flow', dest='ot_cond_flow', action='store_true')
    parser.set_defaults(ot_cond_flow=False)

    parser.add_argument("--num_importance_samples", type=int, default=0)
    parser.add_argument("--mcmc_per_flow_steps", type=float, default=10)
    parser.add_argument('--num_chain', type=int, default=128) #16
    parser.add_argument("--learning_iter", type=int, default=400)
    parser.add_argument("--eval_iter", type=int, default=10) #400

    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--anneal_iter", type=int, default=200)
    parser.add_argument('--num_anneal_temp', type=int, default=200) #10

    parser.add_argument('--non_linearity', type=str, default='relu')
    parser.add_argument('--hidden_x', type=int, nargs='+', default=[128, 128]) #[64, 64]
    parser.add_argument('--hidden_t', type=int, nargs='+', default=[128, 128]) #[64, 64]
    parser.add_argument('--hidden_xt', type=int, nargs='+', default=[128, 128]) #[64, 64]

    parser.add_argument('--step_size', type=float, default=0.4) #.4

    parser.add_argument('--do_flowmc', dest='do_flowmc', action='store_true')
    parser.set_defaults(do_flowmc=False)

    parser.add_argument('--do_pocomc', dest='do_pocomc', action='store_true')
    parser.set_defaults(do_pocomc=False)

    parser.add_argument('--do_dds', dest='do_dds', action='store_true')
    parser.set_defaults(do_dds=False)

    parser.add_argument('--do_smc', dest='do_smc', action='store_true')
    parser.set_defaults(do_smc=False)

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_epsilon', type=float, default=0.999)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=0)

    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--mxstep', type=float, default=1000)
    # parser.add_argument('--flow_rtol', type=float, default=1e-5)
    # parser.add_argument('--flow_atol', type=float, default=1e-5)
    # parser.add_argument('--flow_mxstep', type=float, default=100)

    parser.add_argument('--lim', type=float, nargs=2, default=[-16, 16])
    parser.add_argument('--grid_width', type=int, default=400)
    parser.add_argument('--levels', type=int, default=50)

    parser.add_argument('--check', dest='check', action='store_true')
    parser.set_defaults(check=False)
    args = parser.parse_args()
    main(args)
