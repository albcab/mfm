import argparse

import jax.numpy as jnp

from distributions import GaussianMixture
from exe_flow_matching import run
from exe_others import run as run_others

from jax.config import config
# config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", True)

import wandb


def main(args):

    N_PARAM = args.dim
    if args.do_flowmc:
        job_type = "flowMC," + "mcmc_per_flow_steps=" + str(args.mcmc_per_flow_steps)
    elif args.do_pocomc:
        job_type = "pocomc"
    elif args.do_dds:
        job_type = "denoising diffusion sampler"
    else:
        job_type = "mcmc_per_flow_steps=" + str(args.mcmc_per_flow_steps) + ",num_importance_samples=" + str(args.num_importance_samples)
    wandb.init(project="gaussian-mixture", config=args, group="dim=" + str(N_PARAM), 
        job_type=job_type)

    print("Setting up Gaussian mixture density...")
    modes = [5., 0.]
    covs = [.5, .5]
    weights = jnp.array([.7, .3])
    dist = GaussianMixture(N_PARAM, modes, covs, weights)

    print("Running algorithm...")
    if args.do_flowmc or args.do_pocomc or args.do_dds:
        run_others(dist, args, dist.sample_model)
    else:
        run(dist, args, dist.sample_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument('--dim', type=int, default=2)

    parser.add_argument("--sigma", type=float, default=1e-4)
    parser.add_argument("--fourier_dim", type=int, default=64)
    parser.add_argument("--fourier_std", type=float, default=1.0)
    parser.add_argument('--hutchs', dest='hutchs', action='store_true')
    parser.set_defaults(hutchs=False)

    parser.add_argument("--ref_dist", type=str, default='stdgauss')
    parser.add_argument('--cond_flow', dest='cond_flow', action='store_true')
    parser.set_defaults(cond_flow=False)
    parser.add_argument('--ot_cond_flow', dest='ot_cond_flow', action='store_true')
    parser.set_defaults(ot_cond_flow=False)

    parser.add_argument("--num_importance_samples", type=int, default=0)
    parser.add_argument("--mcmc_per_flow_steps", type=float, default=10)
    parser.add_argument('--num_chain', type=int, default=16)
    parser.add_argument("--learning_iter", type=int, default=400)
    parser.add_argument("--eval_iter", type=int, default=100)

    parser.add_argument("--anneal_iter", type=int, default=200)
    parser.add_argument('--anneal_temp', type=int, nargs='+', default=[(i + 1) / 10 for i in range(10)])
    parser.add_argument("--anneal_dist", type=str, default="flat")

    #defaults from PIS
    parser.add_argument('--non_linearity', type=str, default='relu')
    parser.add_argument('--hidden_x', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--hidden_t', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--hidden_xt', type=int, nargs='+', default=[64, 64])

    parser.add_argument('--step_size', type=float, default=0.1)

    parser.add_argument('--do_flowmc', dest='do_flowmc', action='store_true')
    parser.set_defaults(do_flowmc=False)

    parser.add_argument('--do_pocomc', dest='do_pocomc', action='store_true')
    parser.set_defaults(do_pocomc=False)

    parser.add_argument('--do_dds', dest='do_dds', action='store_true')
    parser.set_defaults(do_dds=False)

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
    parser.add_argument('--flow_rtol', type=float, default=1e-5)
    parser.add_argument('--flow_atol', type=float, default=1e-5)
    parser.add_argument('--flow_mxstep', type=float, default=1000)

    parser.add_argument('--check', dest='check', action='store_true')
    parser.set_defaults(check=False)
    args = parser.parse_args()
    main(args)