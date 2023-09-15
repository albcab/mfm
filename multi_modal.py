import argparse

import jax.numpy as jnp

from distributions import GaussianMixture
from exe_flow_matching import run

from jax.config import config
# config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", True)

import wandb


def main(args):

    N_PARAM = args.dim
    wandb.init(project="gaussian-mixture", config=args, group="dim=" + str(N_PARAM), 
        job_type="mcmc_per_flow_steps=" + str(args.mcmc_per_flow_steps) + ",learning_iter=" + str(args.learning_iter))

    print("Setting up Gaussian mixture density...")
    modes = [5. * jnp.ones(N_PARAM), -5. * jnp.zeros(N_PARAM)]
    covs = [2. * jnp.eye(N_PARAM), 2. * jnp.eye(N_PARAM)]
    weights = jnp.array([0.5, 0.5])
    dist = GaussianMixture(modes, covs, weights)

    print("Running algorithm...")
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

    parser.add_argument("--mcmc_per_flow_steps", type=float, default=10)
    parser.add_argument('--num_chain', type=int, default=16)
    parser.add_argument("--learning_iter", type=int, default=400)
    parser.add_argument("--eval_iter", type=int, default=100)

    #defaults from PIS
    parser.add_argument('--non_linearity', type=str, default='relu')
    parser.add_argument('--hidden_x', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--hidden_t', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--hidden_xt', type=int, nargs='+', default=[64, 64])

    parser.add_argument('--step_size', type=float, default=0.1)

    # parser.add_argument('--cocob', dest='cocob', action='store_true')
    # parser.add_argument('--no-cocob', dest='cocob', action='store_false')
    # parser.set_defaults(cocob=True)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_epsilon', type=float, default=0.999)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=10)

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