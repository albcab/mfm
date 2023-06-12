import argparse

import pandas as pd

import jax
import optax

from distributions import PredatorPrey
from execute import run

from numpyro.util import enable_x64
enable_x64(use_x64=True)

from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


N_PARAM = 8

import arviz
import matplotlib.pyplot as plt

import diffrax

def plots(samples, time, prey, pred):
    arviz.plot_pair(
        samples,
        marginals=True,
        figsize=(12, 8)
    )   
    return None

def main(args):

    print("Loading predator-prey data...")
    data = pd.read_table("lynxhare.txt", sep=" ", names=['year', 'prey', 'pred', ''])
    # data = data.loc[data.year >= 1920]

    print("Setting up predator-prey model...")
    dist = PredatorPrey(
        data.year.values, data.pred.values, data.prey.values,
    )

    [n_warm, n_iter] = args.sampling_param
    schedule = optax.exponential_decay(init_value=1e-4,
        transition_steps=n_warm-10, decay_rate=.1, transition_begin=10)
    optim = optax.adam(schedule)

    particles, samples, *_ = run(dist, args, optim, N_PARAM, batch_fn=jax.vmap)
    plots(particles)
    plots(samples)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('-m', '--max-iter', type=int, default=1)
    parser.add_argument('-b', '--batch-shape', type=int, nargs=2, default=[4, 32])
    parser.add_argument('-nl', '--non-linearity', type=str, default='relu')
    parser.add_argument('-f', '--flow', type=str, default='coupling')
    parser.add_argument('-d', '--distance', type=str, default='kld')
    parser.add_argument('-nh', '--n-hidden', type=int, default=2)
    parser.add_argument('-nf', '--n-flow', type=int, default=2)
    parser.add_argument('-nb', '--num-bins', type=int, default=None)
    parser.add_argument(
        "-s", "--sampling-param", type=int, nargs=2,
        help="Sampling parameters [n_warm, n_iter]",
        default=[400, 100]
    )
    parser.add_argument('-np', '--preconditon_iter', type=int, default=400)
    parser.add_argument('-s1', '--init_step_size', type=float, default=None)
    parser.add_argument('-s2', '--p_init_step_size', type=float, default=None)
    args = parser.parse_args()
    main(args)