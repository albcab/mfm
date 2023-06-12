import argparse

import jax
import jax.numpy as jnp
import jax.random as jrnd
import optax
import distrax as dx
import haiku as hk

from distributions import BioOxygen
from execute import run
from exe import taylor_run
from flows import Flow, RealSufficient, PositiveSufficient, make_cond, shift_scale

from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


N_PARAM = 2


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plots(samples, param, flow, flow_inv):#, sam, sam2):
    np.random.seed(0)
    c, n = samples["x1"].shape
    u1 = np.random.normal(0., 1., size=c * n)
    u2 = np.random.normal(0., 1., size=c * n)
    x1 = samples["x1"].reshape(-1)
    x2 = samples["x2"].reshape(-1)
    phi_samples, phi_weights = jax.vmap(lambda u1, u2: flow(jnp.array([u1, u2]), param))(u1, u2)
    pi_samples, pi_weights = jax.vmap(lambda x1, x2: flow_inv(jnp.array([x1, x2]), param))(x1, x2)
    w = jnp.exp(phi_weights)
    print(jnp.min(w), jnp.max(w))
    w = jnp.exp(pi_weights)
    print(jnp.min(w), jnp.max(w))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    ax[1].set_title(r"$\hat{\phi}(\theta)$")
    ax[1].set_xlabel(r"$\theta_1$")
    ax[1].set_ylabel(r"$\theta_2$")
    # sns.kdeplot(x=phi_samples[:, 0], y=phi_samples[:, 1], ax=ax[2], fill=True)
    sns.histplot(x=phi_samples[:, 0], y=phi_samples[:, 1], weights=jnp.exp(phi_weights), ax=ax[1], bins=50)
    ax[0].set_title(r"$\pi(\theta)$")
    ax[0].set_xlabel(r"$\theta_1$")
    ax[0].set_ylabel(r"$\theta_2$")
    sns.histplot(x=x1, y=x2, ax=ax[0], bins=50)
    # ax[0].sharex(ax[1])
    # ax[0].sharey(ax[1])
    # plt.savefig("biooxygen2.png", bbox_inches='tight')
    # plt.close()

    return None

class Message(Flow):
    def __init__(self, obs, times, num_bins=False, hidden_dims=0, non_linearity=None):
        super().__init__(num_bins, hidden_dims, non_linearity)
        self.obs = obs
        self.times = times

    def flows(self):
        # t0_transform = lambda x: self.obs / (1 - jnp.exp(-x * self.times))
        t0_transform = lambda x: jnp.array([jnp.mean(self.obs), jnp.mean(self.times)])
        t1_transform = lambda x: jnp.array([x[0], jnp.mean(self.obs), jnp.mean(self.times)])
        # t1_transform = lambda x: -jnp.log(1 - self.obs / x) / self.times

        flows = []
        t0_1 = make_cond(1, RealSufficient(axis=1, transform=t0_transform), self)
        flows.append(dx.SplitCoupling(1, 1, t0_1, self.coupling_fn, swap=True))

        t1_1 = make_cond(1, RealSufficient(axis=1, transform=t1_transform), self)
        flows.append(dx.SplitCoupling(1, 1, t1_1, self.coupling_fn, swap=False))

        # t0_2 = make_cond(1, RealSufficient(axis=None, transform=t0_transform), self)
        # flows.append(dx.SplitCoupling(1, 1, t0_2, self.coupling_fn, swap=True))

        # t1_2 = make_cond(1, RealSufficient(axis=None, transform=t1_transform), self)
        # flows.append(dx.SplitCoupling(1, 1, t1_2, self.coupling_fn, swap=False))

        # t0_3 = make_cond(1, RealSufficient(axis=None, transform=t0_transform), self)
        # flows.append(dx.SplitCoupling(1, 1, t0_3, self.coupling_fn, swap=True))

        # t1_3 = make_cond(1, RealSufficient(axis=None, transform=t1_transform), self)
        # flows.append(dx.SplitCoupling(1, 1, t1_3, self.coupling_fn, swap=False))

        return dx.Chain(flows)

def main(args):

    print("Generating synthetic data...")
    N = 20
    theta0 = 1.
    theta1 = .1
    var = 2 * 10 ** (-4)
    times = jnp.arange(1, 5, 4/N)
    std_norms = jrnd.normal(jrnd.PRNGKey(args.seed), (N,))
    obs = theta0 * (1. - jnp.exp(-theta1 * times)) + jnp.sqrt(var) * std_norms

    print("Setting up Biochemical oxygen demand density...")
    dist = BioOxygen(times, obs, var)

    [n_warm, n_iter] = args.sampling_param
    schedule = optax.exponential_decay(init_value=1e-2,
        transition_steps=n_warm-10, decay_rate=.1, transition_begin=10)
    optim = optax.adam(schedule)

    particles, *plot_params = run(dist, args, optim, N_PARAM, batch_fn=jax.vmap)
    plots(*plot_params)
    plots(particles, *(plot_params[1:]))

    flow = Message(obs, times, num_bins=20)
    plot_params = taylor_run(dist, args, flow, optim, N_PARAM, jax.vmap)
    plots(*plot_params)
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
    parser.add_argument('-s1', '--init_step_size', type=float, default=.0000001)
    parser.add_argument('-s2', '--p_init_step_size', type=float, default=.0000001)
    args = parser.parse_args()
    main(args)