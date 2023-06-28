import argparse

import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.flatten_util import ravel_pytree
import optax
import distrax as dx
import haiku as hk

from distributions import BioOxygen, MultivarNormal
from execute import run
from exe import taylor_run
from exe_fm import run as fm_run
from flows import make_dense, affine_coupling
from blackjax.optimizers.cocob import cocob

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
    sns.histplot(x=phi_samples[:, 0], y=phi_samples[:, 1], ax=ax[1], bins=50)
    # sns.histplot(x=u1, y=u2, ax=ax[0], bins=50)
    ax[0].set_title(r"$\pi(\theta)$")
    ax[0].set_xlabel(r"$\theta_1$")
    ax[0].set_ylabel(r"$\theta_2$")
    sns.histplot(x=x1, y=x2, ax=ax[0], bins=50)
    # ax[0].sharex(ax[1])
    # ax[0].sharey(ax[1])
    # plt.savefig("biooxygen2.png", bbox_inches='tight')
    # plt.close()

    N = 20
    theta0 = 1.
    theta1 = .1
    times = jnp.arange(1, 5, 4/N)
    obs = theta0 * (1. - jnp.exp(-theta1 * times))
    samo = jax.vmap(lambda t0, t1: jnp.sum(jnp.abs(obs - t0 * (1. - jnp.exp(-t1 * times)))))(x1, x2)
    print(samo.mean(), samo.var())

    return None

def get_affine(obs, times):
    data = jnp.concatenate([obs, times, obs*times])
    # data = jnp.array([jnp.mean(obs), jnp.mean(obs**2), jnp.mean(obs * times)])
    # data = jnp.array([obs[0] / (2 - obs[1]/obs[0]), -jnp.log(1 - (2 - obs[1]/obs[0]))/times[0]])
    # norm = hk.LayerNorm(-1, True, True)
    # data = norm(data)
    dense = make_dense(N_PARAM, 0, True, None, None)
    param = dense(data)
    data_affine = affine_coupling(param)
    encoder = make_dense(N_PARAM-1, 0, False, None, None)
    self_affine = dx.SplitCoupling(N_PARAM-1, 1, encoder, affine_coupling, swap=False)
    return data_affine, self_affine

def _flow(u, obs, times):
    data_affine, self_affine = get_affine(obs, times)
    u_, ldj_ = data_affine.forward_and_log_det(u)
    x, ldj = self_affine.forward_and_log_det(u_)
    return x, jnp.sum(ldj_) + ldj
flow_ = hk.transform(_flow)
def flow(u, obs, times, param):
    u, unravel_fn = ravel_pytree(u)
    x, ldj = flow_.apply(param, None, u, obs, times)
    return unravel_fn(x), jnp.sum(ldj)
param_init = flow_.init

def _flow_inv(x, obs, times):
    data_affine, self_affine = get_affine(obs, times)
    x_, ldj_ = data_affine.inverse_and_log_det(x)
    u, ldj = self_affine.inverse_and_log_det(x_)
    return u, jnp.sum(ldj_) + ldj
flow_inv_ = hk.transform(_flow_inv)
def flow_inv(x, obs, times, param):
    x, unravel_fn = ravel_pytree(x)
    u, ldj = flow_inv_.apply(param, None, x, obs, times)
    return unravel_fn(u), jnp.sum(ldj)

def main(args):

    print("Generating synthetic data...")
    N = 20
    # N=2
    theta0 = 1.
    theta1 = .1
    var = 2 * 10 ** (-4)
    times = jnp.arange(1, 5, 4/N)
    # times = jnp.array([1, 2])
    rng_key = jrnd.PRNGKey(args.seed)
    key1, key2 = jrnd.split(rng_key)
    std_norms = jrnd.normal(key1, (N,))
    obs = theta0 * (1. - jnp.exp(-theta1 * times)) + jnp.sqrt(var) * std_norms
    print(theta0 * (1. - jnp.exp(-theta1 * times)))

    print("Setting up Biochemical oxygen demand density...")
    prior_mean = 0.
    prior_sd = 1.
    # dist = BioOxygen(times, obs, var, prior_mean, prior_sd)
    mean = jnp.array([-10, 10])
    cov = jnp.array([[1, .5], [.5, 1]])
    dist = MultivarNormal(mean, cov)
    def prior_gn(key):
        k0, k1 = jrnd.split(key)
        x0 = -10. + 2. * jrnd.normal(k0)
        x1 = 10. + 2. * jrnd.normal(k1)
        return jnp.array([x0, x1])
    # def prior_gn(key):
    #     k0, k1 = jrnd.split(key)
    #     theta0 = prior_mean + prior_sd * jrnd.normal(k0)
    #     theta1 = prior_mean + prior_sd * jrnd.normal(k1)
    #     return jnp.array([theta0, theta1])
    # def likelihood_gn(key, params):
    #     theta0, theta1 = params
    #     std_norms = jrnd.normal(key, (N,))
    #     obs = theta0 * (1. - jnp.exp(-theta1 * times)) + jnp.sqrt(var) * std_norms
    #     return obs, times

    [n_warm, n_iter] = args.sampling_param
    # schedule = optax.exponential_decay(init_value=1e-4,
    #     transition_steps=n_warm-10, decay_rate=.1, transition_begin=10)
    # optim = optax.adam(schedule)
    optim = cocob()

    plot_params = fm_run(dist, args, N_PARAM, optim, 0.1, prior_gn)
    plots(*plot_params)
    keys = jrnd.split(key2, 100*4*32)
    real = jax.vmap(lambda k: jrnd.multivariate_normal(k, mean, cov))(keys)
    fig, ax = plt.subplots()
    sns.histplot(x=real[:, 0], y=real[:, 1], ax=ax, bins=50)
    plt.show()

    # particles, *plot_params = run(dist, args, optim, N_PARAM, batch_fn=jax.vmap)
    # plots(*plot_params)
    # plots(particles, *(plot_params[1:]))

    # position = jnp.array([theta0, theta1])
    # init_param = param_init(key2, position, obs, times)
    # plot_params = taylor_run(dist, args, init_param, flow, flow_inv, optim, N, likelihood_gn, prior_gn)
    # plots(*plot_params)
    # plt.show()


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