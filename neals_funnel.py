import argparse

import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.scipy.stats import norm

from distributions import NealsFunnel
from execute import run
from exe import taylor_run
from exe_fm import run as fm_run
from blackjax.optimizers.cocob import cocob
from mcmc_utils import stein_disc, max_mean_disc

import optax

from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import wandb

def plots(param, state, others, flow, flow_inv, dist, real):
    x1 = real["x1"]
    n, d = real["x2"].shape
    x2 = real["x2"]
    u = jax.random.normal(jax.random.PRNGKey(0), shape=(n, d+1))
    phi_samples, phi_weights = jax.vmap(lambda u: flow(u, param, state))(u)
    # phi_samples, phi_weights = flow(u, param, state)
    # pi_samples, pi_weights = jax.vmap(lambda x1, x2: flow_inv(jnp.array([x1, x2]), param))(x1, x2)
    # w = jnp.exp(phi_weights)
    # print(jnp.min(w), jnp.max(w))
    # w = jnp.exp(pi_weights)
    # print(jnp.min(w), jnp.max(w))
    
    # print("Logpdf of NF+MCMC samples=", jax.vmap(dist.logprob)(x1, x2).sum())
    logpdf = jax.vmap(dist.logprob)(phi_samples[:, 0], phi_samples[:, 1:]).sum()
    print("Logpdf of flow samples=", logpdf)
    samples_phi = {'x1': phi_samples[:, 0], 'x2': phi_samples[:, 1:]}
    stein = stein_disc(samples_phi, dist.logprob_fn)
    print("Stein U, V disc of flow samples=", stein[0], stein[1])
    mmd = max_mean_disc(real, samples_phi)
    print("Max mean disc of flow samples=", mmd)
    # samples = {'x1': x1, 'x2': x2}
    # stein = stein_disc(samples, dist.logprob_fn)
    # print("Stein U, V disc of NF+MCMC samples=", stein[0], stein[1])
    # mmd = max_mean_disc(real, samples)
    # print("Max mean disc of NF+MCMC samples=", mmd)
    print()
    data = [logpdf, stein[0], stein[1], mmd] + list(others)
    
    for i in range(d):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
        ax[1].set_title(r"$\hat{\phi}$")
        ax[1].set_xlabel(r"$\sigma$")
        ax[1].set_ylabel(r"$\theta$")
        sns.histplot(x=phi_samples[:, 0], y=phi_samples[:, i+1], ax=ax[1], bins=50)
        ax[0].set_title(r"$\pi$")
        ax[0].set_xlabel(r"$\sigma$")
        ax[0].set_ylabel(r"$\theta$")
        sns.histplot(x=x1, y=x2[:, i], ax=ax[0], bins=50)
        data.append(wandb.Image(fig))
        plt.close()
        if i > 8:
            break #only the first 10 dimensions

    return data

def main(args):

    N_PARAM = args.target_dimension
    wandb.init(project="neals-funnel", config=args, group="dim=" + str(N_PARAM), 
        job_type="step_size=" + str(args.step_size) + ",n_steps=" + str(args.sampler_iter))

    print("Setting up Neal's funnel density...")
    dist = NealsFunnel(N_PARAM)
    def target_gn(key):
        k0, k1 = jrnd.split(key)
        x0 = jrnd.normal(k0)
        x1 = jnp.exp(.5 * x0) * jrnd.normal(k1, (N_PARAM-1,))
        return jnp.hstack([x0, x1])

    [n_warm, n_iter] = args.sampling_iter
    clipper = optax.clip(args.gradient_clip)
    if args.cocob:
        optim = optax.chain(clipper, cocob())
    else:
        schedule = optax.exponential_decay(init_value=args.learning_rate,
            transition_steps=n_warm * args.optim_iter - 10 * args.optim_iter, decay_rate=args.decay_rate, transition_begin=10 * args.optim_iter)
        scale_lr = optax.scale_by_schedule(schedule)
        optim = optax.chain(clipper, optax.scale_by_adam(), scale_lr, optax.scale(-1))

    rng_key = jrnd.PRNGKey(args.seed)
    keys = jrnd.split(rng_key, n_iter * args.num_chain)
    real = jax.vmap(target_gn)(keys)
    print("Logpdf of real samples=", jax.vmap(dist.logprob)(real[:, 0], real[:, 1:]).sum())
    samples = {'x1': real[:, 0], 'x2': real[:, 1:]}
    stein = stein_disc(samples, dist.logprob_fn)
    print("Stein U, V disc of real samples=", stein[0], stein[1])
    mmd = max_mean_disc(samples, samples)
    print("Max mean disc of NF+MCMC samples=", mmd)
    print()
    # fig, ax = plt.subplots()
    # sns.histplot(x=real[:, 0], y=real[:, 1], ax=ax, bins=50)
    # plt.savefig("NealsFunnel.png", bbox_inches='tight')
    # plt.close()

    *algos, flows = fm_run(dist, args, N_PARAM, optim, target_gn=target_gn)
    # print("Optimize over integral w/ adjoint gradients:")
    # print("MSC Flow Matching w/ CIS:")
    my_data = []
    print("Flow matching with actual samples:")
    data = plots(*algos[0], *flows, dist, samples)
    my_data.append(["FM (real samples)"] + data)
    # plt.savefig("NealsFunnel real samples Flow matching.png", bbox_inches='tight')
    # plt.savefig("NealsFunnel 4.png", bbox_inches='tight')
    # plt.savefig("NealsFunnel 2 adjoint gradients.png", bbox_inches='tight')
    # plt.close()
    # print("MSC Flow Matching + continuous VI:")
    # print("MSC Flow Matching w/ MALA (refresh):")
    print("MSC Flow matching MALA Mod:")
    data = plots(*algos[1], *flows, dist, samples)
    my_data.append(["Algo 3 mod"] + data)
    # plt.savefig("NealsFunnel 3 refresh (MALA).png", bbox_inches='tight')
    # plt.savefig("NealsFunnel 3 + cont VI.png", bbox_inches='tight')
    # plt.savefig("NealsFunnel 3 MALA mod.png", bbox_inches='tight')
    # plt.close()
    print("MSC Flow Matching w/ refresh:")
    data = plots(*algos[2], *flows, dist, samples)
    my_data.append(["Algo 3"] + data)
    # plt.savefig("NealsFunnel 3 refresh.png", bbox_inches='tight')
    # plt.close()
    print("Base CNF:")
    data = plots(*algos[3], *flows, dist, samples)
    my_data.append(["Base 6.2"] + data)
    # plt.savefig("NealsFunnel Base CNF.png", bbox_inches='tight')
    # plt.close()
    print("Base CNF mod:")
    data = plots(*algos[4], *flows, dist, samples)
    my_data.append(["Base 6.2 mod"] + data)
    # plt.savefig("NealsFunnel Base CNF mod.png", bbox_inches='tight')
    # plt.close()
    print("MSC Flow matching with incremental training data:")
    data = plots(*algos[5], *flows, dist, samples)
    my_data.append(["Algo 3.1"] + data)

    columns = ["Algorithm", "logpdf", "KSD U-stat", "KSD V-stat", "MMD", "time (sec)", "avg accept", "std accept"] + ["plot (x0,x" + str(i+1) + ")" for i in range(min(N_PARAM-1, 10))]
    wandb.log({"summary": wandb.Table(columns, my_data)})
    wandb.finish()

    # *algos, flows = run(dist, args, optim, N_PARAM, batch_fn=jax.vmap)
    # print("TESS:")
    # plots(*algos[0], *flows, dist, samples)
    # plt.savefig("NealsFunnel TESS.png", bbox_inches='tight')
    # plt.close()
    # print("MSC CIS:")
    # plots(*algos[1], *flows, dist, samples)
    # plt.savefig("NealsFunnel MSC CIS.png", bbox_inches='tight')
    # plt.close()
    # # print("SVGD + discrete VI:")
    # print("MSC MALA:")
    # plots(*algos[2], *flows, dist, samples)
    # plt.savefig("NealsFunnel MSC MALA.png", bbox_inches='tight')
    # # plt.savefig("NealsFunnel SVGD + discrete VI.png", bbox_inches='tight')
    # # plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('-oi', '--optim-iter', type=int, default=1)
    parser.add_argument('-nc', '--num_chain', type=int, default=16)
    parser.add_argument('-nl', '--non-linearity', type=str, default='relu')
    parser.add_argument('-f', '--flow', type=str, default='coupling')
    parser.add_argument('-d', '--distance', type=str, default='kld')
    parser.add_argument('-hl', '--hidden-layers', type=int, nargs='+', default=[16] * 2)
    parser.add_argument('-nf', '--n-flow', type=int, default=2)
    parser.add_argument('-nb', '--num-bins', type=int, default=None)
    parser.add_argument(
        "-s", "--sampling-iter", type=int, nargs=2,
        help="Sampling parameters [n_warm, n_iter]",
        default=[400, 100]
    )
    parser.add_argument('-np', '--preconditon_iter', type=int, default=400)
    parser.add_argument('-ss', '--step-size', type=float, default=0.1)
    parser.add_argument('-si', '--sampler-iter', type=int, default=1)
    parser.add_argument('-cf', '--cont-flow', type=str, default='resnet')
    parser.add_argument('--cocob', dest='cocob', action='store_true')
    parser.add_argument('--no-cocob', dest='cocob', action='store_false')
    parser.set_defaults(cocob=True)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('-dr', '--decay-rate', type=float, default=0.1)
    parser.add_argument('-gc', '--gradient-clip', type=float, default=1.0)
    parser.add_argument('-td', '--target-dimension', type=int, default=2)
    parser.add_argument('-l', '--lamda', type=float, default=0.1)
    parser.add_argument('--l2', dest='l2', action='store_true')
    parser.add_argument('--l1', dest='l2', action='store_false')
    parser.set_defaults(l2=True)
    args = parser.parse_args()
    main(args)