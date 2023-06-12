import warnings

import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.flatten_util import ravel_pytree

import pandas as pd
from scipy.stats import qmc

from numpyro.diagnostics import print_summary#, effective_sample_size

from flows import Coupling, ShiftScale
from distances import kullback_liebler, renyi_alpha
from mcmc_utils import inference_loop0, stein_disc, autocorrelation

from blackjax.mcmc.tess import tess
from blackjax.adaptation.atess import atess, optimize
from blackjax.vi.svgd import coin_svgd, svgd, update_median_heuristic


def do_summary(samples, logprob_fn, sec):
    print_summary(samples)
    stein = stein_disc(samples, logprob_fn)
    print(f"Stein U-, V-statistics={stein[0]}, {stein[1]}")
    # stein = [0, 0]
    corr = []
    ess = []
    for name, value in samples.items():
        value = jax.device_get(value)
        n = value.shape
        auto_corr = autocorrelation(value, axis=1)
        factor = 1. - jnp.arange(1, n[1]) / n[1]
        if len(n) == 3:
            auto_corr = jax.vmap(lambda ac: 1./2 + 2 * jnp.sum(factor * ac[:, 1:], axis=1), 2)(auto_corr)
        else:
            auto_corr = 1./2 + 2 * jnp.sum(factor * auto_corr[:, 1:], axis=1)
        corr.append(auto_corr)
        # ind_ess = effective_sample_size(value)
        # ess.append(ind_ess)
    corr = jnp.vstack(corr).T
    ess = n[1] / (2 * corr)
    ess = jnp.median(ess, axis=0)
    print("Min. ESS=", jnp.min(ess) * n[0], jnp.min(ess))
    # ess = jnp.hstack(ess)
    # print("Min. ESS=", jnp.min(ess), jnp.min(ess)/n[0])
    # corr = jnp.max(corr, axis=0)
    # print("Mean and std max int corr=", jnp.mean(corr), jnp.std(corr))
    std_corr = jnp.std(jnp.max(corr, axis=1))
    corr = jnp.median(corr, axis=0)
    print("Mean and std max int corr=", jnp.max(corr), std_corr)
    print(f"{jnp.max(corr):.3f} & {std_corr:.3f} & " + 
        f"{jnp.min(ess) * n[0]:.0f} & {jnp.min(ess):.0f} & " + 
        f"{jnp.min(ess) * n[0] / sec:.3f} & {jnp.min(ess) / sec:.3f} & " + 
        f"{stein[0]:.3e} & {stein[1]:.3e}")
    return None


def run(dist, args, optim, N_PARAM, batch_fn=jax.vmap):
    [n_warm, n_iter] = args.sampling_param
    [batch_iter, batch_size] = args.batch_shape
    kflow, ksam, kinit = jrnd.split(jrnd.PRNGKey(args.seed), 3)
    dist.initialize_model(kinit, batch_iter * batch_size)

    init_param, flow, flow_inv, reverse, forward = initialize_flow(
        kflow, dist.logprob_fn, args.flow, args.distance, N_PARAM, 
        args.n_flow, args.n_hidden, args.non_linearity, args.num_bins
    )

    one_init_param = jax.tree_map(lambda p: p[0], dist.init_params)
    mc_samples = 1000
    precond_iter = args.preconditon_iter
    precond_param = run_precondition(kflow, init_param, one_init_param, 
        optim, reverse, mc_samples, precond_iter)
    
    print("TESS w/ precond.")
    samples, param = run_tess(ksam, dist.logprob_fn, dist.init_params,
        n_warm, n_iter, precond_param, optim, flow, forward, 
        batch_iter, batch_size, args.max_iter, batch_fn)

    print("SVGD")
    dist.initialize_model(kinit, 1 * n_iter)
    push_param = jax.tree_util.tree_map(lambda p: flow(p, param), dist.init_params)
    particles = run_svgd(dist.logprob_fn, push_param, n_warm, n_iter, 1, optim)

    return particles, samples, param, flow, flow_inv


non_lins = {
    'tanh': jax.nn.tanh,
    'elu': jax.nn.elu,
    'relu': jax.nn.relu,
    'swish': jax.nn.swish,
}

flows = {
    'coupling': lambda n, f, h, nl, nb: Coupling(n, f, h, nl, False, nb).get_utilities(),
    'ncoupling': lambda n, f, h, nl, nb: Coupling(n, f, h, nl, True, nb).get_utilities(),
    'shift_scale': lambda n, *_: ShiftScale(n).get_utilities(),
}

distances = {
    'kld': kullback_liebler,
    'ralpha=0.5': lambda fn, n, f, fi: renyi_alpha(fn, n, f, fi, .5),
    'ralpha=2': lambda fn, n, f, fi: renyi_alpha(fn, n, f, fi, 2.),
    'ralpha=0': lambda fn, n, f, fi: renyi_alpha(fn, n, f, fi, 0.),
}


def test_flow(rng_key, dist, flow, n_chain, n_iter, param, position):
    batch_flow = jax.vmap(jax.vmap(lambda u: flow(u, param)[0]))
    p, unraveler_fn = ravel_pytree(position)
    U = jax.vmap(jax.vmap(unraveler_fn))(
        jax.random.normal(rng_key, shape=(n_chain, n_iter) + p.shape))
    samples = batch_flow(U)
    do_summary(samples, dist.logprob_fn, 1.)
    return None


def initialize_flow(
    rng_key, logprob_fn, flow, distance, 
    d, n_flow, n_hidden, non_linearity, num_bins,
):
    if flow in ['iaf', 'riaf'] and any([nh != d for nh in n_hidden]):
        warnings.warn('IAF flows always have dimension of hidden units same as params.')

    if flow in ['iaf', 'riaf'] and num_bins:
        warnings.warn('IAF cannot do rational quadratic splines.')
    
    if flow in ['latent', 'nlatent']:
        warnings.warn('NeuTra samples for fully coupled dense flows including latent parameters are irrelevant, see code.')

    print(f"\nTransformation w/ {n_flow} flows (flow: {flow}, splines? {num_bins is not None}) - hidden layers={n_hidden} - {non_linearity} nonlinearity")
    param_init, flow, flow_inv = flows[flow](d, n_flow, n_hidden, non_lins[non_linearity], num_bins)
    reverse, forward = distances[distance](logprob_fn, flow, flow_inv)
    init_param = param_init(rng_key, jrnd.normal(rng_key, shape=(d,)))
    return init_param, flow, flow_inv, reverse, forward


def run_precondition(
    rng_key, init_param, position,
    optim, reverse,
    batch_size, n_iter,
):
    tic1 = pd.Timestamp.now()
    p, unraveler_fn = ravel_pytree(position)
    U = jax.vmap(unraveler_fn)(
        jax.random.normal(rng_key, shape=(batch_size,) + p.shape)
    )
    opt_state = optim.init(init_param)
    (param, opt_state), loss_value = optimize(
        init_param, opt_state, reverse, 
        optim, n_iter, U,
    )
    tic2 = pd.Timestamp.now()
    print("Runtime for pre-conditioning", tic2 - tic1)
    return param


def run_tess(
    rng_key, logprob_fn,
    init_position, n_warm, n_iter,
    init_param, optim, flow, forward,
    batch_iter, batch_size, maxiter,
    batch_fn = jax.pmap,
):
    check_shapes = jax.tree_leaves(
        jax.tree_map(lambda p: p.shape[0] == batch_iter * batch_size, init_position)
    )
    if not all(check_shapes):
        raise ValueError("Num. of chains on initial positions don't match batch_size * batch_iter")

    tic1 = pd.Timestamp.now()
    k_warm, k_sample = jrnd.split(rng_key)
    if n_warm > 0:
        warmup = atess(logprob_fn, optim, init_param, flow, forward, batch_iter, batch_size, n_warm, maxiter, eca=False, batch_fn=batch_fn)
        chain_state, kernel, param = warmup.run(k_warm, init_position)
        init_state = chain_state.states
    else:
        init, kernel = tess(logprob_fn, lambda u: (u, 0))
        init_state = batch_fn(init)(init_position)
    def one_chain(k_sam, init_state):
        state, info = inference_loop0(k_sam, init_state, kernel, n_iter)
        return state.position, info.subiter.mean()
    k_sample = jrnd.split(k_sample, batch_iter * batch_size)
    samples, subiter = batch_fn(one_chain)(k_sample, init_state)
    # print(subiter)
    tic2 = pd.Timestamp.now()

    sec = (tic2 - tic1).total_seconds()
    do_summary(samples, logprob_fn, sec)
    print("Runtime for TESS", (tic2 - tic1).total_seconds())
    return samples, param


def inference_loop_svgd(init_state, step, n_iter):
    def one_iter(state, _):
        state = step(state)
        return state, state
    state, _ = jax.lax.scan(one_iter, init_state, jax.numpy.arange(n_iter))
    return state

def run_svgd(logprob_fn, init_position, n_warm, n_iter, batch_shape, optim):
    tic1 = pd.Timestamp.now()
    if optim is not None:
        init, step = svgd(jax.grad(logprob_fn), optim)
    else:
        init, step = coin_svgd(jax.grad(logprob_fn))
    init_state = init(init_position)
    init_state = update_median_heuristic(init_state)
    state = inference_loop_svgd(init_state, step, n_warm + n_iter)
    particles = state.particles
    tic2 = pd.Timestamp.now()

    sec = (tic2 - tic1).total_seconds()
    particles = jax.tree_util.tree_map(lambda p: p.reshape((batch_shape, n_iter) + p.shape[1:]), particles)
    do_summary(particles, logprob_fn, sec)
    print("Runtime for SVGD", (tic2 - tic1).total_seconds())
    return particles