import pandas as pd

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import optax

from mcmc_utils import inference_loop0
from execute import do_summary

from blackjax.mcmc.tess import tess
from simulax.snpe.snpe_a import SNPE_A


def taylor_run(dist, args, init_param, flow, flow_inv, optim, N, likeligood_gn, prior_gn, *prior_args):

    [n_warm, n_iter] = args.sampling_param
    [batch_iter, batch_size] = args.batch_shape
    kflow, ksam, kinit = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    dist.initialize_model(kinit, batch_iter * batch_size)

    def approx_fn(approx_params, params, data):
        u, ldj = flow_inv(params, *data, approx_params)
        u = ravel_pytree(u)[0]
        return .5 * jnp.dot(u, u) - ldj

    snpe = SNPE_A(approx_fn, 1, likeligood_gn, prior_gn, *prior_args)
    num_particles = 100
    num_iter = args.preconditon_iter
    sampling_param = simulate_flow(kflow, init_param, optim, snpe, num_particles, num_iter)

    print("TESS w/ sampling precond.")
    samples = run_tess(ksam, dist.logprob_fn, dist.init_params,
        n_iter, sampling_param, flow, batch_iter, batch_size, dist._obs, dist._times)
    
    flow_ = lambda u, param: flow(u, dist._obs, dist._times, param)
    flow_inv_ = lambda x, param: flow_inv(x, dist._obs, dist._times, param)

    return samples, sampling_param, flow_, flow_inv_


def simulate_flow(rng_key, init_param, optim, snpe, num_particles, num_iter):
    tic1 = pd.Timestamp.now()
    loss = snpe.get_loss_function(rng_key, num_particles)
    def step_fn(carry, _):
        params, opt_state = carry
        loss_value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state_ = optim.update(grads, opt_state, params)
        params_ = optax.apply_updates(params, updates)
        return ((params_, opt_state_), loss_value)
        # return jax.lax.cond(
        #     jnp.isfinite(loss_value)
        #     & jnp.isfinite(jax.flatten_util.ravel_pytree(grads)[0]).all(),
        #     lambda _: ((params_, opt_state_), loss_value),
        #     lambda _: ((params, opt_state), jnp.nan),
        #     None,
        # )
    
    opt_state = optim.init(init_param)
    (param, _), _ = jax.lax.scan(step_fn, (init_param, opt_state), jnp.arange(num_iter))
    tic2 = pd.Timestamp.now()
    print("Runtime for simulation+optimization", tic2 - tic1)
    return param


def run_tess(
    rng_key, logprob_fn,
    init_position, n_iter, param,
    flow, batch_iter, batch_size,
    obs, times,
):
    tic1 = pd.Timestamp.now()
    init, kernel = tess(logprob_fn, lambda u: flow(u, obs, times, param))
    init_state = jax.vmap(init)(init_position)
    def one_chain(k_sam, init_state):
        state, info = inference_loop0(k_sam, init_state, kernel, n_iter)
        return state.position, info.subiter.mean()
    k_sample = jax.random.split(rng_key, batch_iter * batch_size)
    samples, subiter = jax.vmap(one_chain)(k_sample, init_state)
    # print(subiter)
    tic2 = pd.Timestamp.now()

    sec = (tic2 - tic1).total_seconds()
    do_summary(samples, logprob_fn, sec)
    print("Runtime for TESS", (tic2 - tic1).total_seconds())
    return samples