"""Markovian Score Climbing with MALA"""

from typing import Callable

import jax
from jax.flatten_util import ravel_pytree

from bblackjax.adaptation.msc import base
from bblackjax.mcmc.mala import build_kernel, init
from bblackjax.adaptation.chain_adaptation import cross_chain, ChainState
from bblackjax.adaptation.atess import optimize
from bblackjax.base import AdaptationAlgorithm
from bblackjax.types import PyTree, PRNGKey


def msc_mala(
    logprob_fn: Callable,
    optim,
    init_param,
    flow,
    loss,
    num_batch: int,
    batch_size: int,
    step_size: float,
    num_steps: int = 1000,
    n_iter: int = 1,
    num_mala_samples: int = 1,
    get_loss = None,
) -> AdaptationAlgorithm:

    kernel = build_kernel()

    def kernel_factory(param: PyTree, opt_state: PyTree):
        def kernel_fn(rng_key, state):
            flat_position, unravel_fn = ravel_pytree(state.position)
            key_init, key_sample = jax.random.split(rng_key)
            ref_sample = jax.random.normal(key_init, flat_position.shape)
            fresh_position = flow(unravel_fn(ref_sample), param)[0]
            fresh_state = init(fresh_position, logprob_fn)
            return jax.lax.scan(lambda s, k: kernel(k, s, logprob_fn, step_size),
                fresh_state, jax.random.split(key_sample, num_mala_samples))

        return kernel_fn

    init_, update, final = base(
        kernel_factory,
        optim,
        loss,
        num_batch,
        batch_size,
        n_iter,
        get_loss,
    )

    init_batch = jax.vmap(lambda pp: init(pp, logprob_fn))
    params = (init_param, optim.init(init_param))

    def one_step(carry, rng_key):
        state, params = carry
        state, parameters, infos = update(rng_key, state, *params)
        return (state, parameters), (state, infos)

    def run(rng_key: PRNGKey, pullback_positions: PyTree):

        states = init_batch(pullback_positions)
        init_state = init_(states)

        keys = jax.random.split(rng_key, num_steps)
        (last_state, parameters), (warmup_states, info) = jax.lax.scan(
            one_step, (init_state, params), keys
        )
        kernel, param = final(last_state, parameters)

        return last_state, kernel, param, info#warmup_states

    return AdaptationAlgorithm(run)  # type: ignore[arg-type]
