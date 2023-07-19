"""Markovian Score Climbing with Conditional Importance sampling"""

from typing import Callable

import jax

from blackjax.mcmc.cis import build_kernel, init
from blackjax.adaptation.chain_adaptation import cross_chain, ChainState
from blackjax.adaptation.atess import optimize
from blackjax.base import AdaptationAlgorithm
from blackjax.types import PyTree, PRNGKey


def base(
    kernel_factory,
    optim,
    loss,
    num_batch: int,
    batch_size: int,
    n_iter: int = 10,
    get_loss = None,
):
    def parameter_gn(batch_state, key, param, state):
        batch_position = batch_state.position
        if get_loss is None:
            param_state, loss_value = optimize(
                param,
                state,
                loss,
                optim,
                n_iter,
                batch_position,
            )
        else:
            param_state, loss_value = optimize(
                param,
                state,
                get_loss(batch_position),
                optim,
                n_iter,
                key=key,
            )
        return param_state

    init, update = cross_chain(
        kernel_factory, parameter_gn, num_batch * batch_size, jax.vmap
    )

    def final(last_state: ChainState, param_state: PyTree) -> PyTree:
        param_state = parameter_gn(
            last_state.states,
            last_state.current_iter,
            *param_state,
        )
        return kernel_factory(*param_state), param_state[0]

    return init, update, final


def msc(
    logprob_fn: Callable,
    optim,
    init_param,
    flow,
    loss,
    num_batch: int,
    batch_size: int,
    num_steps: int = 1000,
    n_iter: int = 1,
    num_importance_samples: int = 1,
    get_loss = None,
) -> AdaptationAlgorithm:

    kernel = build_kernel(num_importance_samples)

    def kernel_factory(param: PyTree, opt_state: PyTree):
        def kernel_fn(rng_key, state):
            return kernel(
                rng_key,
                state,
                logprob_fn,
                lambda u: flow(u, param),
            )

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

    init_batch = jax.vmap(lambda pp: init(pp))
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
