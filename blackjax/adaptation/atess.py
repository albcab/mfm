"""Adaptive Transport Elliptical Slice sampler"""

from typing import Callable

import jax
import jax.numpy as jnp
import optax

from blackjax.mcmc.tess import build_kernel, init
import blackjax.adaptation.chain_adaptation as chain_adaptation
from blackjax.base import AdaptationAlgorithm
from blackjax.types import PyTree, PRNGKey


def base(
    kernel_factory,
    optim,
    loss,
    num_batch: int,
    batch_size: int,
    n_iter: int = 10,
    eca: bool = True,
    batch_fn: Callable = jax.pmap,
):
    def parameter_gn(batch_state, current_iter, param, state):
        batch_position = batch_state.position
        param_state, loss_value = optimize(
            param,
            state,
            loss,
            optim,
            n_iter,
            batch_position,
        )
        return param_state

    if eca:
        init, update = chain_adaptation.parallel_eca(
            kernel_factory, parameter_gn, num_batch, batch_size, batch_fn
        )
    else:
        init, update = chain_adaptation.cross_chain(
            kernel_factory, parameter_gn, num_batch * batch_size, batch_fn
        )

    def final(last_state: chain_adaptation.ChainState, param_state: PyTree) -> PyTree:
        if eca:
            return None
        param_state = parameter_gn(
            last_state.states,
            last_state.current_iter,
            *param_state,
        )
        return kernel_factory(*param_state), param_state[0]

    return init, update, final


def atess(
    logprob_fn: Callable,
    optim,
    init_param,
    flow,
    loss,
    num_batch: int,
    batch_size: int,
    num_steps: int = 1000,
    n_iter: int = 1,
    *,
    eca: bool = False,
    batch_fn: Callable = jax.pmap,
) -> AdaptationAlgorithm:

    kernel = build_kernel()

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
        eca,
        batch_fn,
    )

    batch_init = batch_fn(lambda pp: init(pp))

    if eca:

        @batch_fn
        def init_batch(batch_pposition):
            batch_state = batch_init(batch_pposition)
            return batch_state

        params = batch_fn(lambda _: (init_param, optim.init(init_param)))(
            jnp.zeros(num_batch)
        )

    else:
        init_batch = batch_init
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

        return last_state, kernel, param#warmup_states

    return AdaptationAlgorithm(run)  # type: ignore[arg-type]


def optimize(param, state, loss, optim, n_iter, positions=None, key=None):
    def step_fn(carry, key):
        params, opt_state = carry
        if positions is not None:
            loss_value, grads = jax.value_and_grad(loss)(params, positions)
        else:
            loss_value, grads = jax.value_and_grad(loss)(params, key)
        updates, opt_state_ = optim.update(grads, opt_state, params)
        params_ = optax.apply_updates(params, updates)
        # return (params_, opt_state_), loss_value
        return jax.lax.cond(
            jnp.isfinite(loss_value)
            & jnp.isfinite(jax.flatten_util.ravel_pytree(grads)[0]).all(),
            lambda _: ((params_, opt_state_), loss_value),
            lambda _: ((params, opt_state), jnp.nan),
            None,
        )
    if key is None:
        param_state, loss_value = jax.lax.scan(step_fn, (param, state), jnp.arange(n_iter))
    else:
        keys = jax.random.split(key, n_iter)
        param_state, loss_value = jax.lax.scan(step_fn, (param, state), keys)
    return param_state, loss_value