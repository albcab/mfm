"""Public API for General Cross-Chain Adaptations"""

from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from blackjax.types import PRNGKey, PyTree


class ChainState(NamedTuple):
    states: NamedTuple
    current_iter: int


def cross_chain(
    kernel_factory: Callable,
    parameter_gn: Callable,
    num_chain: int,
    batch_fn: Callable = jax.vmap,
):
    def init(initial_states: NamedTuple) -> ChainState:
        check_leaves_shape = jax.tree_leaves(
            jax.tree_map(lambda s: s.shape[0] == num_chain, initial_states)
        )
        if not all(check_leaves_shape):
            raise ValueError(
                "Cross-chain adaptation got inconsistent sizes for array axes on *State. Every array's shape must be of the form (num_chain, ...)"
            )
        return ChainState(initial_states, 0)

    def update(
        rng_key: PRNGKey, state: ChainState, *param
    ) -> Tuple[ChainState, PyTree, NamedTuple]:
        parameters = parameter_gn(state.states, state.current_iter, *param)
        kernel = batch_fn(kernel_factory(*parameters))
        keys = jax.random.split(rng_key, num_chain)
        new_states, infos = kernel(keys, state.states)
        return ChainState(new_states, state.current_iter + 1), parameters, infos

    return init, update


def parallel_eca(
    kernel_factory: Callable,
    parameter_gn: Callable,
    num_batch: int,
    batch_size: int,
    batch_fn: Callable = jax.vmap,
):
    def init(initial_states: NamedTuple) -> ChainState:
        check_leaves_shape = jax.tree_leaves(
            jax.tree_map(
                lambda s: s.shape[:2] == (num_batch, batch_size), initial_states
            )
        )
        if not all(check_leaves_shape):
            raise ValueError(
                "Parallel Ensemble Chain Adaptations got inconsistent sizes for array axes on *State. Every array's shape must be of the form (num_batch, batch_size, ...)"
            )
        return ChainState(initial_states, 0)

    def update(
        rng_key: PRNGKey, state: ChainState, *param
    ) -> Tuple[ChainState, PyTree, NamedTuple]:
        parameters = batch_fn(
            lambda batch_state, *batch_param: parameter_gn(
                batch_state, state.current_iter, *batch_param
            )
        )(state.states, *param)
        params = jax.tree_map(lambda p: jnp.concatenate([p[1:], p[:1]]), parameters)
        rng_keys = jax.random.split(rng_key, num_batch)
        skip = jnp.ones(num_batch).at[state.current_iter % num_batch].set(0)

        @batch_fn
        def batch_update(rng_key, skip, batch_state, params):
            rng_keys = jax.random.split(rng_key, batch_size)
            kernel = batch_fn(kernel_factory(*params))
            batch_state = jax.lax.cond(  # doesn't return infos because the pytree has to be of the same structure...
                skip,
                lambda _: kernel(rng_keys, batch_state)[0],
                lambda _: batch_state,
                operand=None,
            )
            return batch_state, None  # info

        states, infos = batch_update(rng_keys, skip, state.states, params)
        state = ChainState(states, state.current_iter + 1)
        return state, parameters, infos

    return init, update