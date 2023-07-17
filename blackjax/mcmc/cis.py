"""Public API for the Conditional Importance Sampling Kernel"""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax.base import SamplingAlgorithm
from blackjax.types import PRNGKey, PyTree, Array

from jax.experimental.host_callback import id_print
__all__ = ["CISState", "CISInfo", "init", "kernel"]


class CISState(NamedTuple):
    position: PyTree
    pullback_position: float


class CISInfo(NamedTuple):
    positions: PyTree
    pullback_positions: PyTree
    weights: Array


def init(pullback_position: PyTree):
    return CISState(pullback_position, pullback_position)


def build_kernel(num_sam: int):
    def kernel(
        rng_key: PRNGKey,
        state: CISState,
        logprob_fn: Callable,
        flow: Callable,
    ) -> Tuple[CISState, CISInfo]:
        _, pullback_position = state
        pullback_position, unravel_fn = ravel_pytree(pullback_position)
        reference_gn = lambda k: jax.random.normal(k, pullback_position.shape)
        key_generate, key_sample = jax.random.split(rng_key)

        def transform_fn(u):
            x, ldj = flow(unravel_fn(u))
            return x, jnp.exp(logprob_fn(x) + ldj + 0.5 * jnp.dot(u, u))
        
        keys = jax.random.split(key_generate, num_sam)
        pullback_positions = jax.vmap(reference_gn)(keys)
        pullback_positions = jnp.vstack([pullback_position, pullback_positions])
        positions, weights = jax.vmap(transform_fn)(pullback_positions)

        indx = jax.random.choice(key_sample, num_sam + 1, p=weights)
        position = jax.tree_util.tree_map(lambda p: p[indx], positions)
        pullback_position = unravel_fn(pullback_positions[indx])

        state = CISState(position, pullback_position)
        info = CISInfo(positions, pullback_positions, weights)
        return state, info

    return kernel


class cis:

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logprob_fn: Callable,
        flow: Callable,
        num_importance_samples: int = 1,
    ) -> SamplingAlgorithm:

        kernel = cls.build_kernel(num_importance_samples)

        def init_fn(position: PyTree):
            return cls.init(position)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(rng_key, state, logprob_fn, flow)

        return SamplingAlgorithm(init_fn, step_fn)
