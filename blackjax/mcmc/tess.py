"""Public API for the Transport Elliptical Slice sampler Kernel"""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax.base import SamplingAlgorithm
from blackjax.types import PRNGKey, PyTree

from jax.experimental.host_callback import id_print
__all__ = ["SliceState", "SliceInfo", "init", "kernel"]


class SliceState(NamedTuple):
    position: PyTree
    pullback_position: PyTree


class SliceInfo(NamedTuple):
    momentum: PyTree
    slice: float
    theta: float
    subiter: int


def init(pullback_position: PyTree):
    return SliceState(pullback_position, pullback_position)


def build_kernel():
    def momentum_generator(rng_key, position):
        position, _ = ravel_pytree(position)
        return jax.random.normal(rng_key, shape=jnp.shape(position))

    def kernel(
        rng_key: PRNGKey,
        state: SliceState,
        logprob_fn: Callable,
        flow: Callable,
    ) -> Tuple[SliceState, SliceInfo]:
        def slice_fn(u, m):
            x, ldj = flow(u)
            return logprob_fn(x) + ldj - 0.5 * jnp.dot(m, m)

        proposal_generator = tess_proposal(
            slice_fn,
            momentum_generator,
            lambda u: flow(u)[0],
        )
        return proposal_generator(rng_key, state)

    return kernel


class tess:

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logprob_fn: Callable,
        flow: Callable,
    ) -> SamplingAlgorithm:

        kernel = cls.build_kernel()

        def init_fn(position: PyTree):
            return cls.init(position)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(rng_key, state, logprob_fn, flow)

        return SamplingAlgorithm(init_fn, step_fn)


def ellipsis(p, m, theta, mu=0.0):
    x, unraveler = ravel_pytree(p)
    return (
        unraveler((x - mu) * jnp.cos(theta) + (m - mu) * jnp.sin(theta) + mu),
        (m - mu) * jnp.cos(theta) - (x - mu) * jnp.sin(theta) + mu,
    )


def tess_proposal(
    slice_fn: Callable,
    momentum_generator: Callable,
    T: Callable,
) -> Callable:
    def generate(rng_key: PRNGKey, state: SliceState) -> Tuple[SliceState, SliceInfo]:
        _, u_position = state
        kmomentum, kunif, ktheta = jax.random.split(rng_key, 3)
        # step 1: sample momentum
        momentum = momentum_generator(kmomentum, u_position)
        # step 2-3: get slice (y)
        # step 4: get u
        logy = slice_fn(u_position, momentum) + jnp.log(jax.random.uniform(kunif))
        # step 5-6: get theta (ellipsis move), set inital interval
        theta = 2 * jnp.pi * jax.random.uniform(ktheta)
        theta_min = theta - 2 * jnp.pi
        theta_max = theta
        # step 7-8: proposal
        u, m = ellipsis(u_position, momentum, theta)
        # step 9: get new position
        slice = slice_fn(u, m)
        # step 10-20: acceptance

        def while_fun(vals):
            rng, _, subiter, theta, theta_min, theta_max, *_ = vals
            rng, thetak = jax.random.split(rng)
            theta = jax.random.uniform(thetak, minval=theta_min, maxval=theta_max)
            u, m = ellipsis(u_position, momentum, theta)
            slice = slice_fn(u, m)
            theta_min = jnp.where(theta < 0, theta, theta_min)
            theta_max = jnp.where(theta > 0, theta, theta_max)
            subiter += 1
            return rng, slice, subiter, theta, theta_min, theta_max, u, m

        _, slice, subiter, theta, *_, u, m = jax.lax.while_loop(
            lambda vals: (vals[1] <= logy) | (~jnp.isfinite(vals[1])),
            while_fun,
            (rng_key, slice, 1, theta, theta_min, theta_max, u, m),
        )
        position = T(u)
        return (SliceState(position, u), SliceInfo(m, slice, theta, subiter))

    return generate