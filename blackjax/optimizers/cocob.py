from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from blackjax.types import PyTree


class COCOBOptimizer(NamedTuple):
    init_particles: PyTree
    cumulative_gradients: PyTree
    scale: PyTree
    subgradients: PyTree
    reward: PyTree


def cocob(alpha: float = 100, eps: float = 1e-8):
    """The parameter-free backproagating version of the COntinuous COin Betting optimizer.

    Algorithm for stochastic subgradient descent. Uses a gambling algorithm to find the
    minimizer of a non-smooth objective function by accessing its subgradients. All we need
    is a good gambling strategy. See Algorithm 2 of :cite:p:`orabona2017training`.

    Parameters
    ----------
    alpha
        parameter of the COCOB optimizer
    eps
        jitter term to avoid dividing by 0

    Returns
    -------
    optax.GradientTransformation containing initialization and update functions. Works exactly like
    an optax optimizer and is designed to work with the updating of particles in blackjax.vi.svgd
    """
    def init_fn(initial_particles: PyTree):
        init_adapt = jax.tree_util.tree_map(
            lambda p: jnp.zeros(p.shape), initial_particles
        )
        init_scale = jax.tree_util.tree_map(
            lambda p: eps * jnp.ones(p.shape), initial_particles
        )
        return COCOBOptimizer(
            initial_particles,
            init_adapt,
            init_scale,
            init_adapt,
            init_adapt,
        )

    def update_fn(gradient: PyTree, opt_state: COCOBOptimizer, particles: PyTree):
        init_particles, cumulative_gradients, scale, subgradients, reward = opt_state

        scale = jax.tree_util.tree_map(
            lambda L, c: jnp.maximum(L, jnp.abs(c)), scale, gradient
        )
        subgradients = jax.tree_util.tree_map(
            lambda G, c: G + jnp.abs(c), subgradients, gradient
        )
        reward = jax.tree_util.tree_map(
            lambda R, c, p, p0: jnp.maximum(R - c * (p - p0), 0),
            reward,
            gradient,
            particles,
            init_particles,
        )
        cumulative_gradients = jax.tree_util.tree_map(
            lambda C, c: C - c, cumulative_gradients, gradient
        )

        update = jax.tree_util.tree_map(
            lambda p, p0, C, L, G, R: -p
            + (p0 + C / (L * jnp.maximum(G + L, alpha * L)) * (L + R)),
            particles,
            init_particles,
            cumulative_gradients,
            scale,
            subgradients,
            reward,
        )
        opt_state = COCOBOptimizer(
            init_particles, cumulative_gradients, scale, subgradients, reward
        )

        return update, opt_state

    return optax.GradientTransformation(init_fn, update_fn)
