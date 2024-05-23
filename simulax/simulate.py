from typing import Callable

import jax

from simulax.types import PRNGKey


def simulator(likelihood_gn: Callable) -> Callable:
    def simulate_fn(
        rng_key: PRNGKey, num_obs: int, prior_gn: Callable, *prior_args, **prior_kwargs
    ):
        prior_key, likelihood_key = jax.random.split(rng_key)
        params = prior_gn(prior_key, *prior_args, **prior_kwargs)
        # likelihood_keys = jax.random.split(likelihood_key, num_obs)
        # data = jax.vmap(lambda key: likelihood_gn(key, params))(likelihood_keys)
        data = likelihood_gn(likelihood_key, params)
        return params, data

    return simulate_fn
