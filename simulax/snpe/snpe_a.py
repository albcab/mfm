from typing import Callable

import jax

from simulax.snpe.base import SNPE
from simulax.types import PRNGKey


class SNPE_A(SNPE):
    def get_loss_function(self, rng_key: PRNGKey, num_particles: int) -> Callable:
        simulator_keys = jax.random.split(rng_key, num_particles)
        params, data = jax.vmap(
            lambda key: self.simulator(
                key, self.num_obs, self.prior_gn, *self.prior_args, **self.prior_kwargs
            )
        )(simulator_keys)

        def loss(approx_params):
            logprobs = jax.vmap(
                lambda param, data: self.approx_logprob_fn(approx_params, param, data)
            )(params, data)
            return jax.numpy.sum(logprobs)

        return loss
