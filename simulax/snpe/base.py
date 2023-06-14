import abc
from typing import Callable

import jax

from simulax.simulate import simulator
from simulax.types import PRNGKey


class SNPE(metaclass=abc.ABCMeta):
    def __init__(
        self,
        approx_logprob_fn: Callable,
        num_obs: int,
        likelihood_gn: Callable,
        prior_gn: Callable,
        *prior_args,
        **prior_kwargs
    ) -> None:
        self.approx_logprob_fn = approx_logprob_fn
        self.num_obs = num_obs
        self.simulator = simulator(likelihood_gn)
        self.prior_gn = prior_gn
        self.likelihood_gn = likelihood_gn
        self.prior_args = prior_args
        self.prior_kwargs = prior_kwargs

    def update_prior_generator(self, prior_gn: Callable):
        self.prior_gn = prior_gn

    def update_prior_params(self, *prior_args, **prior_kwargs):
        self.prior_args = prior_args
        self.prior_kwargs = prior_kwargs

    def update_approx_logprob_function(self, approx_logprob_fn: Callable):
        self.approx_logprob_fn = approx_logprob_fn

    @abc.abstractmethod
    def get_loss_function(self, rng_key: PRNGKey, num_particles: int) -> Callable:
        """Returns loss as a function of parameters of the approximation"""
