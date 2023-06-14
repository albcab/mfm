import abc
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.experimental.host_callback import id_print

import haiku as hk
import distrax as dx


def affine_coupling(params):
    return dx.ScalarAffine(shift=params[:, 0], log_scale=params[:, 1])

def rquad_spline_coupling(params):
    return dx.RationalQuadraticSpline(
        params, range_min=-3., range_max=3.)

        
def make_dense(d, hidden_dims, norm, non_linearity, num_bins):
    layers = []
    if norm:
        layers.append(hk.LayerNorm(-1, True, True))
    for _ in range(hidden_dims):
        layers.append(hk.Linear(d, w_init=hk.initializers.VarianceScaling(.01), b_init=hk.initializers.RandomNormal(.01)))
        if norm:
            layers.append(hk.LayerNorm(-1, True, True))
        layers.append(non_linearity)
    if num_bins:
        layers.append(
            hk.Linear(3 * num_bins + 1, w_init=hk.initializers.VarianceScaling(.01), b_init=hk.initializers.RandomNormal(.01))
        )
    else:
        layers.extend([
            hk.Linear(2 * d, w_init=hk.initializers.VarianceScaling(.01), b_init=hk.initializers.RandomNormal(.01)),
            hk.Reshape((d, 2), preserve_dims=-1)
        ])
    return hk.Sequential(layers)


def make_cond(out_dim, first_layer, self, batch=False):
    layers = [first_layer]
    n_param = 3 * self.num_bins + 1 if self.num_bins else 2
    for _ in range(self.hidden_dims):
        layers.append(hk.Linear(n_param * out_dim))
        layers.append(self.non_linearity)
    if batch:
        layers.append(BatchLinear(n_param))
    else:
        layers.append(hk.Linear(n_param * out_dim))
        layers.append(hk.Reshape((out_dim, n_param), preserve_dims=-1))
    return hk.Sequential(layers)


class BatchLinear(hk.Module):
    def __init__(self, n_param, name=None):
        super().__init__(name)
        self.n_param = n_param

    def __call__(self, x):
        return jax.vmap(lambda x: hk.Linear(self.n_param)(x))(x)


class Sufficient(hk.Module):
    def __init__(self, transform, axis=None, name=None):
        super().__init__(name)
        self.axis = axis
        self.transform = transform

class RealSufficient(Sufficient):
    def __call__(self, x):
        x = self.transform(x)
        x = (x - x.mean()) / x.std()
        return x
        first = jnp.sum(x, axis=self.axis)
        second = jnp.sum(x**2, axis=self.axis)
        third = jnp.sum(x**3, axis=self.axis)
        fourth = jnp.sum(x**4, axis=self.axis)
        reg = jnp.array([first, second, third, fourth])
        if self.axis is not None:
            reg = ((reg - reg.mean(axis=0)) / reg.std(axis=0)).T
        else:
            reg = (reg - reg.mean()) / reg.std()
        return reg

class PositiveSufficient(Sufficient):
    def __call__(self, log_x):
        log_x = self.transform(log_x)
        logfirst = jnp.sum(log_x, axis=self.axis)
        nfirst = jnp.sum(-jnp.exp(log_x), axis=self.axis)
        nsecond = jnp.sum(-jnp.exp(log_x)**2, axis=self.axis)
        nthird = jnp.sum(-jnp.exp(log_x)**3, axis=self.axis)
        reg = jnp.array([logfirst, nfirst, nsecond, nthird])
        if self.axis is not None:
            reg = ((reg - reg.mean(axis=0)) / reg.std(axis=0)).T
        else:
            reg = (reg - reg.mean()) / reg.std()
        return reg


class Flow(metaclass=abc.ABCMeta):

    def __init__(self, num_bins, hidden_dims, non_linearity):
        if num_bins:
            self.coupling_fn = affine_coupling
        else:
            self.coupling_fn = affine_coupling
        self.num_bins = num_bins
        self.hidden_dims = hidden_dims
        self.non_linearity = non_linearity

    @abc.abstractmethod
    def flows(self):
        pass
    
    def get_utilities(self):
        forward_and_log_det = hk.transform(lambda u: self.flows().forward_and_log_det(u))
        inverse_and_log_det = hk.transform(lambda x: self.flows().inverse_and_log_det(x))

        def flow(u, param):
            u, unravel_fn = ravel_pytree(u)
            x, ldj = forward_and_log_det.apply(param, None, u)
            return unravel_fn(x), ldj
        
        def flow_inv(x, param):
            x, unravel_fn = ravel_pytree(x)
            u, ldj = inverse_and_log_det.apply(param, None, x)
            return unravel_fn(u), ldj

        return forward_and_log_det.init, flow, flow_inv


class Coupling(Flow):
    
    def __init__(self,
        d: int, n_flow: int,
        hidden_dims: int, non_linearity: Callable, norm: bool,
        num_bins: int = None,
    ):
        super().__init__(num_bins, hidden_dims, non_linearity)
        self.split = int(d/2 + .5)
        self.d = d
        self.n_flow = n_flow
        self.norm = norm

    def flows(self):
        flows = []
        if self.num_bins:
            flows.append(shift_scale(self.d))
        for _ in range(self.n_flow):
            encoder = make_dense(self.split, self.hidden_dims, self.norm, self.non_linearity, self.num_bins)
            flows.append(dx.SplitCoupling(self.split, 1, encoder, self.coupling_fn, swap=True))
            decoder = make_dense(self.d - self.split, self.hidden_dims, self.norm, self.non_linearity, self.num_bins)
            flows.append(dx.SplitCoupling(self.split, 1, decoder, self.coupling_fn, swap=False))
        if self.num_bins:
            flows.append(shift_scale(self.d))
        return dx.Chain(flows)


class ShiftScale(Flow):

    def __init__(self, d):
        self.d = d

    def flows(self):
        return shift_scale(self.d)


def shift_scale(d):
    lin = hk.Sequential([
        hk.Linear(2 * d, w_init=jnp.zeros, b_init=hk.initializers.RandomNormal(.1)), 
        hk.Reshape((2, d), preserve_dims=-1)
    ])
    return dx.MaskedCoupling(jnp.zeros(d).astype(bool), lin, affine_coupling)
