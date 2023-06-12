import argparse

import pandas as pd
import numpy as np

import jax
import jax.numpy as jnp
import optax
import distrax as dx
import haiku as hk

from distributions import HorseshoeLogisticReg, ProbitReg
from execute import run
from exe import taylor_run
from flows import Flow, RealSufficient, PositiveSufficient, make_cond

from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

from jax.experimental.host_callback import id_print

class HorseshoeSufficient(hk.Module):
    def __call__(self, log_input):
        log_x = log_input[:-1] + log_input[-1]
        logfirst = log_x
        nfirst = -jax.numpy.exp(log_x)
        nsecond = -jax.numpy.exp(log_x)**2
        nthird = -jax.numpy.exp(log_x)**3
        reg = jax.numpy.vstack([logfirst, nfirst, nsecond, nthird])
        reg = ((reg - reg.mean(axis=0)) / reg.std(axis=0)).T
        return reg

class Message(Flow):

    def __init__(self, y, X, num_bins=False, hidden_dims=0, non_linearity=None): 
        super().__init__(num_bins, hidden_dims, non_linearity)
        self.y = y
        self.X = X
        _, j = X.shape
        d = j * 2 + 1
        false = [False] * j
        true = [True] * j
        self.tau_mask = np.array(true + true + [False])
        self.lambda_mask = np.array(true + false + [True])
        self.beta_mask = np.array(false + true + [True])
        self.d = d
        self.j = j

    def flows(self):
        flows = []
        tau_prior_layer = PositiveSufficient(axis=None, transform=lambda _: jnp.log(1))
        tau_prior = make_cond(self.d, tau_prior_layer, self)
        flows.append(dx.MaskedCoupling(self.tau_mask, tau_prior, self.coupling_fn))
        
        lambda_prior_layer = PositiveSufficient(axis=None, transform=lambda _: jnp.log(1))        
        lambda_prior = make_cond(self.d, lambda_prior_layer, self)
        flows.append(dx.MaskedCoupling(self.lambda_mask, lambda_prior, self.coupling_fn))
        
        lambda_tau = lambda log_x: jnp.atleast_2d(log_x[:-1] + log_x[-1])
        beta_prior_layer = PositiveSufficient(axis=0, transform=lambda_tau)
        beta_prior = make_cond(self.j, beta_prior_layer, self, batch=True)
        flows.append(dx.SplitCoupling(self.j, 1, beta_prior, self.coupling_fn, swap=True))

        #what is the best use for the first moment? OLS?
        data = lambda _: self.y * self.X.T
        beta_lik_layer = RealSufficient(axis=1, transform=data)
        beta_lik = make_cond(self.j, beta_lik_layer, self, batch=True)
        flows.append(dx.SplitCoupling(self.j, 1, beta_lik, self.coupling_fn, swap=True))

        beta = lambda x: jnp.atleast_2d(jnp.concatenate([x[:self.j], x[:self.j], x[:1]]))
        lambda_lik_layer = RealSufficient(axis=0, transform=beta)
        lambda_lik = make_cond(self.d, lambda_lik_layer, self, batch=True)
        flows.append(dx.MaskedCoupling(self.lambda_mask, lambda_lik, self.coupling_fn))

        tau_lik_layer = RealSufficient(axis=None, transform=beta)
        tau_lik = make_cond(self.d, tau_lik_layer, self)
        flows.append(dx.MaskedCoupling(self.tau_mask, tau_lik, self.coupling_fn))

        return dx.Chain(flows)


def main(args):

    print("Loading German credit data...")
    data = pd.read_table('german.data-numeric', header=None, delim_whitespace=True)
    ### Pre processing data as in NeuTra paper
    y = -1 * (data.iloc[:, -1].values - 2)
    X = data.iloc[:, :-1].apply(lambda x: -1 + (x - x.min()) * 2 / (x.max() - x.min()), axis=0).values
    # X = data.iloc[:, :-1].apply(lambda x: (x - x.mean()) / x.std(), axis=0).values
    X = np.concatenate([np.ones((1000, 1)), X], axis=1)
    N_OBS, N_REG = X.shape

    [n_warm, n_iter] = args.sampling_param
    schedule = optax.exponential_decay(init_value=2.5e-3,
        transition_steps=n_warm-10, decay_rate=.1, transition_begin=10)
    optim = optax.adam(schedule)

    N_PARAM = N_REG * 2 + 1
    print("\n\nSetting up German credit logistic horseshoe model...")
    dist = HorseshoeLogisticReg(X, y)

    # run(dist, args, optim, N_PARAM, batch_fn=jax.vmap)

    flow = Message(y, X)
    taylor_run(dist, args, flow, optim, N_PARAM, jax.vmap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('-m', '--max-iter', type=int, default=1)
    parser.add_argument('-b', '--batch-shape', type=int, nargs=2, default=[4, 32])
    parser.add_argument('-nl', '--non-linearity', type=str, default='relu')
    parser.add_argument('-f', '--flow', type=str, default='coupling')
    parser.add_argument('-d', '--distance', type=str, default='kld')
    parser.add_argument('-nh', '--n-hidden', type=int, default=2)
    parser.add_argument('-nf', '--n-flow', type=int, default=2)
    parser.add_argument('-nb', '--num-bins', type=int, default=None)
    parser.add_argument(
        "-s", "--sampling-param", type=int, nargs=3,
        help="Sampling parameters [n_warm, n_iter]",
        default=[400, 100]
    )
    parser.add_argument('-np', '--preconditon_iter', type=int, default=400)
    parser.add_argument('-s1', '--init_step_size', type=float, default=None)
    parser.add_argument('-s2', '--p_init_step_size', type=float, default=0.1)
    args = parser.parse_args()
    main(args)