from typing import Callable, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.flatten_util import ravel_pytree


def inference_loop(rng, init_state, kernel, n_iter, param):
    keys = jrnd.split(rng, n_iter)
    def step(state, key):
        state, info = kernel(key, state, param)
        return state, (state, info)
    _, (states, info) = jax.lax.scan(step, init_state, keys)
    return states, info

def inference_loop0(rng, init_state, kernel, n_iter):
    keys = jrnd.split(rng, n_iter)
    def step(state, key):
        state, info = kernel(key, state)
        return state, (state, info)
    _, (states, info) = jax.lax.scan(step, init_state, keys)
    return states, info


def stein_disc(X, logprob_fn, beta=-1/2) -> Tuple:
    """Stein discrepancy with inverse multi-quadric kernel,
    i.e. (1 + (x - x')T(x - x')) ** beta
    returns U-Statistic (unbiased) and V-statistic (biased)
    """

    # X = jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), X)
    if isinstance(X, dict):
        d = 0
        for name, x in X.items():
            s = x.shape
            if not s[1:]:
                d += 1
            else:
                d += s[1]
        T = s[0]
        sub = lambda x, x_: ravel_pytree(x)[0] - ravel_pytree(x_)[0]
        grad = lambda x: ravel_pytree(jax.grad(logprob_fn)(x))[0]
    else: 
        T, d = X.shape
        sub = lambda x, x_: x - x_
        grad = jax.grad(logprob_fn)
    beta = -beta

    # gradk = lambda diff, dot_prod: -2 * beta * (1 + dot_prod) ** (-beta - 1) * diff
    # gradgradk = lambda diff, dot_prod: -2 * beta * jnp.sum(-diff ** 2 * 2 * (-beta - 1) * (1 + dot_prod) ** (-beta - 2) - (1 + dot_prod) ** (-beta - 1))

    # def disc2(x, x_):
    #     diff = sub(x, x_)
    #     dot_prod = jnp.dot(diff, diff)
    #     dx = grad(x)
    #     dx_ = grad(x_)
    #     return (
    #         jnp.dot(dx, dx_) * (1 + dot_prod) ** (-beta)
    #         + jnp.dot(dx, -gradk(diff, dot_prod)) + jnp.dot(dx_, gradk(diff, dot_prod))
    #         + gradgradk(diff, dot_prod)
    #     )

    def disc(x, x_):
        diff = sub(x, x_)
        dot_prod = jnp.dot(diff, diff)
        dx = grad(x)
        dx_ = grad(x_)
        return (
            -4 * beta * (beta+1) * dot_prod / ((1 + dot_prod) ** (beta + 2))
            + 2 * beta * (d + jnp.dot(dx - dx_, diff)) / ((1 + dot_prod) ** (1 + beta))
            + jnp.dot(dx, dx_) / ((1 + dot_prod) ** beta)
        )

    _disc = jax.vmap(disc, (None, 0))
    # _disc2 = jax.vmap(disc2, (None, 0))
    # try:
    #     mc_sum = jax.vmap(_disc, (0, None))(X, X).sum()
    #     # mc_sum2 = jax.vmap(_disc2, (0, None))(X, X).sum()
    #     # print(mc_sum, mc_sum2)
    # except RuntimeError:
    mc_sum = jax.lax.map(lambda x: _disc(x, X).sum(), X).sum()
    return (mc_sum - jax.vmap(lambda x: disc(x, x))(X).sum()) / (T * (T-1)), mc_sum / T**2


def max_mean_disc(X, Y):
    if isinstance(X, dict):
        for name, x in X.items():
            m = x.shape[0]
            break
        sub = lambda x, x_: ravel_pytree(x)[0] - ravel_pytree(x_)[0]
    else: 
        m = X.shape[0]
        sub = lambda x, x_: x - x_

    def kernel(x, y, sigma2=1.):
        diff = sub(x, y)
        return jnp.exp(-.5 * jnp.square(diff).sum() / sigma2)
    # def kernel(x, y, beta=-.5):
    #     diff = sub(x, y)
    #     return (1 + jnp.square(diff).sum()) ** beta
    
    _disc = jax.vmap(kernel, (None, 0))
    disc = jax.vmap(_disc, (0, None))
    disc_x = disc(X, X).sum() - m
    disc_y = disc(Y, Y).sum() - m
    disc_xy = disc(X, Y).sum()
    m2 = m * m
    return disc_x / (m2 - m) - 2 * disc_xy / m2 + disc_y / (m2 - m) 


def _fft_next_fast_len(target):
    # find the smallest number >= N such that the only divisors are 2, 3, 5
    # works just like scipy.fftpack.next_fast_len
    if target <= 2:
        return target
    while True:
        m = target
        while m % 2 == 0:
            m //= 2
        while m % 3 == 0:
            m //= 3
        while m % 5 == 0:
            m //= 5
        if m == 1:
            return target
        target += 1

def autocorrelation(x, axis=0):
    """
    Computes the autocorrelation of samples at dimension ``axis``.

    :param numpy.ndarray x: the input array.
    :param int axis: the dimension to calculate autocorrelation.
    :return: autocorrelation of ``x``.
    :rtype: numpy.ndarray
    """
    # Ref: https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation
    # Adapted from Stan implementation
    # https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/autocorrelation.hpp
    N = x.shape[axis]
    M = _fft_next_fast_len(N)
    M2 = 2 * M

    # transpose axis with -1 for Fourier transform
    x = np.swapaxes(x, axis, -1)

    # centering x
    centered_signal = x - x.mean(axis=-1, keepdims=True)

    # Fourier transform
    freqvec = np.fft.rfft(centered_signal, n=M2, axis=-1)
    # take square of magnitude of freqvec (or freqvec x freqvec*)
    freqvec_gram = freqvec * np.conjugate(freqvec)
    # inverse Fourier transform
    autocorr = np.fft.irfft(freqvec_gram, n=M2, axis=-1)

    # truncate and normalize the result, then transpose back to original shape
    autocorr = autocorr[..., :N]
    # autocorr = autocorr / np.arange(N, 0.0, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        autocorr = autocorr / autocorr[..., :1] / 2
    return np.swapaxes(autocorr, axis, -1)