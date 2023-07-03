import pandas as pd

import jax
from jax.flatten_util import ravel_pytree
from jax.experimental.ode import odeint
from jax.experimental.host_callback import id_print

import optax
import haiku as hk

from execute import do_summary

from blackjax.mcmc.mala import mala
from fmx.data import flow_matching
from fmx.sampling import flow_matching as flow_matching_sampling

init_W = hk.initializers.VarianceScaling(0.1)
init_b = hk.initializers.RandomNormal(0.01)


def mlp_generator(out_dim, hidden_dims=[64, 64], non_linearity=jax.nn.relu, norm=False):
    layers = []
    if norm:
        layers.append(hk.LayerNorm(-1, True, True))
    for out in hidden_dims:
        layers.append(hk.Linear(out, w_init=init_W, b_init=init_b))
        layers.append(non_linearity)
        if norm:
            layers.append(hk.LayerNorm(-1, True, True))
    layers.append(hk.Linear(out_dim, w_init=init_W, b_init=init_b))
    return hk.Sequential(layers)


non_lins = {
    'tanh': jax.nn.tanh,
    'elu': jax.nn.elu,
    'relu': jax.nn.relu,
    'swish': jax.nn.swish,
}


def run(dist, args, N_PARAM, optim, step_size, prior_gn=None):
    [n_warm, n_iter] = args.sampling_param
    [batch_iter, batch_size] = args.batch_shape
    n_chain = batch_iter * batch_size
    key_precond, key_sample, key_init = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    dist.initialize_model(key_init, n_chain)
    non_lin = non_lins[args.non_linearity]

    odeintegrator = lambda func, x0: odeint(func, x0, jax.numpy.linspace(0.0, 1.0, 11))

    def mlp_vector_field(time, sample):  # , frequencies=3):
        (out_dim,) = sample.shape
        mlp = mlp_generator(out_dim, hidden_dims=[256] * args.n_hidden, non_linearity=non_lin)
        # frequencies = 2 ** jax.numpy.arange(frequencies) * jax.numpy.pi
        # input = jax.numpy.hstack([jax.numpy.cos(frequencies * time), jax.numpy.sin(frequencies * time), sample])
        input = jax.numpy.hstack([time, sample])
        return mlp(input)

    vector_field = hk.without_apply_rng(hk.transform(mlp_vector_field))
    one_init_position = jax.tree_map(lambda p: p[0], dist.init_params)
    vector_field_params = vector_field.init(key_init, 0.0, jax.numpy.zeros(N_PARAM))
    if prior_gn is not None:
        prior_samples = 1000
        precond_iter = args.preconditon_iter
        vector_field_params = prior_precondition(key_precond, prior_gn, prior_samples, 
            precond_iter, vector_field, vector_field_params, optim)
    
    # print("Algorithm 2")
    # samples, params = algo_2(key_sample, dist.logprob_fn, one_init_position,
    #     n_iter, vector_field, vector_field_params, optim, n_chain, step_size, args.max_iter)

    print("Algorithm 3")
    samples, params = algo_3(key_sample, dist.logprob_fn, dist.init_params,
        n_iter, vector_field, vector_field_params, optim, n_chain, step_size, args.max_iter)
    
    def flow(u, param):
        u, unravel_fn = ravel_pytree(u)
        flow = odeintegrator(lambda u, time: vector_field.apply(param, time, u), u)
        return unravel_fn(flow[-1]), 1
    
    def flow_inv(x, param):
        x, unravel_fn = ravel_pytree(x)
        flow = odeintegrator(lambda x, time: vector_field.apply(param, 1.0 - time, x), x)
        return unravel_fn(flow[-1]), 1
    
    return samples, params, flow, flow_inv


def prior_precondition(rng_key, prior_gn, n_samples, n_iter, vector_field, init_params, optim):
    k_sample, k_optim = jax.random.split(rng_key)
    ks_sample = jax.random.split(k_sample, n_samples)
    prior_samples = jax.vmap(prior_gn)(ks_sample)
    fmx = flow_matching(vector_field.apply, prior_samples, reference_gn=None)
    optim_state = optim.init(init_params)
    def one_iter(carry, key):
        optim_state, params = carry
        loss_value, grads = jax.value_and_grad(fmx.loss, 1)(key, params, n_samples)
        updates, optim_state = optim.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return (optim_state, params), None
    ks_optim = jax.random.split(k_optim, n_iter)
    (_, params), _ = jax.lax.scan(one_iter, (optim_state, init_params), ks_optim)
    return params


def algo_3(rng_key, logprob_fn, init_position, n_iter, vector_field, init_params, optim, n_chain, step_size, optim_iter):
    kernel = mala(logprob_fn, step_size)
    mapped_step = jax.vmap(kernel.step)

    def one_iter(carry, key):
        states, optim_state, params = carry
        k_sample, k_optim = jax.random.split(key)
        ks_sample = jax.random.split(k_sample, n_chain)
        states, infos = mapped_step(ks_sample, states)
        fmx = flow_matching(vector_field.apply, states.position, reference_gn=None)
        def one_optim_iter(carry, key):
            optim_state, params = carry
            loss_value, grads = jax.value_and_grad(fmx.loss, 1)(key, params, n_chain)
            updates, optim_state = optim.update(grads, optim_state, params)
            params = optax.apply_updates(params, updates)
            return (optim_state, params), loss_value
        ks_optim = jax.random.split(k_optim, optim_iter)
        (optim_state, params), loss_value = jax.lax.scan(
            one_optim_iter, (optim_state, params), ks_optim)
        return (states, optim_state, params), (states.position, loss_value)

    tic1 = pd.Timestamp.now()
    rng_key, key_init = jax.random.split(rng_key)
    fmx = flow_matching(vector_field.apply, init_position, reference_gn=None)
    init_position_fm = fmx.sample(key_init, init_params, n_chain)
    init_states = jax.vmap(kernel.init)(init_position_fm)
    keys = jax.random.split(rng_key, n_iter)
    optim_state = optim.init(init_params)
    (_, optim_state, params), (samples, losses) = jax.lax.scan(
        one_iter, (init_states, optim_state, init_params), keys)
    # samples = states.position
    tic2 = pd.Timestamp.now()

    print("Runtime for Algo 3", (tic2 - tic1).total_seconds())
    samples = jax.tree_util.tree_map(lambda s: s.reshape(1, -1), samples)
    return samples, params

def algo_2(rng_key, logprob_fn, init_position, n_iter, vector_field, init_params, optim, n_chain, step_size, optim_iter):
    kernel = mala(logprob_fn, step_size)
    fmx = flow_matching_sampling(vector_field.apply, init_position, kernel.init, kernel.step, reference_gn=None, adjoint_method=True)

    def one_iter(carry, key):
        optim_state, params = carry
        def one_optim_iter(carry, key):
            optim_state, params = carry
            id_print(key)
            grads, loss_value, samples = fmx.loss(key, params, n_chain)
            # grads = jax.tree_util.tree_map(lambda g: g.sum(), grads)
            updates, optim_state = optim.update(grads, optim_state, params)
            params = optax.apply_updates(params, updates)
            return (optim_state, params), (loss_value, samples)
        (optim_state, params), (loss_value, samples) = jax.lax.scan(
            one_optim_iter, (optim_state, params), jax.random.split(key, optim_iter))
        return (optim_state, params), (samples, loss_value)

    tic1 = pd.Timestamp.now()
    keys = jax.random.split(rng_key, n_iter)
    optim_state = optim.init(init_params)
    (optim_state, params), (samples, losses) = jax.lax.scan(
        one_iter, (optim_state, init_params), keys)
    tic2 = pd.Timestamp.now()

    print("Runtime for Algo 2", (tic2 - tic1).total_seconds())
    samples = jax.tree_util.tree_map(lambda s: s.reshape(1, -1), samples)
    return samples, params