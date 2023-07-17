import pandas as pd

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.experimental.ode import odeint
from jax.experimental.host_callback import id_print

import optax
import haiku as hk

from execute import do_summary

from blackjax.mcmc.mala import mala
from fmx.data import flow_matching
from fmx.sampling import flow_matching as flow_matching_sampling

scale_W = 0.1
init_W = hk.initializers.VarianceScaling(scale_W)
scale_b = 0.01
init_b = hk.initializers.RandomNormal(scale_b)


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


def run(dist, args, N_PARAM, optim, ot_flow=False, prior_gn=None):
    [n_warm, n_iter] = args.sampling_param
    [batch_iter, batch_size] = args.batch_shape
    n_chain = batch_iter * batch_size
    key_precond, key_sample, key_init = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    dist.initialize_model(key_init, n_chain)
    non_lin = non_lins[args.non_linearity]

    odeintegrator = lambda func, x0: odeint(func, x0, jnp.linspace(0.0, 1.0, 11))

    if ot_flow:
        step_size = 1.
        def apply_fn(param, time, x):
            s = jnp.hstack([x, time])
            linear0 = param['K0'] @ s + param['b0']
            u0 = jnp.log(jnp.exp(linear0) + jnp.exp(-linear0))
            layer1 = jax.nn.tanh(param['K1'] @ u0 + param['b1'])
            z1 = param['w'] + step_size * param['K1'].T * layer1 @ param['w']
            layer0 = jax.nn.tanh(linear0)
            grad_resnet = param['K0'].T * layer0 @ z1
            grad_potential = grad_resnet + param['A'].T @ param['A'] @ s + param['b']
            return -grad_potential[:N_PARAM]

        m = args.n_hidden[0]
        stddev = jnp.sqrt(scale_W) / .87962566103423978
        variance_scaling = lambda k, shape: stddev / shape[0] * jax.random.truncated_normal(k, -2., 2., shape)
        random_normal = lambda k, shape: scale_b * jax.random.normal(k, shape)
        kw, kA, kb, kK0, kK1, kb0, kb1 = jax.random.split(key_init, 7)
        vector_field_params = {
            "w": variance_scaling(kw, (m,)),
            "A": variance_scaling(kA, (min(10, N_PARAM), N_PARAM + 1)),
            "b": random_normal(kb, (N_PARAM + 1,)),
            "K0": variance_scaling(kK0, (m, N_PARAM + 1)),
            "K1": variance_scaling(kK1, (m, m)),
            "b0": random_normal(kb0, (m,)),
            "b1": random_normal(kb1, (m,)),
        }

        grad_tanh = lambda x: 1.0 - jnp.square(jax.nn.tanh(x))
        def vector_field_with_jacobian_trace(param, time, x):
            s = jnp.hstack([x, time])
            linear0 = param['K0'] @ s + param['b0']
            u0 = jnp.log(jnp.exp(linear0) + jnp.exp(-linear0))
            linear1 = param['K1'] @ u0 + param['b1']
            layer1 = jax.nn.tanh(linear1)
            z1 = param['w'] + step_size * param['K1'].T * layer1 @ param['w']
            layer0 = jax.nn.tanh(linear0)
            grad_resnet = param['K0'].T * layer0 @ z1
            grad_potential = grad_resnet + param['A'].T @ param['A'] @ s + param['b']

            reduK0 = param['K0'][:, :N_PARAM]
            gradu0 = jax.nn.tanh(linear0).reshape(m, 1) * reduK0
            t0 = jnp.sum((grad_tanh(linear0) * z1) @ (reduK0 * reduK0))
            reduJ = param['K1'] @ gradu0
            t1 = jnp.sum((grad_tanh(linear1) * param['w']) @ (reduJ * reduJ))
            reduA = param['A'][:, :N_PARAM]
            traceAtA = jnp.sum(jnp.square(reduA))
            return -grad_potential[:N_PARAM], -(t0 + step_size * t1 + traceAtA)
        
        flow_matching_fn = lambda data: flow_matching(apply_fn, data, reference_gn=None, vector_field_with_jacobian_trace=vector_field_with_jacobian_trace)

    else:
        def mlp_vector_field(time, sample):  # , frequencies=3):
            (out_dim,) = sample.shape
            mlp = mlp_generator(out_dim, hidden_dims=args.n_hidden, non_linearity=non_lin)
            # frequencies = 2 ** jnp.arange(frequencies) * jnp.pi
            # input = jnp.hstack([jnp.cos(frequencies * time), jnp.sin(frequencies * time), sample])
            input = jnp.hstack([time, sample])
            return mlp(input)

        vector_field = hk.without_apply_rng(hk.transform(mlp_vector_field))
        vector_field_params = vector_field.init(key_init, 0.0, jnp.zeros(N_PARAM))
        flow_matching_fn = lambda data: flow_matching(vector_field.apply, data, reference_gn=None, vector_field_with_jacobian_trace=None)
        apply_fn = vector_field.apply

    if prior_gn is not None:
        prior_samples = 1000
        precond_iter = args.preconditon_iter
        vector_field_params = prior_precondition(key_precond, prior_gn, prior_samples, 
            precond_iter, flow_matching_fn, vector_field_params, optim)
    mc_samples = batch_iter * batch_size
    precond_iter = n_warm * args.max_iter #args.preconditon_iter
    vi_vector_field_params = variational_inference(key_precond, dist.init_params, dist.logprob_fn,
        mc_samples, precond_iter, flow_matching_fn, vector_field_params, optim)
        
    print("Base CNF")
    base_cnf_out = base_cnf(key_sample, dist.logprob_fn, dist.init_params, n_warm, n_iter, 
        flow_matching_fn, vector_field_params, optim, n_chain, args.step_size[3], n_iter * args.max_iter)
    
    print("Algorithm 3 w/ refresh")
    algo_3_out_r = algo_3(key_sample, dist.logprob_fn, dist.init_params, n_warm, n_iter, 
        flow_matching_fn, vector_field_params, optim, n_chain, args.sample_iter, args.step_size[2], args.max_iter)
    
    # print("Algorithm 3 w/o refresh")
    # algo_3_out_nr = algo_3(key_sample, dist.logprob_fn, dist.init_params,
    #     n_warm, n_iter, vector_field, vector_field_params, optim, n_chain, args.sample_iter, args.step_size[1], args.max_iter, False)
    
    # print("Algorithm 2 w/ adjoint gradients")
    # algo_2_out_a = algo_2(key_sample, dist.logprob_fn, one_init_position, n_warm, n_iter, 
    #     vector_field, vector_field_params, optim, n_chain, args.sample_iter, args.step_size[0], args.max_iter)
    algo_2_out_a = None

    # print("Algorithm 2 w/ discretize-then-optimize")
    # algo_2_out_dto = algo_2(key_sample, dist.logprob_fn, one_init_position,
    #     n_warm, n_iter, vector_field, vector_field_params, optim, n_chain, args.sample_iter, args.step_size[0], args.max_iter, False)
    algo_2_out_dto = (algo_3_out_r[0], vi_vector_field_params)
    print()
    
    def flow(u, param):
        u, unravel_fn = ravel_pytree(u)
        flow = odeintegrator(lambda u, time: apply_fn(param, time, u), u)
        return unravel_fn(flow[-1]), 1
    
    def flow_inv(x, param):
        x, unravel_fn = ravel_pytree(x)
        flow = odeintegrator(lambda x, time: apply_fn(param, 1.0 - time, x), x)
        return unravel_fn(flow[-1]), 1
    
    return algo_2_out_a, algo_2_out_dto, algo_3_out_r, base_cnf_out, (flow, flow_inv)


def prior_precondition(rng_key, prior_gn, n_samples, n_iter, flow_matching_fn, init_params, optim):
    tic1 = pd.Timestamp.now()
    k_sample, k_optim = jax.random.split(rng_key)
    ks_sample = jax.random.split(k_sample, n_samples)
    prior_samples = jax.vmap(prior_gn)(ks_sample)
    fmx = flow_matching_fn(prior_samples)
    optim_state = optim.init(init_params)
    def one_iter(carry, key):
        optim_state, params = carry
        loss_value, grads = jax.value_and_grad(fmx.loss, 1)(key, params, n_samples)
        updates, optim_state = optim.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return (optim_state, params), None
    ks_optim = jax.random.split(k_optim, n_iter)
    (_, params), _ = jax.lax.scan(one_iter, (optim_state, init_params), ks_optim)
    tic2 = pd.Timestamp.now()
    print("Runtime for prior precondition", (tic2 - tic1).total_seconds())
    return params


def variational_inference(rng_key, init_position, logprob_fn, n_samples, n_iter, flow_matching_fn, init_params, optim, adjoint=False):
    tic1 = pd.Timestamp.now()
    fmx = flow_matching_fn(init_position)
    one_position = jax.tree_util.tree_map(lambda p: p[0], init_position)
    position, unravel_fn = ravel_pytree(one_position)
    def kld_reverse(param, key):
        U = jax.vmap(unravel_fn)(jax.random.normal(key, (n_samples,) + position.shape))
        X, LDJ = jax.vmap(fmx.transform_and_logdet, (0, None))(U, param)
        return -jnp.sum(jax.vmap(logprob_fn)(X) + LDJ)
    optim_state = optim.init(init_params)
    def one_iter(carry, key):
        optim_state, params = carry
        neglik, grads = jax.value_and_grad(kld_reverse)(params, key)
        updates, optim_state = optim.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return (optim_state, params), neglik
    keys_optim = jax.random.split(rng_key, n_iter)
    (_, params), negliks = jax.lax.scan(one_iter, (optim_state, init_params), keys_optim)
    tic2 = pd.Timestamp.now()
    print("Runtime for reverse KLD training", (tic2 - tic1).total_seconds())
    return params


def algo_3(rng_key, logprob_fn, init_position, n_warm, n_iter, flow_matching_fn, 
    init_params, optim, n_chain, sample_iter, step_size, optim_iter, refresh=True):
    kernel = mala(logprob_fn, step_size)
    mapped_step = jax.vmap(kernel.step)
    mapped_init = jax.vmap(kernel.init)
    fmx_init = flow_matching_fn(init_position)

    def one_iter(carry, key):
        states, optim_state, params = carry
        k_sample, k_optim = jax.random.split(key)
        if refresh:
            positions = fmx_init.sample(k_sample, params, n_chain)
            states = mapped_init(positions)
        def one_sample_iter(states, k_sample):
            ks_sample = jax.random.split(k_sample, n_chain)
            states, infos = mapped_step(ks_sample, states)
            return states, infos
        keys_sample = jax.random.split(k_sample, sample_iter)
        states, infos = jax.lax.scan(one_sample_iter, states, keys_sample)
        fmx = flow_matching_fn(states.position)
        def one_optim_iter(carry, key):
            optim_state, params = carry
            loss_value, grads = jax.value_and_grad(fmx.loss, 1)(key, params, n_chain)
            updates, optim_state = optim.update(grads, optim_state, params)
            params = optax.apply_updates(params, updates)
            return (optim_state, params), loss_value
        ks_optim = jax.random.split(k_optim, optim_iter)
        (optim_state, params), loss_value = jax.lax.scan(
            one_optim_iter, (optim_state, params), ks_optim)
        return (states, optim_state, params), (states.position, loss_value, infos.acceptance_rate.mean())

    tic1 = pd.Timestamp.now()
    key_init, key_optim, key_sample = jax.random.split(rng_key, 3)
    init_position_fm = fmx_init.sample(key_init, init_params, n_chain)
    init_states = jax.vmap(kernel.init)(init_position_fm)
    keys = jax.random.split(key_optim, n_warm)
    optim_state = optim.init(init_params)
    (_, optim_state, params), (samples, losses, acc) = jax.lax.scan(
        one_iter, (init_states, optim_state, init_params), keys)
    # samples = states.position
    key_init, key_sample = jax.random.split(key_sample)
    samples = fmx_init.sample(key_init, params, n_chain)
    states = mapped_init(samples)
    keys_sample = jax.random.split(key_sample, n_iter)
    def one_sample_iter(states, key):
        keys = jax.random.split(key, n_chain)
        states, infos = mapped_step(keys, states)
        return states, (states, infos)
    _, (states, infos) = jax.lax.scan(one_sample_iter, states, keys_sample)
    samples = states.position
    iter_acc = infos.acceptance_rate
    tic2 = pd.Timestamp.now()

    print("Average and std acceptance rates warm=", acc.mean(), acc.std())
    print("Average and std acceptance rates iter=", iter_acc.mean(), iter_acc.std())
    print("Runtime for Algo 3", (tic2 - tic1).total_seconds())
    return samples, params


def base_cnf(rng_key, logprob_fn, init_position, n_warm, n_iter, flow_matching_fn, 
    init_params, optim, n_chain, step_size, optim_iter):
    kernel = mala(logprob_fn, step_size)
    mapped_step = jax.vmap(kernel.step)
    mapped_init = jax.vmap(kernel.init)
    fmx_init = flow_matching_fn(init_position)

    tic1 = pd.Timestamp.now()
    key_init, key_warm, key_sample, key_optim, key_resample = jax.random.split(rng_key, 5)
    init_position_fm = fmx_init.sample(key_init, init_params, n_chain)
    init_states = mapped_init(init_position_fm)
    def one_sample_iter(states, k_sample):
        ks_sample = jax.random.split(k_sample, n_chain)
        states, infos = mapped_step(ks_sample, states)
        return states, (states, infos)
    keys_warm = jax.random.split(key_warm, n_warm)
    states, _ = jax.lax.scan(one_sample_iter, init_states, keys_warm)
    keys_sample = jax.random.split(key_sample, n_warm)
    _, (states, infos) = jax.lax.scan(one_sample_iter, states, keys_sample)
    samples = jax.tree_util.tree_map(lambda s: s.reshape((n_chain * n_warm,) + s.shape[2:]), states.position)
    fmx = flow_matching_fn(samples)
    acc = infos.acceptance_rate.mean()
    def one_optim_iter(carry, key):
        optim_state, params = carry
        loss_value, grads = jax.value_and_grad(fmx.loss, 1)(key, params, n_chain)
        updates, optim_state = optim.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return (optim_state, params), loss_value
    keys_optim = jax.random.split(key_optim, optim_iter)
    optim_state = optim.init(init_params)
    (optim_state, params), loss_value = jax.lax.scan(
        one_optim_iter, (optim_state, init_params), keys_optim)
    key_init, key_sample = jax.random.split(key_resample)
    samples = fmx.sample(key_init, params, n_chain)
    states = mapped_init(samples)
    keys_sample = jax.random.split(key_sample, n_iter)
    def one_sample_iter(states, key):
        keys = jax.random.split(key, n_chain)
        states, infos = mapped_step(keys, states)
        return states, (states, infos)
    _, (states, infos) = jax.lax.scan(one_sample_iter, states, keys_sample)
    samples = states.position
    iter_acc = infos.acceptance_rate
    tic2 = pd.Timestamp.now()

    print("Average and std acceptance rates warm=", acc.mean(), acc.std())
    print("Average and std acceptance rates iter=", iter_acc.mean(), iter_acc.std())
    print("Runtime for Base CNF", (tic2 - tic1).total_seconds())
    return samples, params


def algo_2(rng_key, logprob_fn, init_position, n_warm, n_iter, vector_field, 
    init_params, optim, n_chain, sample_iter, step_size, optim_iter, adjoint=True):
    kernel = mala(logprob_fn, step_size)
    fmx = flow_matching_sampling(vector_field.apply, init_position, kernel.init, kernel.step, 
        sample_iter, reference_gn=None, adjoint_method=adjoint)

    def one_iter(carry, key):
        optim_state, params = carry
        def one_optim_iter(carry, key):
            optim_state, params = carry
            grads, loss_value, samples, infos = fmx.loss(key, params, n_chain)
            updates, optim_state = optim.update(grads, optim_state, params)
            params = optax.apply_updates(params, updates)
            return (optim_state, params), (loss_value, samples, infos.acceptance_rate.mean())
        (optim_state, params), (loss_value, samples, acc) = jax.lax.scan(
            one_optim_iter, (optim_state, params), jax.random.split(key, optim_iter))
        return (optim_state, params), (samples, loss_value, acc.mean())

    tic1 = pd.Timestamp.now()
    key_optim, key_sample = jax.random.split(rng_key)
    keys = jax.random.split(key_optim, n_warm)
    optim_state = optim.init(init_params)
    (optim_state, params), (samples, losses, acc) = jax.lax.scan(
        one_iter, (optim_state, init_params), keys)
    key_init, key_sample = jax.random.split(key_sample)
    samples = fmx.sample(key_init, params, n_chain)
    states = jax.vmap(kernel.init)(samples)
    keys_sample = jax.random.split(key_sample, n_iter)
    def one_sample_iter(states, key):
        keys = jax.random.split(key, n_chain)
        states, infos = jax.vmap(kernel.step)(keys, states)
        return states, (states, infos)
    _, (states, infos) = jax.lax.scan(one_sample_iter, states, keys_sample)
    samples = states.position
    iter_acc = infos.acceptance_rate
    tic2 = pd.Timestamp.now()

    print("Average and std acceptance rates warm=", acc.mean(), acc.std())
    print("Average and std acceptance rates iter=", iter_acc.mean(), iter_acc.std())
    print("Runtime for Algo 2", (tic2 - tic1).total_seconds())
    return samples, params