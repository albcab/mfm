from functools import partial

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
from blackjax.adaptation.msc import msc
from blackjax.adaptation.msc_mala import msc_mala
from mcmc_utils import inference_loop0
from fmx.data import flow_matching
from fmx.sampling import flow_matching as flow_matching_sampling

from cont_flows import mlp_flow, resnet_flow, ot_flow

cflows = {
    'mlp': mlp_flow,
    'resnet': resnet_flow,
    'ot': ot_flow,
}


def run(dist, args, N_PARAM, optim, target_gn=None):
    [n_warm, n_iter] = args.sampling_iter
    n_chain = args.num_chain
    key_precond, key_sample, key_init = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    dist.initialize_model(key_init, n_chain)

    odeintegrator = lambda func, x0: odeint(func, x0, jnp.linspace(0.0, 1.0, 2), rtol=1e-5, atol=1e-5, mxstep=1000)
    vector_field_params, vector_field_state, apply_fn, vector_field_with_jacobian_trace = cflows[args.cont_flow](key_init, args, N_PARAM)
    flow_matching_fn = lambda data, weights=None: flow_matching(
        apply_fn, data, weights, odeint=odeintegrator, reference_gn=None, vector_field_with_jacobian_trace=vector_field_with_jacobian_trace)

    if target_gn is not None:
        prior_samples = n_warm
        precond_iter = args.preconditon_iter
        _vector_field_params, _vector_field_state = prior_precondition(key_precond, target_gn, prior_samples, 
            precond_iter, flow_matching_fn, vector_field_params, vector_field_state, optim)
        # vector_field_params, vector_field_state = _vector_field_params, _vector_field_state
    # mc_samples = batch_iter * batch_size
    # precond_iter = n_warm * args.optim_iter #args.preconditon_iter
    # vi_vector_field_params = variational_inference(key_precond, dist.init_params, dist.logprob_fn,
    #     mc_samples, precond_iter, flow_matching_fn, vector_field_params, optim)
        
    print("Base CNF")
    base_cnf_out = base_cnf(key_sample, dist.logprob_fn, dist.init_params, n_warm, n_iter, 
        flow_matching_fn, vector_field_params, vector_field_state, optim, n_chain, args.step_size, n_iter * args.optim_iter)

    print("Algorithm 3 w/ refresh")
    algo_3_out_r = algo_3(key_sample, dist.logprob_fn, dist.init_params, n_warm, n_iter, 
        flow_matching_fn, vector_field_params, vector_field_state, optim, n_chain, args.sampler_iter, args.step_size, args.optim_iter)

    print("Algorithm 3 mod")
    algo_3_out_mod = algo_3_mod(key_sample, dist.logprob_fn, dist.init_params, n_warm, n_iter, 
        flow_matching_fn, vector_field_params, vector_field_state, optim, n_chain, args.sampler_iter, args.step_size, args.optim_iter)
    
    # print("Algorithm 3 w/o refresh")
    # algo_3_out_nr = algo_3(key_sample, dist.logprob_fn, dist.init_params,
    #     n_warm, n_iter, vector_field, vector_field_params, optim, n_chain, args.sampler_iter, args.step_size, args.optim_iter, False)
    
    # print("Algorithm 4")
    # algo_4_out_cis = algo_4(key_sample, dist.logprob_fn, dist.init_params, n_warm, n_iter, 
    #     flow_matching_fn, vector_field_params, optim, batch_iter, batch_size, args.optim_iter, args.sampler_iter)
    algo_4_out_cis = None
    algo_4_out_cis = (base_cnf_out[0], _vector_field_params, _vector_field_state)

    # print("Algorithm 3 w/ MALA (refresh)")
    # algo_4_out_mala = algo_4(key_sample, dist.logprob_fn, dist.init_params, n_warm, n_iter, 
    #     flow_matching_fn, vector_field_params, optim, batch_iter, batch_size, args.optim_iter, args.sampler_iter, args.step_size)
    algo_4_out_mala = None

    # print("Algorithm 2 w/ adjoint gradients")
    # algo_2_out_a = algo_2(key_sample, dist.logprob_fn, one_init_position, n_warm, n_iter, 
    #     vector_field, vector_field_params, optim, n_chain, args.sampler_iter, args.step_size, args.optim_iter)
    algo_2_out_a = None

    # print("Algorithm 2 w/ discretize-then-optimize")
    # algo_2_out_dto = algo_2(key_sample, dist.logprob_fn, one_init_position,
    #     n_warm, n_iter, vector_field, vector_field_params, optim, n_chain, args.sampler_iter, args.step_size, args.optim_iter, False)
    # algo_2_out_dto = (algo_3_out_r[0], vi_vector_field_params)
    print()
    
    def flow(u, param, state):
        u, unravel_fn = ravel_pytree(u)
        flow = odeintegrator(lambda u, time: apply_fn(param, state, time * jnp.ones(1), u.reshape(1, -1), is_training=False)[0], u)
        return unravel_fn(flow[-1]), 1
    # def flow(U, param, state):
    #     unravel_fn = ravel_pytree(jax.tree_util.tree_map(lambda u: u[0], U))[1]
    #     U = jax.vmap(lambda u: ravel_pytree(u)[0])(U)
    #     batch_size, _ = U.shape
    #     flow = odeintegrator(lambda u, time: apply_fn(param, state, time * jnp.ones(batch_size), u, is_training=False)[0], U)
    #     return jax.vmap(unravel_fn)(flow[-1]), jnp.ones(batch_size)
    
    def flow_inv(x, param, state):
        x, unravel_fn = ravel_pytree(x)
        flow = odeintegrator(lambda x, time: apply_fn(param, state, (1.0 - time) * jnp.ones(1), x.reshape(1, -1), is_training=False)[0], x)
        return unravel_fn(flow[-1]), 1
    # def flow_inv(X, param, state):    
    #     unravel_fn = ravel_pytree(jax.tree_util.tree_map(lambda x: x[0], X))[1]
    #     X = jax.vmap(lambda x: ravel_pytree(x)[0])(X)
    #     batch_size, _ = X.shape
    #     flow = odeintegrator(lambda x, time: apply_fn(param, state, (1.0 - time) * jnp.ones(batch_size), x, is_training=False)[0], X)
    #     return jax.vmap(unravel_fn)(flow[-1]), jnp.ones(batch_size)
    
    return algo_4_out_cis, algo_3_out_mod, algo_3_out_r, base_cnf_out, (flow, flow_inv)


def prior_precondition(rng_key, prior_gn, n_samples, n_iter, flow_matching_fn, init_params, init_state, optim):
    tic1 = pd.Timestamp.now()
    k_sample, k_optim = jax.random.split(rng_key)
    ks_sample = jax.random.split(k_sample, n_samples)
    prior_samples = jax.vmap(prior_gn)(ks_sample)
    fmx = flow_matching_fn(prior_samples)
    optim_state = optim.init(init_params)
    def one_iter(carry, key):
        optim_state, params, state = carry
        (loss_value, state), grads = jax.value_and_grad(fmx.loss, 1, has_aux=True)(key, params, state, n_samples, is_training=True)
        updates, optim_state = optim.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return (optim_state, params, state), None
    ks_optim = jax.random.split(k_optim, n_iter)
    (_, params, state), _ = jax.lax.scan(one_iter, (optim_state, init_params, init_state), ks_optim)
    tic2 = pd.Timestamp.now()
    print("Runtime for prior precondition", (tic2 - tic1).total_seconds())
    return params, state


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
    init_params, init_state, optim, n_chain, sample_iter, step_size, optim_iter, refresh=True):
    kernel = mala(logprob_fn, step_size)
    mapped_step = jax.vmap(kernel.step)
    mapped_init = jax.vmap(kernel.init)
    fmx_init = flow_matching_fn(init_position)

    def one_iter(carry, key):
        states, optim_state, params, state = carry
        k_sample, k_optim = jax.random.split(key)
        if refresh:
            positions = fmx_init.sample(k_sample, params, state, n_chain, is_training=False)
            states = mapped_init(positions)
        def one_sample_iter(states, k_sample):
            ks_sample = jax.random.split(k_sample, n_chain)
            states, infos = mapped_step(ks_sample, states)
            return states, (states, infos)
        keys_sample = jax.random.split(k_sample, sample_iter)
        states, (sstates, infos) = jax.lax.scan(one_sample_iter, states, keys_sample)
        samples = jax.tree_util.tree_map(lambda s: s.reshape((n_chain * sample_iter,) + s.shape[2:]), sstates.position)
        fmx = flow_matching_fn(samples)
        def one_optim_iter(carry, key):
            optim_state, params, state = carry
            (loss_value, state), grads = jax.value_and_grad(fmx.loss, 1, has_aux=True)(key, params, state, n_chain, is_training=True)
            updates, optim_state = optim.update(grads, optim_state, params)
            params = optax.apply_updates(params, updates)
            return (optim_state, params, state), loss_value
        ks_optim = jax.random.split(k_optim, optim_iter)
        (optim_state, params, state), loss_value = jax.lax.scan(
            one_optim_iter, (optim_state, params, state), ks_optim)
        return (states, optim_state, params, state), (states.position, loss_value, infos.acceptance_rate.mean())

    tic1 = pd.Timestamp.now()
    key_init, key_optim, key_sample = jax.random.split(rng_key, 3)
    init_position_fm = fmx_init.sample(key_init, init_params, init_state, n_chain, is_training=False)
    init_states = jax.vmap(kernel.init)(init_position_fm)
    keys = jax.random.split(key_optim, n_warm)
    optim_state = optim.init(init_params)
    (_, optim_state, params, state), (samples, losses, acc) = jax.lax.scan(
        one_iter, (init_states, optim_state, init_params, init_state), keys)
    # samples = states.position
    key_init, key_sample = jax.random.split(key_sample)
    samples = fmx_init.sample(key_init, params, state, n_chain, is_training=False)
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
    return samples, params, state


def algo_4(rng_key, logprob_fn, init_position, n_warm, n_iter, flow_matching_fn,
    init_param, optim, batch_iter, batch_size, maxiter, num_samples, step_size=None):
    fmx = flow_matching_fn(init_position)
    flow = fmx.transform_and_logdet
    def get_fm_loss(positions):
        fmx = flow_matching_fn(positions)
        return lambda param, key: fmx.loss(key, param, batch_iter * batch_size)

    tic1 = pd.Timestamp.now()
    k_warm, k_sample = jax.random.split(rng_key)
    if step_size is not None:
        warmup = msc_mala(logprob_fn, optim, init_param, flow, None, batch_iter, batch_size, 
            step_size, n_warm, maxiter, num_mala_samples=num_samples, get_loss=get_fm_loss)
    else:
        warmup = msc(logprob_fn, optim, init_param, flow, None, batch_iter, batch_size, 
            n_warm, maxiter, num_importance_samples=num_samples, get_loss=get_fm_loss)
    chain_state, kernel, param, info = warmup.run(k_warm, init_position)
    init_state = chain_state.states
    def one_chain(k_sam, init_state):
        state, info = inference_loop0(k_sam, init_state, kernel, n_iter)
        return state.position, info
    k_sample = jax.random.split(k_sample, batch_iter * batch_size)
    samples, infos = jax.vmap(one_chain)(k_sample, init_state)
    tic2 = pd.Timestamp.now()

    sec = (tic2 - tic1).total_seconds()
    # do_summary(samples, logprob_fn, sec)
    if step_size is None:
        weights = info.weights.reshape(batch_iter * batch_size * n_warm, num_samples + 1)
        weights /= weights.sum(axis=1).reshape(-1, 1)
        print("Average weights warm=", weights.mean(axis=0))
        weights = infos.weights.reshape(batch_iter * batch_size * n_iter, num_samples + 1)
        weights /= weights.sum(axis=1).reshape(-1, 1)
        print("Average weights iter=", weights.mean(axis=0))
        print("Runtime for MSC CIS", (tic2 - tic1).total_seconds())
    else:
        print("Average and std acceptance rates warm=", info.acceptance_rate.mean(), info.acceptance_rate.std())
        print("Average and std acceptance rates iter=", infos.acceptance_rate.mean(), infos.acceptance_rate.std())
        print("Runtime for MSC MALA", (tic2 - tic1).total_seconds())
    return samples, param


def base_cnf(rng_key, logprob_fn, init_position, n_warm, n_iter, flow_matching_fn, 
    init_params, init_state, optim, n_chain, step_size, optim_iter):
    kernel = mala(logprob_fn, step_size)
    mapped_step = jax.vmap(kernel.step)
    mapped_init = jax.vmap(kernel.init)
    fmx_init = flow_matching_fn(init_position)

    tic1 = pd.Timestamp.now()
    key_init, key_warm, key_sample, key_optim, key_resample = jax.random.split(rng_key, 5)
    init_position_fm = fmx_init.sample(key_init, init_params, init_state, n_chain, is_training=False)
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
        optim_state, params, state = carry
        (loss_value, state), grads = jax.value_and_grad(fmx.loss, 1, has_aux=True)(key, params, state, n_chain, is_training=True)
        updates, optim_state = optim.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return (optim_state, params, state), loss_value
    keys_optim = jax.random.split(key_optim, optim_iter)
    optim_state = optim.init(init_params)
    (optim_state, params, state), loss_value = jax.lax.scan(
        one_optim_iter, (optim_state, init_params, init_state), keys_optim)
    key_init, key_sample = jax.random.split(key_resample)
    samples = fmx.sample(key_init, params, state, n_chain, is_training=False)
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
    return samples, params, state


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


def algo_3_mod(rng_key, logprob_fn, init_position, n_warm, n_iter, flow_matching_fn, 
    init_params, init_state, optim, n_chain, sample_iter, step_size, optim_iter):
    kernel = mala(logprob_fn, step_size)
    fmx_init = flow_matching_fn(init_position)

    def one_iter(carry, key):
        _, optim_state, params, state = carry
        k_sample, k_optim = jax.random.split(key)
        positions = fmx_init.sample(k_sample, params, state, 1, is_training=False)
        position = jax.tree_util.tree_map(lambda p: p[0], positions)
        kernel_state = kernel.init(position)
        def one_sample_iter(state, k_sample):
            state, info = kernel.step(k_sample, state)
            return state, (info.proposed_position, info.proposed_weight, info)
        keys_sample = jax.random.split(k_sample, sample_iter * n_chain)
        _, (positions, weights, infos) = jax.lax.scan(one_sample_iter, kernel_state, keys_sample)
        fmx = flow_matching_fn(positions, weights)
        def one_optim_iter(carry, key):
            optim_state, params, state = carry
            (loss_value, state), grads = jax.value_and_grad(fmx.loss, 1, has_aux=True)(key, params, state, n_chain, is_training=True)
            updates, optim_state = optim.update(grads, optim_state, params)
            params = optax.apply_updates(params, updates)
            return (optim_state, params, state), loss_value
        ks_optim = jax.random.split(k_optim, optim_iter)
        (optim_state, params, state), loss_value = jax.lax.scan(
            one_optim_iter, (optim_state, params, state), ks_optim)
        return (kernel_state, optim_state, params, state), (positions, loss_value, infos.acceptance_rate.mean())

    tic1 = pd.Timestamp.now()
    key_init, key_optim, key_sample = jax.random.split(rng_key, 3)
    keys = jax.random.split(key_optim, n_warm)
    optim_state = optim.init(init_params)
    init_kenel_state = kernel.init(jax.tree_util.tree_map(lambda p: p[0], init_position))
    (_, optim_state, params, state), (samples, losses, acc) = jax.lax.scan(
        one_iter, (init_kenel_state, optim_state, init_params, init_state), keys)
    # samples = states.position
    key_init, key_sample = jax.random.split(key_sample)
    samples = fmx_init.sample(key_init, params, state, n_chain, is_training=False)
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
    print("Runtime for Algo 3", (tic2 - tic1).total_seconds())
    return samples, params, state
