from functools import partial

import pandas as pd

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.experimental.ode import odeint
from jax.experimental.host_callback import id_print

import optax
import haiku as hk

import wandb

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
    key_target, key_sample, key_init = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    dist.initialize_model(key_init, n_chain)

    odeintegrator = lambda func, x0: odeint(func, x0, jnp.linspace(0.0, 1.0, 2), rtol=1e-5, atol=1e-5, mxstep=1000)
    vector_field_params, vector_field_state, apply_fn, vector_field_with_jacobian_trace = cflows[args.cont_flow](
        key_init, args, N_PARAM, jax.grad(lambda input: dist.logprob(input[0], input[1:])))
    flow_matching_fn = lambda data, weights=None: flow_matching(
        apply_fn, data, weights, odeint=odeintegrator, reference_gn=None, vector_field_with_jacobian_trace=vector_field_with_jacobian_trace)

    if target_gn is not None:
        key_gen, key_loss = jax.random.split(key_target)
        keys_target = jax.random.split(key_gen, n_iter)
        real_samples = jax.vmap(target_gn)(keys_target)
        fmx = flow_matching_fn(jnp.expand_dims(real_samples, 1))
        check_target = lambda params, state: fmx.loss(key_loss, params, state, is_training=False)[0]
        
        vi_out = prior_precondition(key_sample, target_gn, n_chain, args.sampler_iter, 
            n_warm, args.optim_iter, flow_matching_fn, vector_field_params, vector_field_state, optim, check_target)
        # vector_field_params, vector_field_state = vi_out[:2]
    else:
        vi_out = (vector_field_params, vector_field_state)
        check_target = lambda p, s: jnp.nan

    # mc_samples = batch_iter * batch_size
    # precond_iter = n_warm * args.optim_iter #args.preconditon_iter
    # vi_vector_field_params = variational_inference(key_precond, dist.init_params, dist.logprob_fn,
    #     mc_samples, precond_iter, flow_matching_fn, vector_field_params, optim)
        
    print("Base CNF")
    base_cnf_out = base_cnf(key_sample, dist.logprob_fn, dist.init_params, n_warm * args.sampler_iter, 
        flow_matching_fn, vector_field_params, vector_field_state, optim, n_chain, args.step_size, n_warm, args.optim_iter, check_target)
    
    print("Base CNF mod")
    base_mod_out = base_cnf(key_sample, dist.logprob_fn, dist.init_params, n_warm * args.sampler_iter, 
        flow_matching_fn, vector_field_params, vector_field_state, optim, n_chain, args.step_size, n_warm, args.optim_iter, check_target, True)

    print("Algorithm 3 w/ refresh")
    algo_3_out_r = algo_3(key_sample, dist.logprob_fn, dist.init_params, n_warm, 
        flow_matching_fn, vector_field_params, vector_field_state, optim, n_chain, args.sampler_iter, args.step_size, args.optim_iter, check_target)

    print("Algorithm 3 mod")
    algo_3_out_mod = algo_3(key_sample, dist.logprob_fn, dist.init_params, n_warm, 
        flow_matching_fn, vector_field_params, vector_field_state, optim, n_chain, args.sampler_iter, args.step_size, args.optim_iter, check_target, True)
    
    # print("Algorithm 3 w/o refresh")
    # algo_3_out_nr = algo_3(key_sample, dist.logprob_fn, dist.init_params,
    #     n_warm, n_iter, vector_field, vector_field_params, optim, n_chain, args.sampler_iter, args.step_size, args.optim_iter, False)
    
    # print("Algorithm 4")
    # algo_4_out_cis = algo_4(key_sample, dist.logprob_fn, dist.init_params, n_warm, n_iter, 
    #     flow_matching_fn, vector_field_params, optim, batch_iter, batch_size, args.optim_iter, args.sampler_iter)

    # print("Algorithm 3 w/ MALA (refresh)")
    # algo_4_out_mala = algo_4(key_sample, dist.logprob_fn, dist.init_params, n_warm, n_iter, 
    #     flow_matching_fn, vector_field_params, optim, batch_iter, batch_size, args.optim_iter, args.sampler_iter, args.step_size)

    # print("Algorithm 2 w/ adjoint gradients")
    # algo_2_out_a = algo_2(key_sample, dist.logprob_fn, one_init_position, n_warm, n_iter, 
    #     vector_field, vector_field_params, optim, n_chain, args.sampler_iter, args.step_size, args.optim_iter)

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
    
    return vi_out, algo_3_out_mod, algo_3_out_r, base_cnf_out, base_mod_out, (flow, flow_inv)


def prior_precondition(rng_key, prior_gn, batch_size, n_samples, epochs, n_iter, flow_matching_fn, init_params, init_state, optim, check_target):

    @jax.vmap
    def batch_sample(k_sample):
        ks_sample = jax.random.split(k_sample, n_samples)
        return jax.vmap(prior_gn)(ks_sample)
    def one_epoch(carry, key):
        k_sample, k_optim = jax.random.split(key)
        keys_sample = jax.random.split(k_sample, batch_size)
        prior_samples = batch_sample(keys_sample)
        fmx = flow_matching_fn(prior_samples)
        def one_iter(carry, key):
            optim_state, params, state = carry
            (loss_value, state), grads = jax.value_and_grad(fmx.loss, 1, has_aux=True)(key, params, state, is_training=True)
            updates, optim_state = optim.update(grads, optim_state, params)
            params = optax.apply_updates(params, updates)
            return (optim_state, params, state), loss_value
        ks_optim = jax.random.split(k_optim, n_iter)
        new_carry, losses = jax.lax.scan(one_iter, carry, ks_optim)
        target_losses = check_target(*new_carry[1:])
        return new_carry, (target_losses, losses)
    
    tic1 = pd.Timestamp.now()
    optim_state = optim.init(init_params)
    keys_optim = jax.random.split(rng_key, epochs)
    (_, params, state), (target_losses, losses) = jax.lax.scan(one_epoch, (optim_state, init_params, init_state), keys_optim)
    tic2 = pd.Timestamp.now()

    sec = (tic2 - tic1).total_seconds()
    print("Runtime for prior precondition", sec)
    targ, lss = [], []
    for i, (tl, ls) in enumerate(zip(target_losses, losses)):
        targ.append([i, tl])
        lss.append([i, ls.mean()])
    targ_cols = ["epoch", "target loss"]
    targ_name = 'target loss/FM real samples'
    targ_table = wandb.Table(targ_cols, targ)
    wandb.log({targ_name: wandb.plot.line(targ_table, *targ_cols, title=targ_name)})
    lss_cols = ["epoch", "avg loss"]
    lss_name = 'loss/FM real samples'
    lss_table = wandb.Table(lss_cols, lss)
    wandb.log({lss_name: wandb.plot.line(lss_table, *lss_cols, title=lss_name)})
    return params, state, (sec, None, None)


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


def sampling_loop(key, init_states, step, n_chain, sample_iter):
    def one_chain_iter(k_sample, states):
        def one_step(states, key):
            states, infos = step(key, states)
            return states, (states, infos)
        ks_sample = jax.random.split(k_sample, sample_iter)
        return jax.lax.scan(one_step, states, ks_sample)
    keys_sample = jax.random.split(key, n_chain)
    return jax.vmap(one_chain_iter)(keys_sample, init_states)


def algo_3(rng_key, logprob_fn, init_position, n_warm, flow_matching_fn, 
    init_params, init_state, optim, n_chain, sample_iter, step_size, optim_iter, check_target, mod=False):
    kernel = mala(logprob_fn, step_size)
    mapped_init = jax.vmap(kernel.init)
    fmx_init = flow_matching_fn(jax.tree_util.tree_map(lambda p: jnp.expand_dims(p, 0), init_position))

    def one_iter(carry, key):
        _, optim_state, params, state = carry
        k_init, k_sample, k_optim = jax.random.split(key, 3)
        positions = fmx_init.sample(k_init, params, state, n_chain, is_training=False)
        states = mapped_init(positions)
        states, (sstates, infos) = sampling_loop(k_sample, states, kernel.step, n_chain, sample_iter)
        if mod:
            fmx = flow_matching_fn(infos.proposed_position, infos.proposed_weight)
        else:
            fmx = flow_matching_fn(sstates.position)
        def one_optim_iter(carry, key):
            optim_state, params, state = carry
            (loss_value, state), grads = jax.value_and_grad(fmx.loss, 1, has_aux=True)(key, params, state, is_training=True)
            updates, optim_state = optim.update(grads, optim_state, params)
            params = optax.apply_updates(params, updates)
            return (optim_state, params, state), loss_value
        ks_optim = jax.random.split(k_optim, optim_iter)
        (optim_state, params, state), loss_value = jax.lax.scan(
            one_optim_iter, (optim_state, params, state), ks_optim)
        target_loss = check_target(params, state)
        return (states, optim_state, params, state), (target_loss, loss_value, infos.acceptance_rate)

    tic1 = pd.Timestamp.now()
    init_states = mapped_init(init_position)
    keys = jax.random.split(rng_key, n_warm)
    optim_state = optim.init(init_params)
    (_, optim_state, params, state), (target_losses, losses, acc) = jax.lax.scan(
        one_iter, (init_states, optim_state, init_params, init_state), keys)
    tic2 = pd.Timestamp.now()

    sec = (tic2 - tic1).total_seconds()
    print("Average and std acceptance rates warm=", acc.mean(), acc.std())
    print("Runtime for Algo 3", sec)
    targ, lss = [], []
    for i, (tl, ls) in enumerate(zip(target_losses, losses)):
        targ.append([i, tl])
        lss.append([i, ls.mean()])
    targ_cols = ["epoch", "target loss"]
    targ_name = 'target loss/Algo 3' + (' mod' if mod else '')
    targ_table = wandb.Table(targ_cols, targ)
    wandb.log({targ_name: wandb.plot.line(targ_table, *targ_cols, title=targ_name)})
    lss_cols = ["epoch", "avg loss"]
    lss_name = 'loss/Algo 3' + (' mod' if mod else '')
    lss_table = wandb.Table(lss_cols, lss)
    wandb.log({lss_name: wandb.plot.line(lss_table, *lss_cols, title=lss_name)})
    return params, state, (sec, acc.mean(), acc.std())


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


def base_cnf(rng_key, logprob_fn, init_position, n_warm, flow_matching_fn, 
    init_params, init_state, optim, n_chain, step_size, epochs, optim_iter, check_target, mod=False):
    kernel = mala(logprob_fn, step_size)

    tic1 = pd.Timestamp.now()
    key_sample, key_optim = jax.random.split(rng_key)
    init_states = jax.vmap(kernel.init)(init_position)
    # states, _ = sampling_loop(key_warm, init_states, kernel.step, n_chain, n_warm)
    states = init_states
    _, (states, infos) = sampling_loop(key_sample, states, kernel.step, n_chain, n_warm)
    if mod:
        fmx = flow_matching_fn(infos.proposed_position, infos.proposed_weight)
    else:
        fmx = flow_matching_fn(states.position)
    acc = infos.acceptance_rate
    def one_epoch(carry, key):
        def one_iter(carry, key):
            optim_state, params, state = carry
            (loss_value, state), grads = jax.value_and_grad(fmx.loss, 1, has_aux=True)(key, params, state, is_training=True)
            updates, optim_state = optim.update(grads, optim_state, params)
            params = optax.apply_updates(params, updates)
            return (optim_state, params, state), loss_value
        ks_optim = jax.random.split(key, optim_iter)
        new_carry, losses = jax.lax.scan(one_iter, carry, ks_optim)
        target_losses = check_target(*new_carry[1:])
        return new_carry, (target_losses, losses)
    optim_state = optim.init(init_params)
    keys_optim = jax.random.split(key_optim, epochs)
    (_, params, state), (target_losses, losses) = jax.lax.scan(one_epoch, (optim_state, init_params, init_state), keys_optim)
    tic2 = pd.Timestamp.now()

    sec = (tic2 - tic1).total_seconds()
    print("Average and std acceptance rates warm=", acc.mean(), acc.std())
    print("Runtime for Base CNF", sec)
    targ, lss = [], []
    for i, (tl, ls) in enumerate(zip(target_losses, losses)):
        targ.append([i, tl])
        lss.append([i, ls.mean()])
    targ_cols = ["epoch", "target loss"]
    targ_name = 'target loss/Base 6\.2' + (' mod ' if mod else '')
    targ_table = wandb.Table(targ_cols, targ)
    wandb.log({targ_name: wandb.plot.line(targ_table, *targ_cols, title=targ_name)})
    lss_cols = ["epoch", "avg loss"]
    lss_name = 'loss/Base 6\.2' + (' mod' if mod else '')
    lss_table = wandb.Table(lss_cols, lss)
    wandb.log({lss_name: wandb.plot.line(lss_table, *lss_cols, title=lss_name)})
    return params, state, (sec, acc.mean(), acc.std())


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

