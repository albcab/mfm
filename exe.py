
import jax

from distances import kullback_liebler
from execute import run_precondition, run_tess


def taylor_run(dist, args, flow, optim, N_PARAM, batch_fn=jax.vmap):

    [n_warm, n_iter] = args.sampling_param
    [batch_iter, batch_size] = args.batch_shape
    kflow, ksam, kinit = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    dist.initialize_model(kinit, batch_iter * batch_size)

    init_param, flow, reverse, forward = initialize_flow(
        kflow, N_PARAM, dist.logprob_fn, flow)

    one_init_param = jax.tree_map(lambda p: p[0], dist.init_params)
    mc_samples = 1000
    precond_iter = args.preconditon_iter
    precond_param = run_precondition(kflow, init_param, one_init_param, 
        optim, reverse, mc_samples, precond_iter)

    print("TESS w/ precond.")
    samples, param = run_tess(ksam, dist.logprob_fn, dist.init_params,
        n_warm, n_iter, precond_param, optim, flow, forward, 
        batch_iter, batch_size, args.max_iter, batch_fn)


def initialize_flow(rng_key, d, logprob_fn, flow, distance=kullback_liebler):

    param_init, flow, flow_inv = flow.get_utilities()
    reverse, forward = distance(logprob_fn, flow, flow_inv)
    init_param = param_init(rng_key, jax.random.normal(rng_key, shape=(d,)))

    return init_param, flow, reverse, forward
