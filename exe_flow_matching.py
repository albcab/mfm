import logging
import time
from typing import Callable, List

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

import flax.linen as nn
from flax import struct, traverse_util
from flax.training import train_state

from jax.experimental.ode import odeint
from diffrax import Tsit5, Dopri5, Heun, Kvaerno3, Kvaerno4, Kvaerno5
from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController

import optax

# from ott.geometry import costs, pointcloud
# from ott.problems.linear import linear_problem
# from ott.solvers.linear import sinkhorn

from jaxopt import Bisection

import wandb
from tqdm import tqdm

from bblackjax.mcmc.mala import init, build_kernel, MALAState, MALAInfo
from mcmc_utils import stein_disc, max_mean_disc

from distributions import GaussianMixture, IndepGaussian, FlatDistribution, PhiFourBase

import matplotlib.pyplot as plt
# import seaborn as sns


logger = logging.getLogger(__name__)


non_lins = {
    'tanh': jax.nn.tanh,
    'elu': jax.nn.elu,
    'relu': jax.nn.relu,
    'gelu': jax.nn.gelu,
    'swish': jax.nn.swish,
}

ref_dists = {
    'stdgauss': lambda dim: IndepGaussian(dim),
    'widegauss': lambda dim: IndepGaussian(dim, var=5.),
    'bimodal': lambda dim: GaussianMixture(dim),
    'flat': lambda dim: FlatDistribution(),
    'phifour': lambda dim: PhiFourBase(dim),
}

class VectorFieldNet(nn.Module):

    fourier_random: jax.Array
    grad_logporob: Callable
    hidden_x: list
    hidden_t: list
    hidden_xt: list
    act_fn: Callable = jax.nn.relu
    grad_clip: float = None

    @nn.compact
    def __call__(self, x, t):
        dim, = x.shape
        #Fourier feature augmentation of time
        degt = 2 * jnp.pi * self.fourier_random * t
        ffat = jnp.concatenate([jnp.cos(degt), jnp.sin(degt)])
        #time signal
        signal_t = ffat
        for h in self.hidden_t:
            signal_t = self.act_fn(nn.Dense(h)(signal_t))
        #x signal
        signal_x = x
        for h in self.hidden_x:
            signal_x = self.act_fn(nn.Dense(h)(signal_x))
        #grad component
        nn_t = nn.Dense(dim, kernel_init=nn.initializers.zeros_init())(signal_t)
        #joint component
        nn_xt = jnp.concatenate([signal_x, signal_t])
        for h in self.hidden_xt:
            nn_xt = self.act_fn(nn.Dense(h)(nn_xt))
        nn_xt = nn.Dense(dim, kernel_init=nn.initializers.zeros_init())(nn_xt)
        if self.grad_clip:
            return nn_xt + nn_t * jnp.clip(self.grad_logporob(x), -self.grad_clip, self.grad_clip)
        else:
            return nn_xt + nn_t * self.grad_logporob(x)


def create_train_state(
    vector_field_apply,
    vector_field_param,
    learning_rate_fn: Callable[[int], float],
    args,
) -> train_state.TrainState:
    """Create initial training state."""

    class TrainState(train_state.TrainState):
        """Train state with an Optax optimizer.

        The two functions below differ depending on whether the task is classification
        or regression.

        Args:
          loss_fn: Function to compute the loss.
        """
        loss_fn: Callable = struct.field(pytree_node=False)

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        # find out all LayerNorm parameters
        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
        layer_norm_named_params = {
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        }
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    tx = optax.adamw(
        learning_rate=learning_rate_fn,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
        mask=decay_mask_fn,
    )
    clipper = optax.clip(args.gradient_clip)

    def flow_fn(rng_key, samples):
        batch_size, n_dim = samples.shape
        key_time, key_ref_sample = jax.random.split(rng_key)
        times = jax.random.uniform(key_time, (batch_size, 1))
        ref_samples = jax.random.normal(key_ref_sample, (batch_size, n_dim))
        sds = 1.0 - (1.0 - args.sigma) * times
        cond_samples = times * samples + sds * ref_samples
        target_vector_fields = samples - (1 - args.sigma) * ref_samples
        return times, cond_samples, target_vector_fields

    ref_dist = ref_dists[args.ref_dist](args.dim)

    def cond_flow_fn(rng_key, samples):
        batch_size, n_dim = samples.shape
        key_time, key_ref_sample, key_gaussian, key_ot = jax.random.split(rng_key, 4)
        times = jax.random.uniform(key_time, (batch_size, 1))
        ref_samples = jax.vmap(ref_dist.sample_model)(jax.random.split(key_ref_sample, batch_size))
        if args.ot_cond_flow:
            geom = pointcloud.PointCloud(samples, ref_samples)
            ot_prob = linear_problem.LinearProblem(geom)
            solver = sinkhorn.Sinkhorn()
            ot = solver(ot_prob)
            P = ot.matrix.flatten()
            choices = jax.random.choice(key_ot, batch_size * batch_size, (batch_size,), p=P)
            i, j = jnp.divmod(choices, batch_size)
            samples = samples.at[i].get()
            ref_samples = ref_samples.at[j].get()
        norm_samples = jax.random.normal(key_gaussian, (batch_size, n_dim))
        cond_samples = args.sigma * norm_samples + times * samples + (1 - times) * ref_samples
        target_vector_fields = samples - ref_samples
        return times, cond_samples, target_vector_fields

    def flow_matching_loss(rng_key, samples, vector_field_param):
        if args.cond_flow or args.ot_cond_flow:
            times, cond_samples, target_vector_fields = cond_flow_fn(rng_key, samples)
        else:
            times, cond_samples, target_vector_fields = flow_fn(rng_key, samples)
        approx_vector_fields = jax.vmap(vector_field_apply, (None, 0, 0))(vector_field_param, cond_samples, times)
        diffs = approx_vector_fields - target_vector_fields
        return jnp.square(diffs).sum()
        # return jnp.square(diffs).mean()

    return TrainState.create(
        apply_fn=vector_field_apply,
        params=vector_field_param,
        tx=optax.apply_if_finite(optax.chain(tx, clipper), 10),
        loss_fn=flow_matching_loss,
    )


def create_learning_rate_fn(
    num_train_steps, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def create_train_data_gn(dist, 
        vector_field_apply, ode_integrator, args):
    kernel = build_kernel()
    batch_size, dim = dist.init_params.shape

    def transform_and_logdet(key, ref_sample, vector_field_param, **vector_field_kwargs):

        def augmented_vector_field(u_ldj, time):
            u, _ = u_ldj
            du = vector_field_apply(vector_field_param, u, time, **vector_field_kwargs)
            if args.hutchs:
                rand_norm = jax.random.normal(key, (dim,))
                _, jvp = jax.jvp(lambda u: vector_field_apply(vector_field_param, u, time, **vector_field_kwargs), (u,), (rand_norm,))
                dldj = jnp.dot(rand_norm, jvp)
            else:
                jacobian = jax.jacfwd(vector_field_apply, 1)(vector_field_param, u, time, **vector_field_kwargs)
                dldj = jnp.trace(jacobian)
            return du, -dldj

        flow, ldj_flow = ode_integrator(augmented_vector_field, (ref_sample, jnp.zeros(())))
        return flow[-1], ldj_flow[-1]
    
    def inverse_and_logdet(key, target_sample, vector_field_param, **vector_field_kwargs):

        def augmented_vector_field(x_ldj, time):
            x, _ = x_ldj
            #instead of doing t=1 to t=0 with vector_field(x, t) we do t=-1 to t=0 with -vector_field(x, -t)
            #or equivalently t=0 to t=1 with -vector_field(x, 1-t)
            time = 1. - time
            dx = -vector_field_apply(vector_field_param, x, time, **vector_field_kwargs)
            if args.hutchs:
                rand_norm = jax.random.normal(key, (dim,))
                _, jvp = jax.jvp(lambda x: vector_field_apply(vector_field_param, x, time, **vector_field_kwargs), (x,), (rand_norm,))
                dldj = jnp.dot(rand_norm, jvp)
            else:
                jacobian = jax.jacfwd(vector_field_apply, 1)(vector_field_param, x, time, **vector_field_kwargs)
                dldj = jnp.trace(jacobian)
            #negation of trace cancels out
            return dx, dldj

        inv_flow, ldj_inv_flow = ode_integrator(augmented_vector_field, (target_sample, jnp.zeros(())))
        return inv_flow[-1], ldj_inv_flow[-1]
    
    ref_dist = ref_dists[args.ref_dist](dim)

    def indep_metropolis_hastings(rng_key, prev_state, logprob, vector_field_param, **vector_field_kwargs):
        key_gen, key_acc, key_hutch1, key_hutch2 = jax.random.split(rng_key, 4)
        initial_position = prev_state.position
        proposed_pullback = ref_dist.sample_model(key_gen)
        proposed_position, proposed_delta_vol = transform_and_logdet(key_hutch1, proposed_pullback, vector_field_param, **vector_field_kwargs)
        initial_pullback, initial_delta_vol = inverse_and_logdet(key_hutch2, initial_position, vector_field_param, **vector_field_kwargs)
        proposed_logdensity, proposed_logdensity_grad = jax.value_and_grad(logprob)(proposed_position)
        acceptance_prob = jnp.exp(
            proposed_logdensity - ref_dist.logprob(proposed_pullback) - proposed_delta_vol
            + ref_dist.logprob(initial_pullback) - initial_delta_vol - prev_state.logdensity
        )
        return jax.lax.cond(jax.random.uniform(key_acc) <= acceptance_prob,
            lambda _: (MALAState(proposed_position, proposed_logdensity, proposed_logdensity_grad), MALAInfo(acceptance_prob, True, proposed_position, 0.)),
            lambda _: (prev_state, MALAInfo(acceptance_prob, False, proposed_position, 0.)),
            operand=None)
    
    optimal_scale = 2.38 / jnp.sqrt(dim)

    def random_walk_metropolis_hastings(rng_key, prev_state, logprob, vector_field_param, **vector_field_kwargs):
        key_gen, key_acc, key_hutch1, key_hutch2 = jax.random.split(rng_key, 4)
        initial_position = prev_state.position
        initial_pullback, initial_delta_vol = inverse_and_logdet(key_hutch2, initial_position, vector_field_param, **vector_field_kwargs)
        proposed_pullback = initial_pullback + optimal_scale * jax.random.normal(key_gen, (dim,))
        proposed_position, proposed_delta_vol = transform_and_logdet(key_hutch1, proposed_pullback, vector_field_param, **vector_field_kwargs)
        proposed_logdensity, proposed_logdensity_grad = jax.value_and_grad(logprob)(proposed_position)
        acceptance_prob = jnp.exp(
            proposed_logdensity - proposed_delta_vol
            - prev_state.logdensity - initial_delta_vol
        )
        return jax.lax.cond(jax.random.uniform(key_acc) <= acceptance_prob,
            lambda _: (MALAState(proposed_position, proposed_logdensity, proposed_logdensity_grad), MALAInfo(acceptance_prob, True, proposed_position, 0.)),
            lambda _: (prev_state, MALAInfo(acceptance_prob, False, proposed_position, 0.)),
            operand=None)
    
    def conditional_importance_sampling(rng_key, prev_state, logprob, vector_field_param, **vector_field_kwargs):
        key_sample, key_hutch_prev, key_hutch, key_choice = jax.random.split(rng_key, 4)
        pullback_prev, vol_prev = inverse_and_logdet(key_hutch_prev, prev_state.position, vector_field_param, **vector_field_kwargs)
        prev_weight = jnp.exp(prev_state.logdensity - ref_dist.logprob(pullback_prev) - vol_prev)
        keys_sample = jax.random.split(key_sample, args.num_importance_samples)
        ref_samples = jax.vmap(ref_dist.sample_model)(keys_sample)
        keys_hutch = jax.random.split(key_hutch, args.num_importance_samples)
        samples, vols = jax.vmap(transform_and_logdet, (0, 0, None))(keys_hutch, ref_samples, vector_field_param, **vector_field_kwargs)
        samples_logdensity = jax.vmap(logprob)(samples)
        weights = jax.vmap(lambda logdensity, ref_sample, vol: jnp.exp(logdensity - ref_dist.logprob(ref_sample) - vol))(samples_logdensity, ref_samples, vols)
        sum_weights = prev_weight + weights.sum()
        norm_weights = jnp.hstack([prev_weight, weights]) / sum_weights
        choice = jax.random.choice(key_choice, args.num_importance_samples + 1, p=norm_weights)
        return jax.lax.cond(choice == 0,
            lambda _: (prev_state, MALAInfo(norm_weights[0], False, prev_state.position, norm_weights[0])),
            lambda _: (MALAState(samples[choice - 1], samples_logdensity[choice - 1], prev_state.logdensity_grad), MALAInfo(norm_weights[choice], True, samples[choice - 1], norm_weights[choice])),
            operand=None)
    
    flow_step = conditional_importance_sampling if args.num_importance_samples > 0 else indep_metropolis_hastings if args.num_importance_samples < 0 else random_walk_metropolis_hastings

    def train_data_generator(rng_key, states, count, vector_field_param, beta=1., **vector_field_kwargs):
        logprob = lambda position: beta * dist.loglik(position) + dist.logprior(position)
        # logprob = lambda position: beta * dist.logprob(position) + (1 - beta) * ref_dist.logprob(position)
        keys = jax.random.split(rng_key, batch_size)
        if args.mcmc_per_flow_steps > 0 and args.mcmc_per_flow_steps < 1:
            flow_per_mcmc_steps = int(1 / args.mcmc_per_flow_steps)
            return jax.lax.cond(count % (flow_per_mcmc_steps + 1) == 0,
                lambda _: jax.vmap(lambda k, s: kernel(k, s, logprob, args.step_size))(keys, states),
                lambda _: jax.vmap(flow_step, (0, 0, None, None))(keys, states, logprob, vector_field_param, **vector_field_kwargs),
                operand=None)
        else:
            return jax.lax.cond(count % (int(args.mcmc_per_flow_steps) + 1) == 0,
                lambda _: jax.vmap(flow_step, (0, 0, None, None))(keys, states, logprob, vector_field_param, **vector_field_kwargs),
                lambda _: jax.vmap(lambda k, s: kernel(k, s, logprob, args.step_size))(keys, states),
                operand=None)

    init_fn = lambda init_positions, beta=1.: jax.vmap(init, (0, None))(init_positions, lambda position: beta * dist.loglik(position) + dist.logprior(position))
    # init_fn = lambda init_positions, beta=1.: jax.vmap(init, (0, None))(init_positions, lambda position: beta * dist.logprob(position) + (1 - beta) * ref_dist.logprob(position))
    return train_data_generator, init_fn, transform_and_logdet


def run(dist, args, target_gn=None):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    use_real_samples = args.mcmc_per_flow_steps < 0
    learning_iter = args.learning_iter
    iter_per_temp = args.anneal_iter // args.num_anneal_temp
    n_iter = args.eval_iter
    n_chain = args.num_chain
    key_target, key_sample, key_init, key_dist, key_fourier, key_gen = jax.random.split(jax.random.PRNGKey(args.seed), 6)
    dist.initialize_model(key_dist, n_chain)
    
    # if args.dim > 64:
    #     def odeintegrator(func, x0):
    #         term = ODETerm(lambda t, y, args: func(y, t))
    #         solver = Dopri5()
    #         saveat = SaveAt(ts=[0., 1.])
    #         stepsize_controller = PIDController(rtol=args.rtol, atol=args.atol)
    #         return diffeqsolve(term, solver, t0=0, t1=1, dt0=None, y0=x0, saveat=saveat,
    #                         stepsize_controller=stepsize_controller).ys#, max_steps=None).ys
    # else:
    odeintegrator = lambda func, x0: odeint(
        func, x0, 
        jnp.linspace(0.0, 1.0, 5 if args.example == "4-mode" else 2), 
        rtol=args.rtol, atol=args.atol, 
        mxstep=args.mxstep)
    fourier_random = args.fourier_std * jax.random.normal(key_fourier, (args.fourier_dim,))
    model = VectorFieldNet(fourier_random, jax.grad(dist.logprob), args.hidden_x, args.hidden_t, args.hidden_xt, non_lins[args.non_linearity], args.gradient_clip if args.dim > 128 else None)
    # model = GaussianMixtureTransformer(fourier_random, jax.grad(dist.logprob), [128, 64, 32, 16], 16, 16, non_lins[args.non_linearity], args.gradient_clip if args.dim > 128 else None)
    vector_field_param = model.init(key_init, dist.init_params[0], 0.)

    learning_rate_fn = create_learning_rate_fn(
        learning_iter,
        args.warmup_steps,
        args.learning_rate,
    )
    state = create_train_state(model.apply, vector_field_param, learning_rate_fn, args)

    def train_step(state: train_state.TrainState, positions, rng_key):
        """Trains model with an optimizer (both in `state`) on `batch`, returning a pair `(new_state, loss)`."""
        grad_fn = jax.value_and_grad(state.loss_fn, argnums=2)
        loss, grad = grad_fn(rng_key, positions, state.params)
        new_state = state.apply_gradients(grads=grad)
        metrics = {"loss": loss, "learning_rate": learning_rate_fn(state.step)}
        return new_state, metrics

    if target_gn is not None:
        key_gen, key_loss = jax.random.split(key_target)
        keys_target = jax.random.split(key_gen, n_iter * n_chain)
        real_samples = jax.vmap(target_gn)(keys_target)
        eval_step = jax.jit(lambda state: state.loss_fn(key_loss, real_samples, state.params))


    logger.info(f"===== Starting training seed {args.seed} w/ {learning_iter} iterations =====")
    logger.info("mcmc_per_flow_steps=" + str(args.mcmc_per_flow_steps) + ",learning_iter=" + str(args.learning_iter) + (",hutchs" if args.hutchs else ""))

    train_data_generator, init_fn, transform_and_logdet = create_train_data_gn(dist,
        model.apply, odeintegrator, args)
    if use_real_samples:
        train_data_generator = lambda key, *_: jax.vmap(
            lambda k: (MALAState(target_gn(k), None, None), MALAInfo(jnp.nan, None, None, None))
        )(jax.random.split(key, n_chain))
        init_fn = lambda positions, *_: jax.vmap(lambda p: MALAState(p, None, None))(positions)
        
    ref_dist = ref_dists[args.ref_dist](args.dim)
    sample_reference = lambda key: jax.vmap(ref_dist.sample_model)(jax.random.split(key, n_iter * n_chain))

    def beta_fn(prev_beta, logliks):
    # def beta_fn(prev_beta, logprobs, reflogprobs):
        def ess_zero(beta):
            logw = logliks * (beta - prev_beta)
            # logw = (logprobs - reflogprobs) * (beta - prev_beta)
            logw_max = jnp.max(logw)
            logw_normed = logw - logw_max
            weights = jnp.exp(logw_normed) / jnp.sum(jnp.exp(logw_normed))
            return 1.0 / jnp.sum(weights * weights) - args.alpha * n_chain
        bisec = Bisection(optimality_fun=ess_zero, lower=prev_beta, upper=1., maxiter=30, tol=1e-5, check_bracket=False)
        beta = bisec.run().params
        return beta, logliks * (beta - prev_beta)
        # return beta, (logprobs - reflogprobs) * (beta - prev_beta)
        
    # train_start = time.time() #pre jit
    train_data_generator = jax.jit(train_data_generator)
    init_fn = jax.jit(init_fn)
    train_step = jax.jit(train_step)
    beta_fn = jax.jit(beta_fn)
    @jax.jit
    def beta_gen(beta, train_states):
        def new_beta(beta, train_states):
            beta, _ = beta_fn(beta, mapped_loglik(train_states.position))
            # beta, _ = beta_fn(beta, mapped_logprob(train_states.position), mapped_ref(train_states.position))
            train_states = init_fn(train_states.position, beta)
            return beta, train_states
        return jax.lax.cond(beta < 1., new_beta, lambda b, ts: (b, ts), beta, train_states)
    mapped_loglik = jax.vmap(dist.loglik)
    mapped_logprob = jax.vmap(dist.logprob)
    mapped_ref = jax.vmap(ref_dist.logprob)
    train_start = time.time() #post jit

    # if args.anneal_iter > 0 and not use_real_samples:
    #     beta = args.anneal_temp.pop(0)
    if not use_real_samples:
        beta, _ = beta_fn(0., mapped_loglik(dist.init_params))
        # beta, _ = beta_fn(0., mapped_logprob(dist.init_params), mapped_ref(dist.init_params))
        logger.info(f"Initial beta= {beta}")
    else:
        beta = 1.
    train_states = init_fn(dist.init_params, beta)
    for count in tqdm(range(1, learning_iter + 1), desc="Training..."):
        key_sample, key_train_gn, key_train_step = jax.random.split(key_sample, 3)
        # if args.anneal_iter >= count and not use_real_samples:
        #     if count % (iter_per_temp + 1) == 0:
        #         beta = args.anneal_temp.pop(0)
        #         train_states = init_fn(train_states.position, beta)
        train_states, infos = train_data_generator(key_train_gn, train_states, count, state.params, beta)
        state, train_metric = train_step(state, train_states.position, key_train_step)
        if not use_real_samples and count % iter_per_temp == 0:
            beta, train_states = beta_gen(beta, train_states)
        train_metric["acceptance avg."] = infos.acceptance_rate.mean()
        train_metric["acceptance std."] = infos.acceptance_rate.std()
        if target_gn is not None:
            target_loss = eval_step(state)
            train_metric["target_loss"] = target_loss
        train_time = time.time() - train_start
        train_metric["train_time"] = train_time
        wandb.log(train_metric)
    logger.info(f"Final beta= {beta}")


    u = sample_reference(key_gen)
    key_hutch, key_choice = jax.random.split(key_gen)
    flow_samples, vols = jax.vmap(lambda u: transform_and_logdet(key_hutch, u, state.params))(u)
    samples_logdensity = jax.vmap(dist.logprob)(flow_samples)
    log_weights = jax.vmap(lambda logdensity, ref_sample, vol: logdensity - ref_dist.logprob(ref_sample) - vol)(samples_logdensity, u, vols)
    weights = jnp.exp(log_weights - log_weights.max())
    exact_samples = jax.random.choice(key_choice, flow_samples, (n_iter * n_chain,), p=weights)


    if args.check:
        logger.info(f"Logpdf of real samples= {jax.vmap(dist.logprob)(real_samples).mean()}")
        stein = stein_disc(real_samples, dist.logprob)
        logger.info(f"Stein U, V disc of real samples= {stein[0]}, {stein[1]}")
        mmd = max_mean_disc(real_samples, real_samples)
        logger.info(f"Max mean disc of NF+MCMC samples= {mmd}")

    logpdf = jax.vmap(dist.logprob)(flow_samples).mean()
    logger.info(f"Logpdf of flow samples= {logpdf}")
    stein = stein_disc(flow_samples, dist.logprob)
    logger.info(f"Stein U, V disc of flow samples= {stein[0]}, {stein[1]}")
    logpdf_ = jax.vmap(dist.logprob)(exact_samples).mean()
    logger.info(f"Logpdf of exact samples= {logpdf_}")
    stein_ = stein_disc(exact_samples, dist.logprob)
    logger.info(f"Stein U, V disc of exact samples= {stein_[0]}, {stein_[1]}")
    data = [args.mcmc_per_flow_steps, args.learning_iter, train_time, logpdf, logpdf_, stein[0], stein_[0], stein[1], stein_[1]]
    columns = ["mcmc/flow", "learn iter", "train time", "logpdf", "logpdf*", "KSD U-stat", "KSD U-stat*", "KSD V-stat", "KSD V-stat*"]

    if target_gn is not None:
        mmd = max_mean_disc(real_samples, flow_samples)
        logger.info(f"Max mean disc of flow samples= {mmd}")
        data.append(mmd)
        columns.append("MMD")
        mmd_ = max_mean_disc(real_samples, exact_samples)
        logger.info(f"Max mean disc of exact samples= {mmd_}")
        data.append(mmd_)
        columns.append("MMD*")
    else:
        mmd = mmd_ = 0.

    if args.example == "phi-four":
        #fields
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
        ax[1].set_title(r"$\hat{\phi}$")
        ax[1].set_xlabel(r"$d$")
        ax[1].set_ylabel(r"$\phi$")
        flow_samples = jnp.pad(flow_samples, ((0, 0), (1, 1))) #for the phi-four example
        for i in range(flow_samples.shape[0]):
            ax[1].plot(flow_samples[i], color='red', alpha=0.1)
        ax[0].set_title(r"$\pi$")
        ax[0].set_xlabel(r"$d$")
        ax[0].set_ylabel(r"$\phi$")
        exact_samples = jnp.pad(exact_samples, ((0, 0), (1, 1))) #for the phi-four example
        for i in range(exact_samples.shape[0]):
            ax[0].plot(exact_samples[i], color='red', alpha=0.1)
        plt.setp(ax, xlim=[0, args.dim + 1], ylim=args.lim)
        data.append(wandb.Image(fig))
        columns.append("plot phi")
        plt.close()
    
    #mixtures
    for i in range(args.dim - 1):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[1].set_title(r"$\hat{\phi}$")
        ax[1].set_xlabel(r"$x_1$")
        ax[1].set_ylabel(r"$x_{-1}$")
        # sns.histplot(x=flow_samples[:, 0], y=flow_samples[:, i+1], ax=ax[1], bins=50)
        ax[1].plot(flow_samples[:, 0], flow_samples[:, i+1], '.', alpha=.2, color="blue")
        ax[0].set_title(r"$\pi$")
        ax[0].set_xlabel(r"$x_1$")
        ax[0].set_ylabel(r"$x_{-1}$")
        # sns.histplot(x=exact_samples[:, 0], y=exact_samples[:, i+1], ax=ax[0], bins=50)
        ax[0].plot(exact_samples[:, 0], exact_samples[:, i+1], '.', alpha=.2, color="blue")
        plt.setp(ax, xlim=args.lim or plt.xlim(), ylim=args.lim or plt.ylim())
        if args.dim == 2:
            plot_contours(dist.logprob, ax, args)
        data.append(wandb.Image(fig))
        columns.append("plot (x0,x" + str(i+1) + ")")
        plt.close()
        if i > 8:
            break #only the first 10 dimensions

    if args.example == "4-mode":
        #4-mode mixture
        flow = lambda u: odeintegrator(lambda u, t: state.apply_fn(state.params, u, t), u)
        flow_inv = lambda x: odeintegrator(lambda x, t: -state.apply_fn(state.params, x, 1-t), x)
        forward_prog = jax.vmap(flow)(u)
        n_col = forward_prog.shape[1]
        fig, ax = plt.subplots(1, n_col, figsize=(25, 3))
        for i in range(n_col):
            ax[i].plot(forward_prog[:, i, 0], forward_prog[:, i, 1], '.', alpha=.2, color="blue")
        data.append(wandb.Image(fig))
        columns.append("forward progression")
        plt.close()
        fig, ax = plt.subplots(1, n_col, figsize=(25, 3))
        mode_chains = n_chain // 4
        colors = ['red', 'blue', 'green', 'yellow']
        for j in range(4):
            keys_mode = keys_target[j * (n_iter * mode_chains):(j + 1) * (n_iter * mode_chains)]
            mode_u = jax.vmap(lambda k: dist.modes[j] + dist.chol_covs[j] * jax.random.normal(k, (args.dim,)))(keys_mode)
            backward_prog = jax.vmap(flow_inv)(mode_u)
            for i in range(n_col):
                ax[n_col - i - 1].plot(backward_prog[:, i, 0], backward_prog[:, i, 1], '.', alpha=.2, color=colors[j])
        data.append(wandb.Image(fig))
        columns.append("backwards progression")
        plt.close()

    wandb.log({"summary": wandb.Table(columns, [data])})
    wandb.finish()
    return jnp.array([logpdf, stein[0], stein[1], mmd, train_time]), jnp.array([logpdf_, stein_[0], stein_[1], mmd_, train_time])

import itertools
def plot_contours(log_prob_func: Callable, ax: plt.Axes, args):
    """Plot contours of a log_prob_func that is defined on 2D"""
    x_points_dim1 = jnp.linspace(args.lim[0], args.lim[1], args.grid_width)
    x_points_dim2 = x_points_dim1
    x_points = jnp.array(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_p_x = jax.vmap(log_prob_func)(x_points)
    log_p_x = jnp.maximum(log_p_x, -1000)
    log_p_x = log_p_x.reshape((args.grid_width, args.grid_width))
    x_points_dim1 = x_points[:, 0].reshape((args.grid_width, args.grid_width))
    x_points_dim2 = x_points[:, 1].reshape((args.grid_width, args.grid_width))
    ax[0].contour(x_points_dim1, x_points_dim2, log_p_x, levels=args.levels)
    ax[1].contour(x_points_dim1, x_points_dim2, log_p_x, levels=args.levels)
