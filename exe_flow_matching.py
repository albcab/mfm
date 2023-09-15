import logging
import time
from typing import Callable

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax import struct, traverse_util
from flax.training import train_state

from jax.experimental.ode import odeint

import optax

import wandb
from tqdm import tqdm

from blackjax.mcmc.mala import mala, MALAState, MALAInfo
from mcmc_utils import stein_disc, max_mean_disc

import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


non_lins = {
    'tanh': jax.nn.tanh,
    'elu': jax.nn.elu,
    'relu': jax.nn.relu,
    'swish': jax.nn.swish,
}

class VectorFieldNet(nn.Module):

    fourier_random: jax.Array
    grad_logporob: Callable
    hidden_x: list
    hidden_t: list
    hidden_xt: list
    act_fn: Callable = jax.nn.relu

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

    def flow_matching_loss(rng_key, samples, vector_field_param):
        times, cond_samples, target_vector_fields = flow_fn(rng_key, samples)
        approx_vector_fields = jax.vmap(vector_field_apply, (None, 0, 0))(vector_field_param, cond_samples, times)
        diffs = approx_vector_fields - target_vector_fields
        return jnp.square(diffs).sum()
        # return jnp.square(diffs).mean()

    return TrainState.create(
        apply_fn=vector_field_apply,
        params=vector_field_param,
        tx=optax.chain(tx, clipper),
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


def create_train_data_gn(logprob_fn, initial_positions, 
        vector_field_apply, ode_integrator,
        mcmc_per_flow_steps, mala_step_size, hutchs):
    init, step = mala(logprob_fn, mala_step_size)
    batch_size, dim = initial_positions.shape

    def transform_and_logdet(key, ref_sample, vector_field_param, **vector_field_kwargs):

        def augmented_vector_field(u_ldj, time):
            u, _ = u_ldj
            du = vector_field_apply(vector_field_param, u, time, **vector_field_kwargs)
            if hutchs:
                rand_norm = jax.random.normal(key, (dim,))
                _, jvp = jax.jvp(lambda u: vector_field_apply(vector_field_param, u, time, **vector_field_kwargs), (u,), (rand_norm,))
                dldj = jnp.dot(rand_norm, jvp)
            else:
                jacobian = jax.jacfwd(vector_field_apply, 1)(vector_field_param, u, time, **vector_field_kwargs)
                dldj = jnp.einsum("ii", jacobian)
            return du, -dldj

        flow, ldj_flow = ode_integrator(augmented_vector_field, (ref_sample, jnp.zeros(())))
        return flow[-1], ldj_flow[-1]
    
    def inverse_and_logdet(key, target_sample, vector_field_param, **vector_field_kwargs):

        def augmented_vector_field(x_ldj, time):
            x, _ = x_ldj
            time = 1.0 - time
            dx = vector_field_apply(vector_field_param, x, time, **vector_field_kwargs)
            if hutchs:
                rand_norm = jax.random.normal(key, (dim,))
                _, jvp = jax.jvp(lambda x: vector_field_apply(vector_field_param, x, time, **vector_field_kwargs), (x,), (rand_norm,))
                dldj = jnp.dot(rand_norm, jvp)
            else:
                jacobian = jax.jacfwd(vector_field_apply, 1)(vector_field_param, x, time, **vector_field_kwargs)
                dldj = jnp.einsum("ii", jacobian)
            return dx, -dldj

        inv_flow, ldj_inv_flow = ode_integrator(augmented_vector_field, (target_sample, jnp.zeros(())))
        return inv_flow[-1], ldj_inv_flow[-1]
    
    def indep_metropolis_hastings(rng_key, prev_state, vector_field_param, **vector_field_kwargs):
        key_gen, key_acc, key_hutch1, key_hutch2 = jax.random.split(rng_key, 4)
        initial_position = prev_state.position
        proposed_pullback = jax.random.normal(key_gen, (dim,))
        proposed_position, proposed_delta_vol = transform_and_logdet(key_hutch1, proposed_pullback, vector_field_param, **vector_field_kwargs)
        initial_pullback, initial_delta_vol = inverse_and_logdet(key_hutch2, initial_position, vector_field_param, **vector_field_kwargs)
        proposed_logdensity, proposed_logdensity_grad = jax.value_and_grad(logprob_fn)(proposed_position)
        acceptance_prob = jnp.exp(
            proposed_logdensity + .5 * jnp.square(proposed_pullback).sum() - proposed_delta_vol
            -.5 * jnp.square(initial_pullback).sum() - initial_delta_vol - prev_state.logdensity
        )
        return jax.lax.cond(jax.random.uniform(key_acc) <= acceptance_prob,
            lambda _: (MALAState(proposed_position, proposed_logdensity, proposed_logdensity_grad), MALAInfo(acceptance_prob, True, proposed_position, 0.)),
            lambda _: (prev_state, MALAInfo(acceptance_prob, False, proposed_position, 0.)),
            operand=None)
    
    def train_data_generator(rng_key, states, count, vector_field_param, **vector_field_kwargs):
        keys = jax.random.split(rng_key, batch_size)
        if mcmc_per_flow_steps > 0 and mcmc_per_flow_steps < 1:
            flow_per_mcmc_steps = int(1 / mcmc_per_flow_steps)
            return jax.lax.cond(count % (flow_per_mcmc_steps + 1) == 0,
                lambda _: (jax.vmap(step)(keys, states), count + 1),
                lambda _: (jax.vmap(indep_metropolis_hastings, (0, 0, None))(keys, states, vector_field_param, **vector_field_kwargs), count + 1),
                operand=None)
        else:
            return jax.lax.cond(count % (int(mcmc_per_flow_steps) + 1) == 0,
                lambda _: (jax.vmap(indep_metropolis_hastings, (0, 0, None))(keys, states, vector_field_param, **vector_field_kwargs), count + 1),
                lambda _: (jax.vmap(step)(keys, states), count + 1),
                operand=None)

    initial_states = jax.vmap(init)(initial_positions)
    initial_count = 0
    return train_data_generator, initial_states, initial_count


def run(dist, args, target_gn=None):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    use_real_samples = args.mcmc_per_flow_steps < 0
    n_warm = args.learning_iter
    n_iter = args.eval_iter
    n_chain = args.num_chain
    key_target, key_sample, key_init, key_dist, key_fourier, key_gen = jax.random.split(jax.random.PRNGKey(args.seed), 6)
    dist.initialize_model(key_dist, n_chain)

    odeintegrator = lambda func, x0: odeint(
        func, x0, 
        jnp.linspace(0.0, 1.0, 2), 
        rtol=args.rtol, atol=args.atol, 
        mxstep=args.mxstep)
    fourier_random = args.fourier_std * jax.random.normal(key_fourier, (args.fourier_dim,))
    model = VectorFieldNet(fourier_random, jax.grad(dist.logprob), args.hidden_x, args.hidden_t, args.hidden_xt, non_lins[args.non_linearity])
    vector_field_param = model.init(key_init, dist.init_params[0], 0.)

    learning_rate_fn = create_learning_rate_fn(
        n_warm,
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


    logger.info(f"===== Starting training ({n_warm} iterations) =====")

    if use_real_samples:
        train_data_generator = lambda key, *_: jax.vmap(
            lambda k: ((MALAState(target_gn(k), None, None), MALAInfo(jnp.nan, None, None, None)), 0)
        )(jax.random.split(key, n_chain))
        train_states = jax.vmap(lambda p: MALAState(p, None, None))(dist.init_params)
        count = 0
    else:
        train_data_generator, train_states, count = create_train_data_gn(dist.logprob, dist.init_params, 
            model.apply, odeintegrator, args.mcmc_per_flow_steps, args.step_size, args.hutchs)
        
    # train_start = time.time() #pre jit
    train_data_generator = jax.jit(train_data_generator)
    train_step = jax.jit(train_step)
    train_start = time.time() #post jit
    for _ in tqdm(range(n_warm), desc="Training..."):
        key_sample, key_train_gn, key_train_step = jax.random.split(key_sample, 3)
        (train_states, infos), count = train_data_generator(key_train_gn, train_states, count, vector_field_param)
        state, train_metric = train_step(state, train_states.position, key_train_step)
        train_metric["acceptance avg."] = infos.acceptance_rate.mean()
        train_metric["acceptance std."] = infos.acceptance_rate.std()
        if target_gn is not None:
            target_loss = eval_step(state)
            train_metric["target_loss"] = target_loss
        train_time = time.time() - train_start
        train_metric["train_time"] = train_time
        wandb.log(train_metric)


    flow_odeintegrator = lambda func, x0: odeint(
        func, x0, 
        jnp.linspace(0.0, 1.0, 2), 
        rtol=args.flow_rtol, atol=args.flow_atol, 
        mxstep=args.flow_mxstep)
    
    def flow(u, param):
        flow = flow_odeintegrator(lambda u, time: model.apply(param, u, time), u)
        return flow[-1]


    u = jax.random.normal(key_gen, shape=(n_iter * n_chain, dist.init_params.shape[1]))
    flow_samples = jax.vmap(lambda u: flow(u, state.params))(u)


    if args.check:
        print("Logpdf of real samples=", jax.vmap(dist.logprob)(real_samples).sum())
        stein = stein_disc(real_samples, dist.logprob_fn)
        print("Stein U, V disc of real samples=", stein[0], stein[1])
        mmd = max_mean_disc(real_samples, real_samples)
        print("Max mean disc of NF+MCMC samples=", mmd)
        print()

    logpdf = jax.vmap(dist.logprob)(flow_samples).sum()
    print("Logpdf of flow samples=", logpdf)
    stein = stein_disc(flow_samples, dist.logprob)
    print("Stein U, V disc of flow samples=", stein[0], stein[1])
    data = [args.mcmc_per_flow_steps, args.learning_iter, logpdf, stein[0], stein[1]]
    columns = ["mcmc/flow", "learn iter", "logpdf", "KSD U-stat", "KSD V-stat"]

    if target_gn is not None:
        mmd = max_mean_disc(real_samples, flow_samples)
        print("Max mean disc of flow samples=", mmd)
        data.append(mmd)
        columns.append("MMD")
        print()
        
        for i in range(real_samples.shape[1] - 1):
            fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
            ax[1].set_title(r"$\hat{\phi}$")
            ax[1].set_xlabel(r"$x_1$")
            ax[1].set_ylabel(r"$x_{-1}$")
            sns.histplot(x=flow_samples[:, 0], y=flow_samples[:, i+1], ax=ax[1], bins=50)
            ax[0].set_title(r"$\pi$")
            ax[0].set_xlabel(r"$x_1$")
            ax[0].set_ylabel(r"$x_{-1}$")
            sns.histplot(x=real_samples[:, 0], y=real_samples[:, i+1], ax=ax[0], bins=50)
            data.append(wandb.Image(fig))
            columns.append("plot (x0,x" + str(i+1) + ")")
            plt.close()
            if i > 8:
                break #only the first 10 dimensions

    wandb.log({"summary": wandb.Table(columns, [data])})
    wandb.finish()
    return None
