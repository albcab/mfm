
import jax
import jax.numpy as jnp

import haiku as hk


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


def mlp_flow(key_init, args, N_PARAM, grad_logporob):
    n_chain = args.num_chain
    non_lin = non_lins[args.non_linearity]

    def mlp_vector_field(time, sample):
        (out_dim,) = sample.shape
        # mlp = mlp_generator(out_dim, hidden_dims=args.hidden_layers, non_linearity=non_lin)
        mlp = hk.nets.MLP(args.hidden_layers + [out_dim], activation=non_lin)
        input = jnp.hstack([time, sample])
        return mlp(input)

    # vector_field = hk.without_apply_rng(hk.transform(mlp_vector_field))
    # vector_field_params = vector_field.init(key_init, 0.0, jnp.zeros(N_PARAM))

    def get_input(sample, time):
        stacked = jnp.hstack([sample, time])
        # stack_outer = jnp.outer(stacked, stacked)
        stack_outer = jnp.stack([stacked, stacked[::-1]])
        return stack_outer.reshape(stack_outer.shape + (1,))
        # sample_outer = jnp.outer(sample, sample)
        # freq = 2 * jnp.pi * time
        # return jnp.stack([sample_outer, sample_outer * jnp.cos(freq), sample_outer * jnp.sin(freq)]).transpose((1, 2, 0))

    def batch_vector_field(times, samples, is_training):
        batch_size, dim = samples.shape
        # net = hk.nets.ResNet18(dim, resnet_v2=True, bn_config={'decay_rate': 0.9})
        net = hk.nets.MLP(args.hidden_layers + [dim], activation=non_lin)
        input = jnp.concatenate([samples, times.reshape(batch_size, 1)], axis=1)#.reshape(batch_size, dim + 1, 1, 1)
        # ngrad_logprobs = jax.vmap(lambda s: -grad_logporob(s))(samples)
        # batched_times = times.reshape(batch_size, 1)
        # input = jnp.concatenate([samples, ngrad_logprobs, batched_times], axis=1)#.reshape(batch_size, dim + 1, 1, 1)
        return jax.vmap(net)(input)
        # input = jax.vmap(get_input)(samples, times)
        # return net(input, is_training=is_training)

    vector_field = hk.without_apply_rng(hk.transform_with_state(batch_vector_field))
    apply_fn = vector_field.apply
    vector_field_params, vector_field_state = vector_field.init(key_init, jnp.zeros(n_chain), jnp.zeros((n_chain, N_PARAM)), is_training=True)
    return vector_field_params, vector_field_state, apply_fn, None


def resnet_flow(key_init, args, N_PARAM):
    n_chain = args.num_chain
    m = args.hidden_layers[0]

    def resnet_vector_field(times, samples, is_training):
        batch_size, dim = samples.shape
        freqs = jax.vmap(lambda t: jnp.array([jnp.sin(2 * jnp.pi * t), jnp.cos(2 * jnp.pi * t)]))(times)
        net1_time = hk.nets.MLP(args.hidden_layers, activation=jax.nn.silu)
        time_embs = jax.vmap(lambda f: net1_time(f))(freqs)
        hidden_states = samples.reshape(batch_size, dim, 1)
        hidden_states = hk.Conv1D(m, 2)(hidden_states)
        residual = hidden_states
        # hidden_states = hk.BatchNorm(True, True, 0.9)(hidden_states, is_training=is_training)
        hidden_states = jax.nn.swish(hidden_states)
        hidden_states = hk.Conv1D(m, 2)(hidden_states)
        net2_time = hk.Sequential([jax.nn.swish, hk.Linear(m)])
        temb = jnp.expand_dims(jax.vmap(lambda t: net2_time(t))(time_embs), 1)
        hidden_states = hidden_states + temb
        # hidden_states = hk.BatchNorm(True, True, 0.9)(hidden_states, is_training=is_training)
        hidden_states = jax.nn.swish(hidden_states)
        hidden_states = hk.Conv1D(m, 2)(hidden_states)
        hidden_states = hidden_states + residual
        # hidden_states = hk.BatchNorm(True, True, 0.9)(hidden_states, is_training=is_training)
        hidden_states = jax.nn.silu(hidden_states)
        hidden_states = hk.Conv1D(1, 2)(hidden_states)
        return hidden_states.squeeze()
    
    vector_field = hk.without_apply_rng(hk.transform_with_state(resnet_vector_field))
    apply_fn = vector_field.apply
    vector_field_params, vector_field_state = vector_field.init(key_init, jnp.zeros(n_chain), jnp.zeros((n_chain, N_PARAM)), is_training=True)
    return vector_field_params, vector_field_state, apply_fn, None


def ot_flow(key_init, args, N_PARAM): #doesn't support state architecture... or batch input
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

    m = args.hidden_layers[0]
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
    
    vector_field_state = None
    return vector_field_params, apply_fn, vector_field_with_jacobian_trace