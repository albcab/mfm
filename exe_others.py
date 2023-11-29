import logging
import time

import jax
import jax.numpy as jnp

import numpy as np

import wandb

from mcmc_utils import stein_disc, max_mean_disc

from exe_flow_matching import plot_contours

import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


def run(dist, args, target_gn=None):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    learning_iter = args.learning_iter
    n_iter = args.eval_iter
    n_chain = args.num_chain
    key_target, key_sample, key_init, key_dist, key_fourier, key_gen = jax.random.split(jax.random.PRNGKey(args.seed), 6)
    dist.initialize_model(key_dist, n_chain)

    if target_gn is not None:
        key_gen, key_loss = jax.random.split(key_target)
        keys_target = jax.random.split(key_gen, n_iter * n_chain)
        real_samples = jax.vmap(target_gn)(keys_target)

    
    if args.do_smc:
        from blackjax.smc import adaptive_tempered, resampling
        from blackjax.mcmc import mala

        tempered = adaptive_tempered.adaptive_tempered_smc(
            dist.logprior,
            dist.loglik,
            mala.build_kernel(),
            mala.init,
            dict(step_size=args.step_size),
            resampling.systematic,
            args.alpha,
            num_mcmc_steps=args.anneal_iter // args.num_anneal_temp,
        )

        @jax.jit
        def one_step(state, key):
            state, info = tempered.step(key, state)
            return state, (state, info)
        
        keys = jax.random.split(jax.random.PRNGKey(args.seed), learning_iter)
        init_state = tempered.init(dist.init_params)
        train_start = time.time()
        state, _ = jax.lax.scan(one_step, init_state, keys)
        train_time = time.time() - train_start
        print("Final temp=", state.lmbda)

        keys = jax.random.split(keys[0], n_iter)
        _, (states, infos) = jax.lax.scan(one_step, state, keys)
        flow_samples = states.particles.reshape((n_chain * n_iter, args.dim)) #not really flow but MCMC
        exact_samples = states.particles.reshape((n_chain * n_iter, args.dim)) #not really flow but MCMC


    elif args.do_flowmc:
        from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
        from flowMC.sampler.MALA import MALA
        from flowMC.sampler.Sampler import Sampler
        from flowMC.utils.PRNG_keys import initialize_rng_keys
        
        n_layers = len(args.hidden_x) + len(args.hidden_t)
        model = MaskedCouplingRQSpline(args.dim, n_layers, args.hidden_xt, args.dim * n_layers, key_init)
        flowmc_logprob = lambda x, data=None: dist.logprob(x)
        MALA_Sampler = MALA(flowmc_logprob, True, {"step_size": args.step_size})

        rng_key_set = initialize_rng_keys(n_chain, seed=args.seed)
        mcmc_per_flow_steps = int(args.mcmc_per_flow_steps)
        nf_sampler = Sampler(
            args.dim,
            rng_key_set,
            jnp.zeros(args.dim),
            MALA_Sampler,
            model,
            n_loop_training=learning_iter // mcmc_per_flow_steps,
            n_loop_production=0,
            n_local_steps=mcmc_per_flow_steps,
            n_global_steps=mcmc_per_flow_steps,
            n_chains=n_chain,
            n_epochs=mcmc_per_flow_steps,
            learning_rate=args.learning_rate,
            max_samples=n_chain * (mcmc_per_flow_steps + 1),
            batch_size=n_chain,
            use_global=True,
        )

        train_start = time.time()
        nf_sampler.sample(dist.init_params, None)
        train_time = time.time() - train_start

        out_train = nf_sampler.get_sampler_state(training=True)
        global_accs = np.array(out_train['global_accs'])
        local_accs = np.array(out_train['local_accs'])
        loss_vals = np.array(out_train['loss_vals'])

        targ, lss = [], []
        i=0
        for acc in local_accs:
            for a in acc:
                i += 1
                targ.append([i, a])
        i=0
        for loss in loss_vals:
            for l in loss:
                i +=1 
                lss.append([i, l])
        targ_cols = ["iter", "local acceptance"]
        targ_name = 'acc'
        targ_table = wandb.Table(targ_cols, targ)
        wandb.log({targ_name: wandb.plot.line(targ_table, *targ_cols, title=targ_name)})
        lss_cols = ["iter", "loss"]
        lss_name = 'loss'
        lss_table = wandb.Table(lss_cols, lss)
        wandb.log({lss_name: wandb.plot.line(lss_table, *lss_cols, title=lss_name)})

        # u = jax.vmap(jax.random.normal)(jax.random.split(key_gen, n_iter * n_chain))
        # key_hutch, key_choice = jax.random.split(key_gen)
        # flow_samples, vols = jax.vmap(lambda u: nf_sampler.nf_model.forward(u))(u)
        # samples_logdensity = jax.vmap(dist.logprob)(flow_samples)
        # weights = jax.vmap(lambda logdensity, ref_sample, vol: jnp.exp(logdensity + .5 * jnp.dot(ref_sample, ref_sample) - vol))(samples_logdensity, u, vols)
        # exact_samples = jax.random.choice(key_choice, flow_samples, (n_iter * n_chain,), p=weights)

        flow_samples = nf_sampler.sample_flow(n_iter * n_chain)
        log_prob_flow = nf_sampler.evalulate_flow(flow_samples)
        samples_logdensity = jax.vmap(dist.logprob)(flow_samples)
        weights = jax.vmap(lambda logdensity, logp_flow: jnp.exp(logdensity - logp_flow))(samples_logdensity, log_prob_flow)
        key_hutch, key_choice = jax.random.split(key_gen)
        exact_samples = jax.random.choice(key_choice, flow_samples, (n_iter * n_chain,), p=weights)


    elif args.do_pocomc:
        import pocomc as pc
        
        np.random.seed(args.seed)

        n_layers = len(args.hidden_x) + len(args.hidden_t)
        sampler = pc.Sampler(
            n_chain,
            args.dim,
            lambda x: np.array(jax.vmap(dist.loglik)(x)),
            lambda x: np.array(jax.vmap(dist.logprior)(x)),
            vectorize_likelihood=False,
            vectorize_prior=False,
            infer_vectorization=False,
            bounds=None,
            flow_config={'n_blocks': n_layers, 'hidden_size': args.hidden_xt[0], 'n_hidden': n_layers // 2, 'batch_norm': False, 'activation': args.non_linearity, 'input_order': 'sequential', 'flow_type': 'maf'},
            train_config={'validation_split': 0.2, 'epochs': learning_iter, 'batch_size': n_chain, 'patience': 30, 'monitor': 'val_loss', 'shuffle': True, 'lr': [args.learning_rate * (.1) ** i for i in range(4)], 'weight_decay': args.weight_decay, 'clip_grad_norm': args.gradient_clip, 'laplace_prior_scale': 0.2, 'gaussian_prior_scale': None, 'device': 'cpu', 'verbose': 0},
        )
        train_start = time.time()
        sampler.run(np.array(dist.init_params))
        train_time = time.time() - train_start

        sampler.add_samples(n_iter * n_chain)
        results = sampler.results
        print("Iter=", results['iter'])
        acc = results['accept']
        loss_vals = results['logz']

        targ, lss = [], []
        for i, (tl, ls) in enumerate(zip(acc, loss_vals)):
            targ.append([i, tl])
            lss.append([i, ls.mean()])
        targ_cols = ["iter", "acceptance"]
        targ_name = 'acc'
        targ_table = wandb.Table(targ_cols, targ)
        wandb.log({targ_name: wandb.plot.line(targ_table, *targ_cols, title=targ_name)})
        lss_cols = ["iter", "loss"]
        lss_name = 'loss'
        lss_table = wandb.Table(lss_cols, lss)
        wandb.log({lss_name: wandb.plot.line(lss_table, *lss_cols, title=lss_name)})

        flow_samples = results['samples'] #not really flow but MCMC
        exact_samples = results['samples'] #not really flow but MCMC


    elif args.do_dds:
        from dds.configs.config import set_task, get_config
        from dds.train_dds import train_dds
    
        config = get_config()
        config.model.tfinal = 6.4
        config.model.dt = 0.05
        config = set_task(config, "mixture_well")
        config.model.reference_process_key = "oudstl"
        config.model.step_scheme_key = "cos_sq"

        config.model.input_dim = args.dim
        config.trainer.lnpi = lambda x: jax.vmap(dist.logprob)(x)
        config.model.target = lambda x: jax.vmap(dist.logprob)(x)
        
        # Opt setting for funnel
        config.model.sigma = 1.075
        config.model.alpha = 0.6875
        config.model.m = 1.0
            
        # Path opt settings    
        config.model.exp_dds = True

        config.model.stl = False
        config.model.detach_stl_drift = False

        # config.trainer.notebook = True
        # config.trainer.epochs = 11000
        # Opt settings we use
        # config.trainer.learning_rate = 0.0001
        config.trainer.learning_rate = 5 * 10**(-3)
        config.trainer.lr_sch_base_dec = 0.99 # For funnel

        config.trainer.epochs = learning_iter #2500
        config.trainer.random_seed = args.seed
        config.model.fully_connected_units = args.hidden_xt
        config.model.batch_size = n_chain #300  # 128
        config.model.elbo_batch_size = n_chain #2000
        config.trainer.timer = False
        config.eval.seeds = 0 #30
        train_start = time.time()
        out_dict = train_dds(config)
        train_time = time.time() - train_start

        name = "aug"
        print(out_dict[-1][name].shape)
        # flow_samples = out_dict[-1]["aug_ode"][:, -1,:args.dim]
        flow_samples = out_dict[-1][name][:, -1, :args.dim]
        energy_cost_dt = out_dict[-1][name][:, -1, -1]
        stl = out_dict[-1][name][:, -1, args.dim]
        terminal_cost = config.model.terminal_cost(flow_samples, 
            config.trainer.lnpi, config.model.sigma, config.model.tfinal, 
            "brown" in str(config.model.reference_process_dict[config.model.reference_process_key]).lower())
        weights = jnp.exp(-energy_cost_dt - terminal_cost - stl)
        key_hutch, key_choice = jax.random.split(key_gen)
        exact_samples = jax.random.choice(key_choice, flow_samples, (out_dict[-1][name].shape[0],), p=weights)

    
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
    logpdf_ = jax.vmap(dist.logprob)(exact_samples).sum()
    print("Logpdf of exact samples=", logpdf_)
    stein_ = stein_disc(exact_samples, dist.logprob)
    print("Stein U, V disc of exact samples=", stein_[0], stein_[1])
    data = [args.mcmc_per_flow_steps, args.learning_iter, train_time, logpdf, logpdf_, stein[0], stein_[0], stein[1], stein_[1]]
    columns = ["mcmc/flow", "learn iter", "train time", "logpdf", "logpdf*", "KSD U-stat", "KSD U-stat*", "KSD V-stat", "KSD V-stat*"]

    if target_gn is not None:
        mmd = max_mean_disc(real_samples, flow_samples)
        print("Max mean disc of flow samples=", mmd)
        data.append(mmd)
        columns.append("MMD")
        mmd_ = max_mean_disc(real_samples, exact_samples)
        print("Max mean disc of exact samples=", mmd_)
        data.append(mmd_)
        columns.append("MMD*")
        print()
        
    # #fields
    # fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # ax[1].set_title(r"$\hat{\phi}$")
    # ax[1].set_xlabel(r"$d$")
    # ax[1].set_ylabel(r"$\phi$")
    # flow_samples = jnp.pad(flow_samples, ((0, 0), (1, 1))) #for the phi-four example
    # for i in range(flow_samples.shape[0]):
    #     ax[1].plot(flow_samples[i], color='red', alpha=0.1)
    # ax[0].set_title(r"$\pi$")
    # ax[0].set_xlabel(r"$d$")
    # ax[0].set_ylabel(r"$\phi$")
    # exact_samples = jnp.pad(exact_samples, ((0, 0), (1, 1))) #for the phi-four example
    # for i in range(exact_samples.shape[0]):
    #     ax[0].plot(exact_samples[i], color='red', alpha=0.1)
    # # plt.setp(ax, xlim=[0, args.dim + 1], ylim=args.lim)
    # data.append(wandb.Image(fig))
    # columns.append("plot phi")
    # plt.close()
    
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
        plt.setp(ax, xlim=args.lim, ylim=args.lim)
        if args.dim == 2:
            plot_contours(dist.logprob, ax, args)
        data.append(wandb.Image(fig))
        columns.append("plot (x0,x" + str(i+1) + ")")
        plt.close()
        if i > 8:
            break #only the first 10 dimensions

    wandb.log({"summary": wandb.Table(columns, [data])})
    wandb.finish()
    return None
