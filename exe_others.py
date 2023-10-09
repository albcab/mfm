import logging
import time

import jax

import numpy as np

import wandb

from mcmc_utils import stein_disc, max_mean_disc

from distributions import FlatDistribution

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


    if args.do_flowmc:
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
            jax.numpy.zeros(args.dim),
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
        print("Train time=", time.time() - train_start)

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

        flow_samples = nf_sampler.sample_flow(n_iter * n_chain)

    elif args.do_pocomc:
        import pocomc as pc
        
        np.random.seed(args.seed)

        n_layers = len(args.hidden_x) + len(args.hidden_t)
        sampler = pc.Sampler(
            n_chain,
            args.dim,
            lambda x: np.array(dist.logprob(x)),
            FlatDistribution().logprob,
            vectorize_likelihood=True,
            vectorize_prior=True,
            bounds=None,
            flow_config={'n_blocks': n_layers, 'hidden_size': args.hidden_xt[0], 'n_hidden': n_layers // 3, 'batch_norm': False, 'activation': args.non_linearity, 'input_order': 'sequential', 'flow_type': 'maf'},
            train_config={'validation_split': 0.2, 'epochs': learning_iter, 'batch_size': n_chain, 'patience': 30, 'monitor': 'val_loss', 'shuffle': True, 'lr': [args.learning_rate * (.1) ** i for i in range(4)], 'weight_decay': args.weight_decay, 'clip_grad_norm': args.gradient_clip, 'laplace_prior_scale': 0.2, 'gaussian_prior_scale': None, 'device': 'cpu', 'verbose': 0},
        )
        train_start = time.time()
        sampler.run(np.array(dist.init_params))
        print("Train time=", time.time() - train_start)

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

        flow_samples = results['samples']


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
        config.trainer.lnpi = dist.logprob
        config.model.target = dist.logprob
        
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
        out_dict = train_dds(config)

        print(out_dict[-1]["aug"].shape)
        # flow_samples = out_dict[-1]["aug_ode"][:, -1,:args.dim]
        flow_samples = out_dict[-1]["aug"][:, -1,:args.dim]

    
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