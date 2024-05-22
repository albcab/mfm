import abc

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


class Distribution(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def logprob(self, x1, x2):
        """defines the log probability function"""

    def logprob_fn(self, x):
        return self.logprob(**x)

    @abc.abstractmethod
    def initialize_model(self, rng_key, n_chain):
        """defines the initialization of paramters"""

    def log_prob(self, x):
        if x.ndim == 1:
            return self.logprob(x)
        else:
            assert x.ndim == 2
        return jax.vmap(self.logprob)(x)

    def sample(self, rng_key, n_samples):
        keys = jax.random.split(rng_key, n_samples)
        return jax.vmap(self.sample_model)(keys)
    
    def visualise(self, samples, axes):
        return None
    def evaluate(self, model_log_prob_fn, model_sample_and_log_prob_fn, key) -> dict:
        """Evaluate a model. Note that reverse ESS will be estimated separately, so should not be estimated here."""
        key1, key2 = jax.random.split(key)

        info = {}
        return info

    
class GaussianMixture(Distribution):
    def __init__(self,
        modes=jnp.array([5. * jnp.ones(2), 0. * jnp.ones(2)]),
        covs=jnp.array([.5 * jnp.eye(2), .5 * jnp.eye(2)]),
        weights=jnp.array([.7, .3]),
    ) -> None:
        self.dim = modes[0].shape[0]
        self.modes = modes
        self.covs = covs
        self.chol_covs = jax.vmap(jnp.sqrt)(covs) #jax.vmap(jnp.linalg.cholesky)(covs)
        self.weights = weights
        self.dim = 2
        self.log_Z = 0.0
        self.n_plots = 0
        self.can_sample = False
    
    def logprob(self, x):
        pdfs = jax.vmap(lambda m, c, w: w * norm.pdf(x, m, c).prod())(self.modes, self.chol_covs, self.weights)
        # pdfs = jax.vmap(lambda m, c, w: w * multivariate_normal.pdf(x, m, c))(self.modes, self.covs, self.weights)
        return jnp.log(pdfs.sum())
    
    def loglik(self, x):
        return self.logprob(x)
    
    def logprior(self, x):
        return 0.
    
    def initialize_model(self, rng_key, n_chain):
        keys = jax.random.split(rng_key, n_chain)
        self.init_params = jax.vmap(lambda k: jax.random.normal(k, (self.dim,)))(keys)
    
    def sample_model(self, rng_key):
        key_choice, key_dist = jax.random.split(rng_key)
        choice = jax.random.choice(key_choice, len(self.modes), p=self.weights)
        return self.modes.at[choice].get() + self.chol_covs.at[choice].get() * jax.random.normal(key_dist, (self.dim,))
        # return self.modes.at[choice].get() + self.chol_covs.at[choice].get() @ jax.random.normal(key_dist, (self.dim,))
    

class IndepGaussian(Distribution):
    def __init__(self, dim, mean=0., var=1.) -> None:
        self.dim = dim
        self.std = jnp.sqrt(var)
        self.mean = mean
        self.log_Z = 0.0
        self.n_plots = 0
        self.can_sample = False

    def logprob(self, x):
        return norm.logpdf(x, self.mean, self.std).sum()
    
    def initialize_model(self, rng_key, n_chain):
        keys = jax.random.split(rng_key, n_chain)
        self.init_params = jax.vmap(lambda k: jax.random.normal(k, (self.dim,)))(keys)

    def sample_model(self, rng_key):
        return self.mean + self.std * jax.random.normal(rng_key, (self.dim,))
    

class FlatDistribution(Distribution):
    def __init__(self) -> None:
        pass

    def logprob(self, x):
        return 0.
    
    def initialize_model(self, rng_key, n_chain):
        pass

    def sample_model(self, rng_key):
        pass


class PhiFour(Distribution):
    def __init__(self,
        dim, a=.1, beta=20.,
        bc=('dirichlet', 0),
        tilt=None
    ) -> None:
        self.a = a
        self.beta = beta
        self.dim = dim
        self.log_Z = 0.0
        self.n_plots = 0
        self.can_sample = False

        self.bc = bc
        assert self.bc[0] == "dirichlet" or self.bc[0] == "pbc"
        self.tilt = tilt

    def V(self, x):
        coef = self.a * self.dim
        diffs = 1. - jnp.square(x)
        V = jnp.dot(diffs, diffs) / 4 / coef
        if self.tilt is not None: 
            tilt = (self.tilt['val'] - x.mean(self.sum_dims)) ** 2 
            tilt = self.tilt["lambda"] * tilt / (4 * self.dim)
            V += tilt
        return V
    
    def U(self, x):

        if self.bc[0] == 'dirichlet':
            x_ = jnp.pad(x, pad_width=1, mode='constant', constant_values=self.bc[1])
        elif self.bc[0] == 'pbc':
            x_ = jnp.pad(x, pad_width=(1, 0), mode='wrap')

        diffs = x_[1:] - x_[:-1]
        grad_term = jnp.dot(diffs, diffs) / 2
        coef = self.a * self.dim
        return grad_term * coef

    def logprob(self, x):
        return self.loglik(x) + self.logprior(x)
    
    def loglik(self, x):
        return -self.beta * (self.U(x) + self.V(x))
    
    def logprior(self, x):
        return 0.
    
    def initialize_model(self, rng_key, n_chain):
        keys = jax.random.split(rng_key, n_chain)
        self.init_params = jax.vmap(lambda k: jax.random.uniform(k, (self.dim,)) * 2 - 1)(keys)
        # self.init_params = jax.vmap(lambda k: jax.random.normal(k, (self.dim,)))(keys)


class PhiFourBase(Distribution):
    def __init__(self, 
        dim, alpha=.1, beta=20., 
        prior_type='coupled', dim_phys=1,
    ):
        # Build the prior
        self.dim = dim
        self.prior_type = prior_type
        self.log_Z = 0.0
        self.n_plots = 0
        self.can_sample = False

        if prior_type == 'coupled':
            self.beta_prior = beta
            self.coef = alpha * dim
            prec = jnp.eye(dim) * (3 * self.coef + 1 / self.coef)
            prec -= self.coef * jnp.triu(jnp.triu(jnp.ones_like(prec), k=-1).T, k=-1)
            prec = beta * prec

        elif prior_type == 'coupled_pbc':
            self.beta_prior = beta
            dim_grid = dim / dim_phys
            eps = 0.1
            quadratic_coef = 4 + eps
            sub_prec = (1 + quadratic_coef) * jnp.eye(dim_grid)
            sub_prec -= jnp.triu(jnp.triu(jnp.ones_like(sub_prec), k=-1).T, k=-1)
            sub_prec[0, -1] = - 1  # pbc
            sub_prec[-1, 0] = - 1  # pbc

            if dim_phys == 1:
                prec = beta * sub_prec

            elif dim_phys == 2:
                # interation along one axis
                prec = jax.scipy.linalg.block_diag(*(sub_prec for d in range(dim_grid)))
                # interation along second axis
                diags = jnp.triu(jnp.triu(jnp.ones_like(prec), k=-dim_grid).T, k=-dim_grid)
                diags -= jnp.triu(jnp.triu(jnp.ones_like(prec), k=-dim_grid+1).T, k=-dim_grid+1)
                prec -= diags
                prec[:dim_grid, -dim_grid:] = - jnp.eye(dim_grid)  # pbc
                prec[-dim_grid:, :dim_grid] = - jnp.eye(dim_grid)  # pbc
                prec = beta * prec

        self.prior_prec = prec
        slogdet = jnp.linalg.slogdet(prec)
        self.prior_log_det = - slogdet[0] * slogdet[1]
        prec_chol = jax.scipy.linalg.cholesky(prec, lower=True)
        self.prior_chol_cov = jax.scipy.linalg.solve_triangular(prec_chol, jnp.eye(dim), lower=True).T

    def logprob(self, value):
        prior_ll = - 0.5 * value @ self.prior_prec @ value
        prior_ll -= 0.5 * (self.dim * jnp.log(2 * jnp.pi) + self.prior_log_det)
        return prior_ll

    def sample_model(self, rng_key):
        return self.prior_chol_cov @ jax.random.normal(rng_key, (self.dim,))
    
    def initialize_model(self, rng_key, n_chain):
        pass


import cox_process_utils as cp_utils
import numpy as np
class LogGaussianCoxPines(Distribution):

    def __init__(self, dim, file_path="finpines.csv", use_whitened=False):

        # Discretization is as in Controlled Sequential Monte Carlo
        # by Heng et al 2017 https://arxiv.org/abs/1708.08396
        num_dim = dim
        self.dim = dim
        self.log_Z = 0.0
        self.n_plots = 0
        self.can_sample = False
        self._num_latents = num_dim
        self._num_grid_per_dim = int(np.sqrt(num_dim))

        bin_counts = jnp.array(
            cp_utils.get_bin_counts(self.get_pines_points(file_path),
                                    self._num_grid_per_dim))

        self._flat_bin_counts = jnp.reshape(bin_counts, (self._num_latents))

        # This normalizes by the number of elements in the grid
        self._poisson_a = 1./self._num_latents
        # Parameters for LGCP are as estimated in Moller et al, 1998
        # "Log Gaussian Cox processes" and are also used in Heng et al.

        self._signal_variance = 1.91
        self._beta = 1./33

        self._bin_vals = cp_utils.get_bin_vals(self._num_grid_per_dim)

        def short_kernel_func(x, y):
            return cp_utils.kernel_func(x, y, self._signal_variance,
                                        self._num_grid_per_dim, self._beta)

        self._gram_matrix = cp_utils.gram(short_kernel_func, self._bin_vals)
        self._cholesky_gram = jnp.linalg.cholesky(self._gram_matrix)
        self._white_gaussian_log_normalizer = -0.5 * self._num_latents * jnp.log(
            2. * jnp.pi)

        half_log_det_gram = jnp.sum(jnp.log(jnp.abs(jnp.diag(self._cholesky_gram))))
        self._unwhitened_gaussian_log_normalizer = -0.5 * self._num_latents * jnp.log(
            2. * jnp.pi) - half_log_det_gram
        # The mean function is a constant with value mu_zero.
        self._mu_zero = jnp.log(126.) - 0.5*self._signal_variance

        if use_whitened:
            self.logprior = self.whitened_prior_log_density
            self.loglik = self.whitened_likelihood_log_density
        else:
            self.logprior = self.unwhitened_prior_log_density
            self.loglik = self.unwhitened_likelihood_log_density

    def get_pines_points(self, file_path):
        """Get the pines data points."""
        with open(file_path, "rt") as input_file:
            b = np.genfromtxt(input_file, delimiter=",")
        return b

    def whitened_prior_log_density(self, white):
        quadratic_term = -0.5 * jnp.sum(white**2)
        return self._white_gaussian_log_normalizer + quadratic_term
        
    def whitened_likelihood_log_density(self, white):
        latent_function = cp_utils.get_latents_from_white(white, self._mu_zero,
                                                        self._cholesky_gram)
        return cp_utils.poisson_process_log_likelihood(
            latent_function, self._poisson_a, self._flat_bin_counts)

    def unwhitened_prior_log_density(self, latents):
        white = cp_utils.get_white_from_latents(latents, self._mu_zero,
                                                self._cholesky_gram)
        return -0.5 * jnp.sum(
            white * white) + self._unwhitened_gaussian_log_normalizer
        
    def unwhitened_likelihood_log_density(self, latents):
        return cp_utils.poisson_process_log_likelihood(
            latents, self._poisson_a, self._flat_bin_counts)
    
    def logprob(self, x):
        return self.loglik(x) + self.logprior(x)
    
    def initialize_model(self, rng_key, n_chain):
        keys = jax.random.split(rng_key, n_chain)
        self.init_params = jax.vmap(lambda k: self._mu_zero + self._cholesky_gram @ jax.random.normal(k, (self._num_latents,)))(keys)
