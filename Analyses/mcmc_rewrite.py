import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from multiprocessing import Pool

import emcee

import numpyro
from numpyro import distributions as dist

import matplotlib.pyplot as plt

from utils import load_optimization_results, parameters_names_bounds
from fit_opt import get_likelihood

# --- ADD THESE GLOBALS FOR MULTIPROCESSING WORKERS ---
_worker_log_posterior = None
CENSUS_AGE_POP = None

def _init_worker(pathogen, lockdown, option1, option2, NAG, census_age_pop):
    """This runs once on each worker process when the Pool starts up."""
    global _worker_log_posterior, CENSUS_AGE_POP
    CENSUS_AGE_POP = census_age_pop
    # Each process compiles its own local JIT version of the heavy ODE model
    _worker_log_posterior = get_emcee_model(pathogen, lockdown, option1, option2, NAG)

def _worker_log_prob_wrapper(x):
    """A top-level, perfectly picklable function that workers can call."""
    return _worker_log_posterior(x)
# -----------------------------------------------------

def get_nuts_model(pathogen, lockdown, option1, option2, xDE, NAG=7):
    _, bounds = parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)
    likelihood, _ = get_likelihood(pathogen, lockdown, option1, option2, 1e-9, normalize=False, NAG=NAG, CENSUS_AGE_POP=CENSUS_AGE_POP)
    # calculate priors based on bounds, and get ready to work in transformed space
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    de_deltas = jnp.maximum(xDE - lower_bounds, 1e-6)
    mu = jnp.log(de_deltas)
    uncontrained_upper = jnp.log(upper_bounds - lower_bounds)
    sigma = jnp.maximum(0.5, jnp.abs(uncontrained_upper - mu) / 2.0) # upper bound is roughly 2 std above the mean, but enforce a minimum sigma to prevent numerical issues
    
    def model():
        lognormal_prior = dist.TransformedDistribution(
            dist.LogNormal(loc=mu, scale=sigma),
            dist.transforms.AffineTransform(loc=lower_bounds, scale=1.0)
        )
        x = numpyro.sample("x", lognormal_prior)
        nll = likelihood(x)
        numpyro.factor("likelihood", -nll)
    return model

def get_emcee_model(pathogen, lockdown, option1, option2, NAG=7):
    _, bounds = parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)
    likelihood, _ = get_likelihood(pathogen, lockdown, option1, option2, 1e-9, normalize=False, NAG=NAG, CENSUS_AGE_POP=CENSUS_AGE_POP)
    # calculate priors based on bounds
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    
    def log_posterior(x):
        in_bounds = jnp.all(jnp.logical_and(x >= lower_bounds, x <= upper_bounds))
        nll = jax.lax.cond(in_bounds, lambda p: likelihood(p), lambda p: jnp.inf, x)
        nll = jnp.where(jnp.isnan(nll), jnp.inf, nll)
        return -nll
    
    return jax.jit(log_posterior)

def run_nuts(key, pathogen, lockdown, option1, option2, seed, NAG=7, num_warmup=500, num_samples=1000, num_chains=1):
    _, xDE, _ = load_optimization_results("", pathogen, seed, lockdown, option1, option2)
    # force 64 bit precision on xDE
    xDE = jnp.asarray(xDE, dtype=jnp.float64)
    model = get_nuts_model(pathogen, lockdown, option1, option2, xDE, NAG)
    nuts_kernel = numpyro.infer.NUTS(model,
                                     init_strategy=numpyro.infer.init_to_value(values={"x": xDE}),
                                     max_tree_depth=6,
                                     dense_mass=True,)
    mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples,
                              num_chains=num_chains, chain_method="parallel")
    mcmc.run(key)
    mcmc.print_summary()
    return mcmc.get_samples()

def run_emcee(key, pathogen, lockdown, option1, option2, seed, NAG=7, startx=None, prefix="", num_walkers=32, spread=0.03, sigma=1e-5, num_steps=1000, census_age_pop=None, pool=None):
    if startx is None:
        _, startx, _ = load_optimization_results(prefix, pathogen, seed, lockdown, option1, option2)
    log_posterior = get_emcee_model(pathogen, lockdown, option1, option2, NAG)
    # initialize walkers randomly within 3% of parameter values, within bounds
    _, bounds = parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)
    lower_bounds = jnp.asarray(bounds[:, 0])
    upper_bounds = jnp.asarray(bounds[:, 1])
    startx = jnp.asarray(startx)
    
    # generate perturbations for all walkers
    key, subkey = jax.random.split(key)
    perturbations = jax.random.uniform(subkey, shape=(num_walkers, len(startx)), minval=-spread, maxval=spread) * startx
    candidates = startx[jnp.newaxis, :] + perturbations
    # ensure candidates are within bounds
    initial_pos = jnp.clip(candidates, lower_bounds + 1e-5, upper_bounds - 1e-5)
    # # loop to resample candidates that give -inf log posterior until all walkers have valid initial positions
    # log_posteriors = log_posterior(initial_pos)
    # while jnp.any(log_posteriors == -jnp.inf):
    #     invalid_mask = log_posteriors == -jnp.inf
    #     num_invalid = jnp.sum(invalid_mask)
    #     print(f"Resampling {num_invalid} invalid initial positions...")
    #     key, subkey = jax.random.split(key)
    #     perturbations = jax.random.uniform(subkey, shape=(num_invalid, len(startx)), minval=-spread, maxval=spread) * startx
    #     candidates = startx[jnp.newaxis, :] + perturbations
    #     candidates = jnp.clip(candidates, lower_bounds + 1e-5, upper_bounds - 1e-5)
    #     initial_pos = initial_pos.at[invalid_mask].set(candidates)
    #     log_posteriors = log_posterior(initial_pos)

    sampler = emcee.EnsembleSampler(
        num_walkers, 
        len(startx), 
        _worker_log_prob_wrapper,  # Pass the picklable top-level wrapper
        pool=pool
    )
    sampler.run_mcmc(initial_pos, num_steps, progress=True)

    # sampler = emcee.EnsembleSampler(num_walkers, len(startx), log_posterior, vectorize=True,
    #                                 moves=emcee.moves.DEMove(sigma=sigma))
    # sampler.run_mcmc(initial_pos, num_steps, progress=True)
    return sampler

def plot_traces(mcmc_samples, param_names, pathogen, lockdown, option1, option2, seed):
    _, n_walkers, n_params = mcmc_samples.shape
    # make directory Figures/mcmc_traces_{pathogen}_{lockdown}_{option1}_{option2}_{seed}
    os.makedirs(f"Figures/mcmc_traces_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}", exist_ok=True)
    for j in range(n_walkers):
        fig, ax = plt.subplots(4,5, figsize=(10,6))
        for i in range(n_params):
            ax[i//5, i%5].plot(mcmc_samples[:,j,i], color='k')
            ax[i//5, i%5].set_title(param_names[i])
        plt.tight_layout()
        plt.savefig(f"Figures/mcmc_traces_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}/walker_{j}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    seed = 260531
    lockdown = "ExponentialODipLinear"
    option1 = "dedupsac"
    NAG = 7
    prefix = "evosax_DE"
    from Parameters.census_population import CENSUS_AGE_POP_sac as CENSUS_AGE_POP

    n_walkers = 64
    burn_in_size = 500

    pathogens = ["RSV", "Parainfluenza3"]
    option2s = ["maxagep028", "maxagep008"]

    pools = {}
    for pathogen, option2 in zip(pathogens, option2s):
        pool = Pool(
            processes=64//2, 
            initializer=_init_worker, 
            initargs=(pathogen, lockdown, option1, option2, NAG, CENSUS_AGE_POP)
        )
        pools[pathogen] = pool

    for pathogen, option2 in zip(pathogens, option2s):
        key = jax.random.PRNGKey(260601)
        sampler = run_emcee(key, pathogen, lockdown, option1, option2, seed, NAG=NAG, prefix=prefix, spread=3e-2, sigma=1e-4, num_walkers=n_walkers, num_steps=burn_in_size,
                            census_age_pop=CENSUS_AGE_POP, pool=pools[pathogen])
        acceptance_fraction = np.mean(sampler.acceptance_fraction)
        print(f"Acceptance fraction: {acceptance_fraction:.4f}")
        samples = sampler.get_chain()
        # save samples
        results_file = f"Outputs/mcmc_samples_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_burnin.csv"
        np.savetxt(results_file, samples.reshape(-1, samples.shape[-1]), delimiter=",")
        lobprob = sampler.get_log_prob()
        np.savetxt(f"Outputs/mcmc_log_prob_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_burnin.csv", lobprob, delimiter=",")    
        param_names, _ = parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)
        plot_traces(samples, param_names, pathogen, lockdown, option1, option2, seed)

    n_samples = 30000
    chunk_size = 500

    samplers_by_pathogen = {}

    for pathogen, option2 in zip(pathogens, option2s):
        samples = np.genfromtxt(f"Outputs/mcmc_samples_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_burnin.csv", delimiter=',')
        samples = samples.reshape(-1, n_walkers, samples.shape[-1]) # reshape to (n_iterations, n_walkers, n_params)
        lobprob = np.genfromtxt(f"Outputs/mcmc_log_prob_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_burnin.csv", delimiter=',')
        best_idx = np.unravel_index(np.argmax(lobprob), lobprob.shape)
        best_params = samples[best_idx]
        psampler = run_emcee(key, pathogen, lockdown, option1, option2, seed, NAG=NAG, startx=best_params, spread=1e-4, sigma=1e-5, num_walkers=n_walkers, num_steps=chunk_size,
                             census_age_pop=CENSUS_AGE_POP, pool=pools[pathogen])
        samplers_by_pathogen[pathogen] = psampler
        acceptance_fraction = np.mean(psampler.acceptance_fraction)
        print(f"Acceptance fraction (chunk 0 of {n_samples // chunk_size}): {acceptance_fraction:.4f}")
        psamples = psampler.get_chain()
        results_file = f"Outputs/mcmc_samples_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_refined.csv"
        np.savetxt(results_file, psamples.reshape(-1, psamples.shape[-1]), delimiter=",")
        lobprob2 = psampler.get_log_prob()
        np.savetxt(f"Outputs/mcmc_log_prob_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_refined.csv", lobprob2, delimiter=",")    

    for chunk_n in range(1, n_samples // chunk_size):
        for pathogen, option2 in zip(pathogens, option2s):
            key = jax.random.PRNGKey(260601 + chunk_n)
            samples = np.genfromtxt(f"Outputs/mcmc_samples_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_refined.csv", delimiter=',')
            samples = samples.reshape(-1, n_walkers, samples.shape[-1]) # reshape to (n_iterations, n_walkers, n_params)
            logprob = np.genfromtxt(f"Outputs/mcmc_log_prob_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_refined.csv", delimiter=',')
            # extract final positions of walkers from previous chunk
            
            final_positions = samples[-1]
            psampler = samplers_by_pathogen[pathogen]
            psampler.run_mcmc(final_positions, chunk_size, progress=True)

            acceptance_fraction = np.mean(psampler.acceptance_fraction)
            print(f"Acceptance fraction (chunk {chunk_n} of {n_samples // chunk_size}): {acceptance_fraction:.4f}")
            psamples = psampler.get_chain()
            logprob = psampler.get_log_prob()
            results_file = f"Outputs/mcmc_samples_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_refined.csv"
            np.savetxt(results_file, psamples.reshape(-1, psamples.shape[-1]), delimiter=",")
            np.savetxt(f"Outputs/mcmc_log_prob_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_refined.csv", logprob, delimiter=",")    

    print("Closing processing pools...")
    for p in pools.values():
        p.close()
        p.join()