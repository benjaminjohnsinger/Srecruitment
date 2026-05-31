import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import emcee

import numpyro
from numpyro import distributions as dist

import pickle
import os
import matplotlib.pyplot as plt

from utils import load_optimization_results, parameters_names_bounds
from fit_opt import get_likelihood

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
        return -nll
    
    return jax.jit(jax.vmap(log_posterior))

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

def run_emcee(key, pathogen, lockdown, option1, option2, seed, NAG=7, startx=None, prefix="", num_walkers=32, spread=0.03, sigma=1e-5, num_steps=1000):
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
    
    sampler = emcee.EnsembleSampler(num_walkers, len(startx), log_posterior, vectorize=True,
                                    moves=emcee.moves.DEMove(sigma=sigma))
    sampler.run_mcmc(initial_pos, num_steps, progress=True)
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
    seed = 260528
    lockdown = "ExponentialODipLinear"
    option1 = "dedupsplit"
    NAG = 8
    prefix = "evosax_DE"
    from Parameters.census_population import CENSUS_AGE_POP_split as CENSUS_AGE_POP
    pathogens = ["InfluenzaA", "RSV", "InfluenzaB", "Adenovirus", "Parainfluenza3", "Metapneumovirus"]
    option2s = ["maxagep04", "maxagep028", "maxagep04", "maxagep004", "maxagep007", "maxagep0085"]
    for pathogen, option2 in zip(pathogens, option2s):
        key = jax.random.PRNGKey(260530)
        if pathogen != "InfluenzaA":
            sampler = run_emcee(key, pathogen, lockdown, option1, option2, seed, NAG=NAG, prefix=prefix, spread=3e-2, sigma=1e-4, num_walkers=64, num_steps=500)
            acceptance_fraction = np.mean(sampler.acceptance_fraction)
            print(f"Acceptance fraction: {acceptance_fraction:.4f}")
            samples = sampler.get_chain()
            # save samples
            results_file = f"Outputs/mcmc_samples_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_burnin.csv"
            np.savetxt(results_file, samples.reshape(-1, samples.shape[-1]), delimiter=",")
            lobprob = sampler.get_log_prob()
            np.savetxt(f"Outputs/mcmc_log_prob_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_burnin.csv", lobprob, delimiter=",")    
        else:
            samples = np.genfromtxt(f"Outputs/mcmc_samples_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_burnin.csv", delimiter=',')
            samples = samples.reshape(-1, 64, samples.shape[-1]) # reshape to (n_iterations, n_walkers, n_params)
            lobprob = np.genfromtxt(f"Outputs/mcmc_log_prob_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_burnin.csv", delimiter=',')

        param_names, _ = parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)
        plot_traces(samples, param_names, pathogen, lockdown, option1, option2, seed)

        best_idx = np.unravel_index(np.argmax(lobprob), lobprob.shape)
        best_params = samples[best_idx]
        sampler2 = run_emcee(key, pathogen, lockdown, option1, option2, seed, NAG=NAG, startx=best_params, spread=1e-4, sigma=1e-5, num_walkers=64, num_steps=2000)
        acceptance_fraction = np.mean(sampler2.acceptance_fraction)
        print(f"Acceptance fraction (refined): {acceptance_fraction:.4f}")
        samples2 = sampler2.get_chain()
        # save refined samples
        results_file = f"Outputs/mcmc_samples_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_refined.csv"
        np.savetxt(results_file, samples2.reshape(-1, samples2.shape[-1]), delimiter=",")
        lobprob2 = sampler2.get_log_prob()
        np.savetxt(f"Outputs/mcmc_log_prob_DEmove_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_refined.csv", lobprob2, delimiter=",")    

        tau = sampler2.get_autocorr_time()
        print(f"Autocorrelation time: {tau}")
    # key = jax.random.PRNGKey(260521)
    # samples = run_nuts(key, pathogen, lockdown, option1, option2, seed, NAG, num_warmup=150, num_samples=300)
    # print("MCMC sampling completed. Sample shape:", samples["x"].shape)
    # # save samples to disk
    # results_file = f"Outputs/mcmc_samples_{pathogen}_{lockdown}_{option1}_{option2}_{seed}.csv"
    # np.savetxt(results_file, samples["x"], delimiter=",")