import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import emcee

import numpyro
from numpyro import distributions as dist

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

def run_emcee(key, pathogen, lockdown, option1, option2, seed, NAG=7, num_walkers=32, num_steps=1000):
    _, xDE, _ = load_optimization_results("", pathogen, seed, lockdown, option1, option2)
    log_posterior = get_emcee_model(pathogen, lockdown, option1, option2, NAG)
    # set random seed from key
    np.random.seed(int(key[0]))
    # initialize walkers in a small Gaussian ball around the DE solution
    jitter = 1.0 + 1e-4 * jax.random.normal(key, shape=(num_walkers, len(xDE)))
    initial_pos = xDE * jitter
    sampler = emcee.EnsembleSampler(num_walkers, len(xDE), log_posterior, vectorize=True)
    sampler.run_mcmc(initial_pos, num_steps, progress=True)
    tau = sampler.get_autocorr_time()
    print(f"Autocorrelation time: {tau}")
    return sampler.get_chain(flat=True)

if __name__ == "__main__":
    pathogen = "RSV"
    seed = 260505
    lockdown = "ExponentialODipEqual"
    option1 = "dedupsplit"
    option2 = "maxagep028"
    NAG = 8
    from Parameters.census_population import CENSUS_AGE_POP_split as CENSUS_AGE_POP

    key = jax.random.PRNGKey(260521)
    samples = run_emcee(key, pathogen, lockdown, option1, option2, seed, NAG, num_walkers=64, num_steps=1000)
    print("MCMC sampling completed. Sample shape:", samples.shape)
    # save samples to disk
    results_file = f"Outputs/mcmc_samples_emcee_{pathogen}_{lockdown}_{option1}_{option2}_{seed}.csv"
    np.savetxt(results_file, samples, delimiter=",")

    # key = jax.random.PRNGKey(260521)
    # samples = run_nuts(key, pathogen, lockdown, option1, option2, seed, NAG, num_warmup=150, num_samples=300)
    # print("MCMC sampling completed. Sample shape:", samples["x"].shape)
    # # save samples to disk
    # results_file = f"Outputs/mcmc_samples_{pathogen}_{lockdown}_{option1}_{option2}_{seed}.csv"
    # np.savetxt(results_file, samples["x"], delimiter=",")