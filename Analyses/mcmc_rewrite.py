import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pickle
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

def run_emcee(
    key,
    pathogen,
    lockdown,
    option1,
    option2,
    seed,
    NAG=7,
    startx=None,
    initial_pos=None,
    prefix="",
    num_walkers=32,
    spread=0.03,
    sigma=1e-5,
    num_steps=1000,
    census_age_pop=None,
    pool=None,
):
    if startx is None:
        _, startx, _ = load_optimization_results(prefix, pathogen, seed, lockdown, option1, option2)
    # initialize walkers randomly within 3% of parameter values, within bounds
    _, bounds = parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)
    lower_bounds = jnp.asarray(bounds[:, 0])
    upper_bounds = jnp.asarray(bounds[:, 1])
    startx = jnp.asarray(startx)
    if startx.ndim > 1:
        # When resuming, callers may pass walker positions; use one parameter vector for shape metadata.
        startx = startx[0]

    if initial_pos is None:
        # generate perturbations for all walkers
        key, subkey = jax.random.split(key)
        perturbations = jax.random.uniform(subkey, shape=(num_walkers, len(startx)), minval=-spread, maxval=spread) * startx
        candidates = startx[jnp.newaxis, :] + perturbations
        # ensure candidates are within bounds
        initial_pos = jnp.clip(candidates, lower_bounds + 1e-5, upper_bounds - 1e-5)
    else:
        initial_pos = jnp.asarray(initial_pos)
        initial_pos = jnp.clip(initial_pos, lower_bounds + 1e-5, upper_bounds - 1e-5)

    n_dim = int(initial_pos.shape[-1])
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
        n_dim,
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

def _load_saved_chain(sample_path, log_prob_path, n_walkers):
    if not os.path.exists(sample_path) or not os.path.exists(log_prob_path):
        return None, None

    samples = np.genfromtxt(sample_path, delimiter=',')
    log_prob = np.genfromtxt(log_prob_path, delimiter=',')

    samples = np.atleast_2d(samples)
    log_prob = np.atleast_2d(log_prob)

    if samples.size == 0 or log_prob.size == 0:
        return None, None

    samples = samples.reshape(-1, n_walkers, samples.shape[-1])
    log_prob = log_prob.reshape(-1, n_walkers)
    return samples, log_prob

def run_burn_in_for_pathogen(key, pathogen, lockdown, option1, option2, seed, NAG, prefix, n_walkers, burn_in_size, census_age_pop, pool):
    sampler = run_emcee(
        key,
        pathogen,
        lockdown,
        option1,
        option2,
        seed,
        NAG=NAG,
        prefix=prefix,
        spread=3e-2,
        sigma=1e-4,
        num_walkers=n_walkers,
        num_steps=burn_in_size,
        census_age_pop=census_age_pop,
        pool=pool,
    )
    return sampler

def load_refined_chain_or_burnin(burnin_sample_path, burnin_log_prob_path, refined_sample_path, refined_log_prob_path, n_walkers, chunk_size):
    burnin_samples = np.genfromtxt(burnin_sample_path, delimiter=',')
    burnin_samples = np.atleast_2d(burnin_samples)
    burnin_samples = burnin_samples.reshape(-1, n_walkers, burnin_samples.shape[-1])
    burnin_log_prob = np.genfromtxt(burnin_log_prob_path, delimiter=',')
    burnin_log_prob = np.atleast_2d(burnin_log_prob).reshape(-1, n_walkers)

    saved_samples, saved_log_prob = _load_saved_chain(refined_sample_path, refined_log_prob_path, n_walkers)
    if saved_samples is None:
        return burnin_samples, burnin_log_prob, 0

    completed_chunks = saved_samples.shape[0] // chunk_size
    return saved_samples, saved_log_prob, completed_chunks

if __name__ == "__main__":
    # seed = 260602
    lockdown = "ExponentialODipLinear"
    option1 = "dedupsac"
    NAG = 7
    prefix = ""
    from Parameters.census_population import CENSUS_AGE_POP_sac as CENSUS_AGE_POP

    n_walkers = 64
    burn_in_size = 500

    pathogens = ["Metapneumovirus",]
    option2s = ["maxagep015",]
    seeds = [260603,]

    pools = {}
    for pathogen, option2, seed in zip(pathogens, option2s, seeds):
        pool = Pool(
            processes=64//2, 
            initializer=_init_worker, 
            initargs=(pathogen, lockdown, option1, option2, NAG, CENSUS_AGE_POP)
        )
        pools[pathogen] = pool

    for pathogen, option2, seed in zip(pathogens, option2s, seeds):
        key = jax.random.PRNGKey(260601)
        burnin_sample_path = f"Outputs/mcmc_samples_DEmove_{prefix}{pathogen}_{lockdown}_{option1}_{option2}_{seed}_burnin.csv"
        burnin_log_prob_path = f"Outputs/mcmc_log_prob_DEmove_{prefix}{pathogen}_{lockdown}_{option1}_{option2}_{seed}_burnin.csv"

        if os.path.exists(burnin_sample_path) and os.path.exists(burnin_log_prob_path):
            print(f"Found existing burn-in files for {pathogen}. Skipping burn-in and moving to refinement.")
            continue

        burnin_sampler = run_burn_in_for_pathogen(
            key,
            pathogen,
            lockdown,
            option1,
            option2,
            seed,
            NAG,
            prefix,
            n_walkers,
            burn_in_size,
            CENSUS_AGE_POP,
            pools[pathogen],
        )
        acceptance_fraction = np.mean(burnin_sampler.acceptance_fraction)
        print(f"Acceptance fraction: {acceptance_fraction:.4f}")
        samples = burnin_sampler.get_chain()
        log_prob = burnin_sampler.get_log_prob()
        np.savetxt(burnin_sample_path, samples.reshape(-1, samples.shape[-1]), delimiter=',')
        np.savetxt(burnin_log_prob_path, log_prob, delimiter=',')
        param_names, _ = parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)
        plot_traces(samples, param_names, pathogen, lockdown, option1, option2, seed)

    n_samples = 100000
    chunk_size = 1000
    total_chunks = n_samples // chunk_size

    refined_state = {}

    for pathogen, option2, seed in zip(pathogens, option2s, seeds):
        key = jax.random.PRNGKey(260601)
        burnin_sample_path = f"Outputs/mcmc_samples_DEmove_{prefix}{pathogen}_{lockdown}_{option1}_{option2}_{seed}_burnin.csv"
        burnin_log_prob_path = f"Outputs/mcmc_log_prob_DEmove_{prefix}{pathogen}_{lockdown}_{option1}_{option2}_{seed}_burnin.csv"
        refined_sample_path = f"Outputs/mcmc_samples_DEmove_{prefix}{pathogen}_{lockdown}_{option1}_{option2}_{seed}_refined.csv"
        refined_log_prob_path = f"Outputs/mcmc_log_prob_DEmove_{prefix}{pathogen}_{lockdown}_{option1}_{option2}_{seed}_refined.csv"

        current_samples, current_log_prob, completed_chunks = load_refined_chain_or_burnin(
            burnin_sample_path,
            burnin_log_prob_path,
            refined_sample_path,
            refined_log_prob_path,
            n_walkers,
            chunk_size,
        )

        if completed_chunks == 0:
            print(f"Starting refined chain from burn-in for {pathogen}.")
        else:
            print(f"Resuming refined chain for {pathogen} from chunk {completed_chunks}.")

        refined_state[pathogen] = {
            "key": key,
            "option2": option2,
            "burnin_sample_path": burnin_sample_path,
            "burnin_log_prob_path": burnin_log_prob_path,
            "refined_sample_path": refined_sample_path,
            "refined_log_prob_path": refined_log_prob_path,
            "current_samples": current_samples,
            "current_log_prob": current_log_prob,
            "completed_chunks": completed_chunks,
        }

    for chunk_n in range(total_chunks):
        for pathogen in pathogens:
            state = refined_state[pathogen]
            if chunk_n < state["completed_chunks"]:
                continue

            initial_pos = state["current_samples"][-1]
            psampler = run_emcee(
                state["key"],
                pathogen,
                lockdown,
                option1,
                state["option2"],
                seed,
                NAG=NAG,
                startx=initial_pos,
                initial_pos=initial_pos,
                spread=1e-4,
                sigma=1e-5,
                num_walkers=n_walkers,
                num_steps=chunk_size,
                census_age_pop=CENSUS_AGE_POP,
                pool=pools[pathogen],
            )

            acceptance_fraction = np.mean(psampler.acceptance_fraction)
            print(f"Acceptance fraction ({pathogen}, chunk {chunk_n} of {total_chunks}): {acceptance_fraction:.4f}")

            new_samples = psampler.get_chain()
            new_log_prob = psampler.get_log_prob()
            state["current_samples"] = np.concatenate([state["current_samples"], new_samples], axis=0)
            state["current_log_prob"] = np.concatenate([state["current_log_prob"], new_log_prob], axis=0)
            np.savetxt(state["refined_sample_path"], state["current_samples"].reshape(-1, state["current_samples"].shape[-1]), delimiter=',')
            np.savetxt(state["refined_log_prob_path"], state["current_log_prob"], delimiter=',')

            best_idx = np.unravel_index(np.argmax(state["current_log_prob"]), state["current_log_prob"].shape)
            best_params = np.asarray(state["current_samples"][best_idx], dtype=np.float64)
            best_neg_log_likelihood = float(-state["current_log_prob"][best_idx])
            results_dir = f"Data/Processed/results{str(seed)[:6]}"
            os.makedirs(results_dir, exist_ok=True)
            emcee_results_path = f"{results_dir}/emcee_{prefix}{pathogen}{lockdown}{option1}{state['option2']}{seed}.pickle"
            with open(emcee_results_path, "wb") as f:
                pickle.dump(
                    {
                        "final_population": np.asarray([best_params]),
                        "final_fitness": np.asarray([best_neg_log_likelihood]),
                    },
                    f,
                )

    print("Closing processing pools...")
    for p in pools.values():
        p.close()
        p.join()