import jax.numpy as jnp
import jax.scipy as jsp
import jax
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from sim_grid import age_ratio_of_rebound
from utils import date_to_t, pathogen_parameters, x_to_params
from plotting import calculate_observations_per_season_jax, kpsc_proportion_positive_incidence_plot
from fit_opt import get_likelihood, latin_hypercube_sample
from fit_MCMC import calculate_expected_obs, run_simulation
import pandas as pd

#### latin hypercube sampling of parameter space and likelihood evaluation
# BETA follows a lognormal distribution with median 0.15
def BETA_distribution(key):
    return jax.random.normal(key, shape=()) * 0.5 + np.log(0.2)

def SEASONALITY_distribution(key):
    return jax.random.normal(key, shape=()) * 0.5 + np.log(0.1)

def OFFSET_distribution(key):
    return jax.random.normal(key, shape=()) * 0.1 + 0.15

def WANE2_distribution(key):
    return jax.random.normal(key, shape=()) * 0.5 + np.log(0.005)

def REL_distribution(key):
    return jax.random.uniform(key, shape=(), minval=0.1, maxval=1.0)

def F1_distribution(key):
    return jax.random.uniform(key, shape=(), minval=0.0, maxval=1.0)

def R1_distribution(key):
    return jax.random.uniform(key, shape=(), minval=0.002, maxval=0.01)

def OBS_distribution(key):
    return jax.random.uniform(key, shape=(), minval=0.0, maxval=1.0)

def quantile_to_params(quantiles, pathogen="RSV"):
    """Convert quantiles [0,1] to parameter values using inverse CDF"""
    BETA = jnp.exp(jsp.special.erfinv(2*quantiles[0] - 1) * 0.5 + np.log(0.2))
    SEASONALITY = jnp.exp(jsp.special.erfinv(2*quantiles[1] - 1) * 0.5 + np.log(0.1))
    OFFSET = jsp.special.erfinv(2*quantiles[2] - 1) * 0.1 + 0.15
    WANE2 = jnp.exp(jsp.special.erfinv(2*quantiles[3] - 1) * 0.5 + np.log(0.005))
    S_REL1 = 0.1 + quantiles[4] * 0.9
    S_REL2 = 0.1 + quantiles[5] * 0.9
    n = 5
    if pathogen != "RSV":
        D_REL1 = 0.1 + quantiles[n] * 0.9
        D_REL2 = 0.1 + quantiles[n+1] * 0.9
    F1 = quantiles[n]
    R1 = 0.002 + quantiles[n+1] * 0.008
    AGE_OBS = jnp.array([quantiles[n+2+i] for i in range(7)])
    if pathogen != "RSV":
        return jnp.concatenate([jnp.array([BETA, SEASONALITY, OFFSET, WANE2, S_REL1, S_REL2, D_REL1, D_REL2, F1, R1]), AGE_OBS])
    return jnp.concatenate([jnp.array([BETA, SEASONALITY, OFFSET, WANE2, S_REL1, S_REL2, F1, R1]), AGE_OBS])

def generate_parameter_samples(pathogen, savepath=None):
    likelihood, _ = get_likelihood(
        pathogen,
        "Exponential",
        "split",
        "nrdaycare5maxagep028",
        1e-9,
        normalize=False,
        CENSUS_AGE_POP=CENSUS_AGE_POP,
        NAG=8,
    )

    # RSV has 15 params in quantile_to_params, non-RSV has 17
    n_params = 15 if pathogen == "RSV" else 17

    print(f"Generating {total_samples} LHS samples for {pathogen}...")
    key, subkey = jax.random.split(key)
    lhs_samples = latin_hypercube_sample(subkey, total_samples, n_params)

    vmap_params = jax.jit(jax.vmap(lambda q: quantile_to_params(q, pathogen=pathogen)))
    vmap_likelihood = jax.jit(jax.vmap(likelihood))

    parameter_samples = []
    likelihoods = []

    start_time = time.time()
    n_chunks = (total_samples + chunk_size - 1) // chunk_size

    for i in range(0, total_samples, chunk_size):
        chunk_id = i // chunk_size + 1
        chunk_end = min(i + chunk_size, total_samples)

        elapsed = time.time() - start_time
        done_chunks = max(1, chunk_id - 1)
        eta_min = ((n_chunks - chunk_id) * (elapsed / done_chunks)) / 60

        print(f"[{pathogen}] chunk {chunk_id}/{n_chunks}, ETA: {eta_min:.2f} min")

        chunk_lhs = lhs_samples[i:chunk_end]
        chunk_params = vmap_params(chunk_lhs)
        chunk_likelihoods = vmap_likelihood(chunk_params)

        parameter_samples.append(chunk_params)
        likelihoods.append(chunk_likelihoods)

    parameter_samples = jnp.concatenate(parameter_samples, axis=0)
    likelihoods = jnp.concatenate(likelihoods, axis=0)

    if savepath is not None:
        with open(savepath, "wb") as f:
            pickle.dump((parameter_samples, likelihoods), f)

    min_idx = jnp.nanargmin(likelihoods)
    print(f"[{pathogen}] Minimum likelihood: {likelihoods[min_idx]}")
    print(f"[{pathogen}] Corresponding parameters: {parameter_samples[min_idx]}")
    return parameter_samples, likelihoods

def plot_parameter_distributions(axes, parameter_samples, likelihoods,):
    # plot PMF of each parameter based on inverse-likelihood weights
    n_plot_params = int(parameter_samples.shape[1])
    parameter_names = [f"param_{i}" for i in range(n_plot_params)]

    axes = axes.flatten()

    weights = 1.0 / (1.0 + np.array(likelihoods))
    weights = weights / np.sum(weights)

    for i, ax in enumerate(axes):
        if i < n_plot_params:
            ax.hist(
                np.array(parameter_samples[:, i]),
                bins=50,
                weights=weights,
                alpha=0.7,
                color="steelblue",
                edgecolor="black",
            )
            ax.set_title(parameter_names[i])
            ax.set_xlabel(parameter_names[i])
            ax.set_ylabel("Probability Mass")
        else:
            ax.axis("off")

if __name__ == "__main__":

    key = jax.random.PRNGKey(0)

    pathogens = ["Metapneumovirus", "InfluenzaA", "InfluenzaB", "Adenovirus"]

    chunk_size = 1000
    total_samples = 1_000_000
    # for p_idx, pathogen in enumerate(pathogens):
    #     print(f"\n=== {pathogen} ({p_idx+1}/{len(pathogens)}) ===")
    #     savepath = f"Data/Processed/{pathogen}_parameter_samples.pkl"
    #     parameter_samples, likelihoods = generate_parameter_samples(pathogen, savepath=savepath)
    #     fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    #     plot_parameter_distributions(axes, parameter_samples, likelihoods)
    #     plt.tight_layout()
    #     plt.savefig(f"Figures/{pathogen}_parameter_pmf_from_likelihood_lhs.png", dpi=300)
    #     plt.close(fig)


    from Parameters.census_population import CENSUS_AGE_POP, AGE_GROUP_NAMES
    # for the first, fifth, and tenth percentile of likelihoods, load nine random parameter sets from the samples with likelihoods closest to those percentiles, run simulations, and plot the resulting incidence curves against observed data.
    N_S, NAG = 3, 7
    ## Initial conditions
    STATE0 = jnp.zeros((2*N_S+1,NAG))
    STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
    STATE0 = STATE0.at[1,:].set(1)
    # # flatten initial state and add maternal immunity compartment
    STATE0 = STATE0.flatten()
    STATE0 = jnp.concatenate((jnp.array([0]), STATE0))
    start_date = '2015-07-04'
    end_date = '2025-05-01'
    EPOCH = pd.to_datetime('1970-01-01')
    START = pd.to_datetime(start_date) 
    END = pd.to_datetime(end_date)
    FULL_PERIOD = pd.date_range(start=EPOCH, end=END, freq='D')
    FULL_POINTS = np.array(date_to_t(FULL_PERIOD))
    PERIOD = pd.date_range(start=START, end=END, freq='D')
    POINTS = np.array(date_to_t(PERIOD))

    # calculate and save age ratios for all parameter samples
    for p_idx, pathogen in enumerate(pathogens):
        print(f"\n=== {pathogen} ({p_idx+1}/{len(pathogens)}) ===")
        with open(f"Outputs/{pathogen}_flexagep1_parameter_samples_lhs.pkl", "rb") as f:
            parameter_samples, likelihoods = pickle.load(f)
        # filter out nan likelihoods and corresponding parameter samples
        valid_indices = ~jnp.isnan(likelihoods)
        likelihoods = likelihoods[valid_indices]
        parameter_samples = parameter_samples[valid_indices]

        age_ratios = []
        n_samples = parameter_samples.shape[0]
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        start_time = time.time()

        for chunk_id in range(n_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min((chunk_id + 1) * chunk_size, n_samples)
            
            elapsed = time.time() - start_time
            done_chunks = max(1, chunk_id)
            eta_min = ((n_chunks - chunk_id - 1) * (elapsed / done_chunks)) / 60
            
            print(f"[{pathogen}] Processing chunk {chunk_id + 1}/{n_chunks} ({start_idx}-{end_idx}), ETA: {eta_min:.2f} min")
            
            chunk_ratios = []
            for idx in range(start_idx, end_idx):
                params = x_to_params(parameter_samples[idx], pathogen, "Exponential", "NA", "flexagep1", print_params=False)
                solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
                values = solution.ys.T
                times = solution.ts
                p_time_to_obs = pathogen_parameters(pathogen, import_multiplier=1e-9, incidence_data=False, hosp=True, NAG=NAG)[3]
                expected_obs = calculate_expected_obs(values, p_time_to_obs, len(times), NAG=NAG)
                expected_obs_per_season = calculate_observations_per_season_jax(expected_obs, age_groups=True, NAG=NAG)
                ratio_of_ratios = age_ratio_of_rebound(jnp.stack([expected_obs_per_season, expected_obs_per_season], axis=0), idx_num=1, idx_den=2, threshold_factor=1/3)
                chunk_ratios.append(ratio_of_ratios)
            
            age_ratios.extend(chunk_ratios)
            
            # Save intermediate results
            age_ratios_array = jnp.array(age_ratios)
            with open(f"Outputs/{pathogen}_age_ratios_partial.pkl", "wb") as f:
                pickle.dump(age_ratios_array, f)

        age_ratios = jnp.array(age_ratios)
        with open(f"Outputs/{pathogen}_age_ratios.pkl", "wb") as f:
            pickle.dump(age_ratios, f)
        print(f"[{pathogen}] Complete - saved {len(age_ratios)} age ratios")


    # from plotting import lockdown_incidence_plot, lockdown_incidence_format
    # for pidx, pathogen in enumerate(pathogens):
    #     _, _, _, p_time_to_obs, _ = pathogen_parameters(pathogen, import_multiplier=1e-9, incidence_data=False, hosp=True, NAG=NAG)

    #     with open(f"Outputs/{pathogen}_flexagep1_parameter_samples_lhs.pkl", "rb") as f:
    #         parameter_samples, likelihoods = pickle.load(f)
    #     # filter out nan likelihoods and corresponding parameter samples
    #     valid_indices = ~jnp.isnan(likelihoods)
    #     likelihoods = likelihoods[valid_indices]
    #     parameter_samples = parameter_samples[valid_indices]
    #     percentiles = [0, 1, 5, 10]
    #     percentile_values = np.percentile(likelihoods, percentiles)
    #     print(percentile_values)

    #     fig, axes = plt.subplots(4, 3, figsize=(10, 10))
    #     for i, perc in enumerate(percentiles):
    #         perc_value = percentile_values[i]
    #         close_indices = np.argsort(np.abs(likelihoods - perc_value))[:3]
    #         for j, idx in enumerate(close_indices):
    #             params = x_to_params(parameter_samples[idx], pathogen, "Exponential", "NA", "flexagep1", print_params=True)
    #             solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
    #             values = solution.ys.T
    #             times = solution.ts
    #             expected_obs = calculate_expected_obs(values, p_time_to_obs, len(times), NAG=NAG)
    #             expected_obs_per_season = calculate_observations_per_season_jax(expected_obs, age_groups=True, NAG=NAG)
    #             ratio_of_ratios = age_ratio_of_rebound(jnp.stack([expected_obs_per_season, expected_obs_per_season], axis=0), idx_num=i, idx_den=j, threshold_factor=1/3)
    #             print(f"Percentile {perc}%, idx {idx}, ratio of ratios: {ratio_of_ratios}")

                
    #             ax = axes[i, j]
    #             aggregation = "MS"
    #             params = params + (NAG,)
    #             kpsc_proportion_positive_incidence_plot(ax, pathogen, None, AGE_GROUP_NAMES, aggregation=aggregation, factor=10000, color="silver", label="Data", hosp=True)                
    #             mx = lockdown_incidence_plot(ax,STATE0,params,POINTS,date_to_t('2020-03-19'),label=None,by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs, color="k", linewidth=2, NAG=NAG)
    #             lockdown_incidence_format(ax,date_to_t('2020-03-19'),365,mx,year_window=2)
    #             # Add observed data plotting here (e.g., kpsc_positive_test_plot)
    #             ax.set_title(f"{pathogen} - {perc}th percentile")
    #             ax.set_xlabel("Time")
    #             ax.set_ylabel("Incidence")
    #             ax.legend(frameon=False)

    #     plt.tight_layout()
    #     plt.savefig(f"Figures/{pathogen}_simulated_incidence_percentiles.png", dpi=300)
    #     plt.close(fig)