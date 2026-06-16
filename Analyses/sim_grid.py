# BJS March 2026
# code to run simulations across a grid of parameters and plot the results

import jax
import jax.numpy as jnp
import numpy as np
import itertools as it
import pandas as pd
import time
import pickle
import os
import re
import glob

from utils import consistent_x_from_DE, x_to_params, date_to_t, calculate_population_size, calculate_R0_from_values, load_mcmc_chain
from fit_MCMC import run_simulation
from data_processing import calculate_proportion_positive_incidence
# from Parameters.times_and_contacts import PERIOD

from scipy import stats
from scipy.stats import qmc, binned_statistic_2d
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
# NAG = 7  # Number of age groups
N_S = 3  # Susceptibility classes

PARAMETER_NAMES = [
    "Basic reproduction number", "First infection duration", "Second infection duration", 
    "Transmissibility", "Seasonality", "Phase", "Waning (per 100 days after first infection)", 
    "Waning (per 100 days after second infection)", "Immunity from first infection", 
    "Immunity from second infection", "Immunity to severe disease after first infection",
    "Immunity to severe disease after second infection", "Maternal immunity",
    "Contact reduction", "Contact recovery",
    "Disease susceptibility <3m", "Disease susceptibility 3–11m", "Disease susceptibility 1–4y",
    "Disease susceptibility 5–17y", "Disease susceptibility 18–49y", "Disease susceptibility 50-64y", 
    "Disease susceptibility 65+y"
]

SHORT_PNAMES = [
    "R0", "rec_up", "rec_same", "transmissibility", "seasonality", "phase", "wane1", "waning", 
    "immunity1", "immunity2", "disease_immunity1", "disease_immunity2", "maternal_immunity", 
    "contact_reduction", "contact_recovery",
    "susceptibility_0_3m", "susceptibility_3_11m", "susceptibility_1_4y", "susceptibility_5_17y", "susceptibility_18_49y", "susceptibility_50_64y", "susceptibility_65y"
]

PATHOGEN_SHORT_NAMES = {
    "RSV": "RSV", "Metapneumovirus": "hMPV", "InfluenzaA": "FluA", 
    "InfluenzaB": "FluB", "Adenovirus": "AdV", "Parainfluenza3": "PIV3"
}


def suppression_duration_single_series(obs_series, anchor_idx, threshold_divisor=20.0):
    """Measure suppression length using dip/rebound logic from plotting + FluNet scripts.

    Uses: threshold = max(post-anchor incidence) / threshold_divisor.
    Duration is from the point immediately before first dip below threshold,
    until first rebound above threshold (or right-censored at series end).
    """
    n_time = obs_series.shape[0]
    invalid_anchor = anchor_idx >= (n_time - 1)
    indices = jnp.arange(n_time)

    pre_anchor_mask = indices < anchor_idx
    post_anchor_mask = indices > anchor_idx
    # Avoid dynamic slicing so this stays JAX-jittable under vmap.
    pre_anchor_vals = jnp.where(pre_anchor_mask, obs_series, -jnp.inf)
    threshold = jnp.max(pre_anchor_vals) / threshold_divisor

    dip_mask = post_anchor_mask & (obs_series < threshold)
    dip_found = jnp.any(dip_mask)
    first_dip_idx = jnp.argmax(dip_mask)
    last_pre_idx = jnp.maximum(first_dip_idx - 1, 0)

    rebound_mask = (indices > last_pre_idx) & (obs_series > threshold)
    rebound_found = jnp.any(rebound_mask)
    first_rebound_idx = jnp.argmax(rebound_mask)

    duration_observed = first_rebound_idx - last_pre_idx
    duration_right_censored = (n_time - 1) - last_pre_idx
    duration = jnp.where(rebound_found, duration_observed, duration_right_censored)

    return jnp.where(jnp.logical_not(invalid_anchor) & dip_found, duration.astype(float), jnp.nan)


def suppression_duration_by_age(obs_by_age, obs_summed_age, anchor_idx, threshold_divisor=20.0):
    """Return suppression duration for each age group plus summed incidence."""
    suppression_by_age = jax.vmap(
        lambda x: suppression_duration_single_series(x, anchor_idx, threshold_divisor),
        in_axes=1,
        out_axes=0,
    )(obs_by_age)
    suppression_summed_age = suppression_duration_single_series(obs_summed_age, anchor_idx, threshold_divisor)
    return jnp.concatenate([suppression_by_age, suppression_summed_age[None]], axis=0)

# ==========================================
# DEFINE SAMPLING SPACE
# ==========================================
def parameter_space(good_simulations, NAG=7, n_samples=None):
    """Extracts parameter sets from good simulations for use in sampling."""
    parameter_sets = []
    if n_samples is None:
        for pathogen_info in good_simulations:
            pathogen, seed, lockdown, option1, option2 = pathogen_info
            x = consistent_x_from_DE(pathogen, lockdown, option1, option2, seed, NAG=NAG)
            parameter_sets.append(x)
    else:
        for pathogen_info in good_simulations:
            pathogen, seed, lockdown, option1, option2 = pathogen_info
            chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, prune=0, prefix="", just_chain=True)
            # draw n_samples randomly from the chain
            sampled_xs = chain[np.random.choice(chain.shape[0], size=n_samples, replace=True)]
            consistent_xs = jax.vmap(lambda x: consistent_x_from_DE(pathogen, lockdown, option1, option2, seed, NAG=NAG, x_DE=x))(sampled_xs)
            parameter_sets.append(consistent_xs.T)
    parameter_sets = jnp.array(parameter_sets)
    return(parameter_sets)
    
def lh_sampling(parameter_sets, n_samples, dimension=None):
    """Generates samples from the parameter space using Latin Hypercube Sampling.
    Args:
        parameter_sets: Array of shape (n_pathogens, n_parameters) OR 
                        (n_pathogens, n_parameters, n_input_samples) containing parameter sets.
        n_samples: Number of samples to generate.
        dimension: The dimension of the simplex to sample from. If None, samples from the full simplex.
                   If 0, samples from vertices; if 1, samples from edges; if k, samples from k-dimensional faces.
    Returns:
        samples: Array of shape (n_samples, n_parameters) containing the sampled parameter sets.
    """
    is_3d = (parameter_sets.ndim == 3)
    
    if is_3d:
        n_pathogens, n_parameters, n_input_samples = parameter_sets.shape
        if n_input_samples < n_samples:
            raise ValueError(f"For 3D parameter_sets, the number of input samples ({n_input_samples}) "
                             f"must be at least the requested n_samples ({n_samples}).")
    else:
        n_pathogens = parameter_sets.shape[0]
    
    if dimension is None:
        dimension = n_pathogens - 1  # Full simplex by default
    
    # Validate dimension parameter
    if dimension < 0 or dimension > n_pathogens - 1:
        raise ValueError(f"Dimension must be between 0 and {n_pathogens - 1}")
    
    # Use Latin Hypercube Sampling for better space-filling properties
    if dimension == 0:
        # Sample from vertices (pure pathogens)
        vertex_indices = np.random.choice(n_pathogens, size=n_samples)
        if is_3d:
            # Pick the i-th sample from the randomly selected pathogen's chain
            samples = jnp.array([parameter_sets[vertex_indices[i], :, i] for i in range(n_samples)])
        else:
            samples = parameter_sets[vertex_indices]
        return samples
    
    elif dimension == 1:
        # Sample from edges (between pairs of pathogens)
        pairs = np.array(list(it.combinations(range(n_pathogens), 2)))
        pairs = np.tile(pairs, (int(np.ceil(n_samples / pairs.shape[0])), 1))[:n_samples]
        
        # Generate weights for each pair
        weights = jnp.arange(n_samples)/(n_samples-1)
        
        samples = []
        for i in range(n_samples):
            idx1, idx2 = pairs[i]
            w = weights[i]
            
            if is_3d:
                # Use the i-th parameter set from the 3D block
                sample = w * parameter_sets[idx1, :, i] + (1 - w) * parameter_sets[idx2, :, i]
            else:
                sample = w * parameter_sets[idx1] + (1 - w) * parameter_sets[idx2]
                
            samples.append(sample)
            
        samples = jnp.array(samples)
        return samples
    
    else:
        # Sample from higher-dimensional faces or full simplex
        # Create Latin Hypercube sampler for the specified dimension
        sampler = qmc.LatinHypercube(d=dimension)
        
        # Generate samples in simplex coordinates
        lhs_samples = sampler.random(n=n_samples)
        
        # Convert to Dirichlet-like weights (ensure they sum to 1)
        # Transform uniform samples to exponential, then normalize
        exp_samples = -jnp.log(1 - lhs_samples + 1e-10)  # Add small epsilon to avoid log(0)
        
        if dimension < n_pathogens - 1:
            # For lower-dimensional faces, systematically iterate through combinations of pathogens
            pathogen_combinations = list(it.combinations(range(n_pathogens), dimension+1))
            
            # Tile combinations to cover all samples, and slice exactly to n_samples
            repeated_combos = pathogen_combinations * (int(np.ceil(n_samples / len(pathogen_combinations))))
            repeated_combos = repeated_combos[:n_samples]
            
            samples = []
            for i in range(n_samples):
                selected_pathogens = repeated_combos[i]
                
                # Create barycentric coordinates for selected pathogens
                weights = jnp.zeros(n_pathogens)
                selected_exp = exp_samples[i]
                
                # Add one more dimension to complete the simplex for selected pathogens
                last_coord = np.random.exponential(1.0)
                full_exp = jnp.concatenate([selected_exp, jnp.array([last_coord])])
                
                # Normalize and assign to selected pathogens
                normalized_weights = full_exp / jnp.sum(full_exp)
                weights = weights.at[jnp.array(selected_pathogens)].set(normalized_weights)
                
                # Compute sample as convex combination
                if is_3d:
                    sample = weights @ parameter_sets[:, :, i]
                else:
                    sample = weights @ parameter_sets
                    
                samples.append(sample)
            
            samples = jnp.array(samples)
            return samples
        
        else:
            # Full simplex case (original code)
            # Add one more dimension to complete the simplex
            last_coord = np.random.exponential(1.0, size=(n_samples, 1))
            full_exp = jnp.concatenate([exp_samples, last_coord], axis=1)
            
            # Normalize to create proper barycentric coordinates
            lh_samples = full_exp / jnp.sum(full_exp, axis=1, keepdims=True)
            
            # Transform to parameter space via convex combination
            if is_3d:
                # lh_samples shape: (n_samples, n_pathogens) -> 'ij'
                # parameter_sets shape: (n_pathogens, n_parameters, n_samples) -> 'jki'
                # desired shape: (n_samples, n_parameters) -> 'ik'
                samples = jnp.einsum('ij,jki->ik', lh_samples, parameter_sets)
            else:
                samples = lh_samples @ parameter_sets
            
            return samples

# ==========================================
# SIMULATION FUNCTIONS AND BATCHING SYSTEM
# ==========================================

def worker(args):
    """Worker function to run a single simulation and extract relevant metrics.
    Args:
        args: A tuple containing (sample, lockdown, POINTS, STATE0, p_time_to_obs, option1, option2, NAG)
    Returns:
        A 3D array containing the following metrics for each season and age group:
        - Observed infections per season (shape: n_seasons x NAG)
        - Time of peak incidence per season (shape: n_seasons x NAG)
        - Proportion of first infectious compartment per season (shape: n_seasons x NAG)
        - Proportion of all infectious compartments per season (shape: n_seasons x NAG)
        - Infections caused by each age group per season (shape: n_seasons x NAG)
        - Force of infection experienced by each age group per season (shape: n_seasons x NAG)
        - Hospitalization-causing infections by age group per season (shape: n_seasons x NAG)
        - Population by age group per season (shape: n_seasons x NAG)
        - Suppression length in days by age group (shape: n_seasons x NAG, repeated each season)
    """
    sample, lockdown, POINTS, STATE0, p_time_to_obs, option1, option2, NAG = args
    params = x_to_params(sample, "sim", lockdown, option1+"mimmwane", option2+"nr", NAG=NAG)
    solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
    # find total observed infections each season in each age group
    values = solution.ys.T
    trajectory = jnp.diff(values[-NAG:,:], axis=1).T
    p_time_to_obs_flipped = jnp.flip(p_time_to_obs.flatten())
    def obs_convolution(x):
        return jnp.convolve(x, p_time_to_obs_flipped, mode='same')
    obs = jax.vmap(obs_convolution, in_axes=1, out_axes=1)(trajectory)
    obs = jax.nn.softplus(obs*100)/100
    obs_summed_age = obs.sum(axis=1)
    # sum obs over each season, starting with the first time point
    n_seasons = int((POINTS[-1] - POINTS[0]) / 365)
    # Curtail pops and obs to fit exact seasons (ignore partial days due to leap years)
    days_to_keep = n_seasons * 365
    population_size = calculate_population_size(values, NAG=NAG)
    population_size_curtailed = population_size[:days_to_keep, :]
    population_reshaped = population_size_curtailed.reshape((n_seasons, 365, NAG))
    population_by_age_by_season = population_reshaped[:,0,:]
    population_by_age_by_season_wsum = jnp.concatenate([population_by_age_by_season,
    population_by_age_by_season.sum(axis=1)[:,None]], axis=1)

    obs_curtailed = obs[:days_to_keep, :]
    obs_summed_age_curtailed = obs_summed_age[:days_to_keep]

    # Aggregate daily obs to monthly (30-day buckets) for suppression duration calculation.
    # This matches the monthly aggregation used for pathogen data in extract_target_value_from_data.
    days_per_month = 30
    n_months = days_to_keep // days_per_month
    obs_monthly = obs_curtailed[:n_months * days_per_month, :].reshape((n_months, days_per_month, NAG)).sum(axis=1)
    obs_summed_age_monthly = obs_summed_age_curtailed[:n_months * days_per_month].reshape((n_months, days_per_month)).sum(axis=1)
    anchor_t = date_to_t(pd.to_datetime("2020-01-01"))
    anchor_day_idx = int(np.searchsorted(np.asarray(POINTS[:days_to_keep]), anchor_t, side='right')) - 1
    anchor_month_idx = anchor_day_idx // days_per_month
    suppression_duration = suppression_duration_by_age(
        obs_monthly,
        obs_summed_age_monthly,
        anchor_idx=anchor_month_idx,
        threshold_divisor=20.0,
    )  # duration is now in months (each index = 1 month)
    suppression_duration_by_season = jnp.tile(suppression_duration[None, :], (n_seasons, 1))

    obs_per_season = obs_curtailed.reshape((n_seasons, 365, NAG)).sum(axis=1)
    obs_summed_age_per_season = obs_summed_age_curtailed.reshape((n_seasons, 365)).sum(axis=1)
    # concatenate to obs_per_season
    obs_per_season = jnp.concatenate([obs_per_season, obs_summed_age_per_season[:, None]], axis=1)
    # find the centre of gravity of incidence in each age group for each season
    obs_reshaped = obs_curtailed.reshape((n_seasons, 365, NAG))
    days = jnp.arange(365)
    peak_times = jnp.sum(days[None, :, None] * obs_reshaped, axis=1) / (jnp.sum(obs_reshaped, axis=1) + 1e-10)
    obs_summed_age_reshaped = obs_summed_age_curtailed.reshape((n_seasons, 365))
    peak_times_summed_age = jnp.sum(days[None, :] * obs_summed_age_reshaped, axis=1) / (jnp.sum(obs_summed_age_reshaped, axis=1) + 1e-10)
    peak_times = jnp.concatenate([peak_times, peak_times_summed_age[:, None]], axis=1)
    # dynamic quantities based on values, not obs
    shaped_values = values[1:, :days_to_keep].reshape((1+2*N_S, NAG, days_to_keep))
    infectious = shaped_values[1:2*N_S:2, :, :]
    susceptible = shaped_values[0:2*N_S:2, :, :]
    # find the proportion of those in the first infected compartment in each age group
    first_infectious = infectious[0]
    first_infectious_by_age = first_infectious.reshape((NAG, n_seasons, 365)).sum(axis=2).T
    proportion_first_infectious = first_infectious_by_age / first_infectious_by_age.sum(axis=1, keepdims=True)
    # and the summed age version for consistency
    first_infectious_summed_age = first_infectious.sum(axis=0)
    first_infectious_summed_age_by_season = first_infectious_summed_age.reshape((n_seasons, 365)).sum(axis=1)
    proportion_first_infectious = jnp.concatenate([proportion_first_infectious,
    first_infectious_summed_age_by_season[:, None]], axis=1)
    # proportion of infected in each age group across all susceptibility classes
    all_infectious = infectious.sum(axis=0)
    all_infectious_by_season = all_infectious.reshape((NAG, n_seasons, 365)).sum(axis=2).T
    proportion_infectious_by_age = all_infectious_by_season / all_infectious_by_season.sum(axis=1, keepdims=True)
    # and the summed age version for consistency
    all_infectious_summed_age = all_infectious.sum(axis=0)
    all_infectious_summed_age_by_season = all_infectious_summed_age.reshape((n_seasons, 365)).sum(axis=1)
    proportion_infectious = jnp.concatenate([proportion_infectious_by_age,
    all_infectious_summed_age_by_season[:, None]], axis=1)
    # how many infections are caused by each age group in each season?
    # Calculate force of infection from each age group to each receiving age group
    # Shape: (NAG_recieving, NAG_causing, time)
    relative_contact = params[10][-days_to_keep:].T
    foi_matrix = params[4] * relative_contact[:, None, :] * relative_contact[None, :, :] * params[3][:, :, None] * (all_infectious[None, :, :] / jnp.sum(population_size_curtailed, axis=1)[None, None, :])
    # Calculate new infections: susceptible * foi * susceptibility by class
    # susceptible shape: (N_S, NAG, time)
    # foi_matrix shape: (NAG_recieving, NAG_causing, time)
    # params[6] shape: (N_S,)
    infections_matrix = params[6][:, None, None, None] * foi_matrix[None, :, :, :] * susceptible[:, :, None, :]
    # Sum over susceptibility classes and receiving age groups to get infections caused by each age group
    # Shape: (NAG_causing, time)
    infections_caused_by_age = infections_matrix.sum(axis=(0, 1))
    # Reshape and sum by season
    infections_caused_by_age_by_season = infections_caused_by_age.reshape((NAG, n_seasons, 365)).sum(axis=2).T
    # summed version for consistency
    infections_caused_by_age_summed = infections_caused_by_age.sum(axis=0)
    infections_caused_by_age_summed_by_season = infections_caused_by_age_summed.reshape((n_seasons, 365)).sum(axis=1)
    infections_caused_by_age_by_season = jnp.concatenate([infections_caused_by_age_by_season,
        infections_caused_by_age_summed_by_season[:, None]], axis=1)
    # weight by infection hospitalization ratio
    P_OBS = params[8]
    OBS_AGE = params[9]
    hospitalizations_caused_by_age = jnp.sum(infections_matrix * P_OBS[:, None, None, None] * OBS_AGE[None, :, None, None], axis=(0, 1))
    hospitalizations_caused_by_age_by_season = hospitalizations_caused_by_age.reshape((NAG, n_seasons, 365)).sum(axis=2).T
    hospitalizations_caused_by_age_summed = hospitalizations_caused_by_age.sum(axis=0)
    hospitalizations_caused_by_age_summed_by_season = hospitalizations_caused_by_age_summed.reshape((n_seasons, 365)).sum(axis=1)
    hospitalizations_caused_by_age_by_season = jnp.concatenate([hospitalizations_caused_by_age_by_season,
        hospitalizations_caused_by_age_summed_by_season[:, None]], axis=1)
    # what is the force of infection experienced by each age group each season?
    # Sum over all age groups that are causing infections (first dimension of foi_matrix)
    # foi_matrix shape: (NAG_causing, NAG_receiving, time)
    foi_experienced_by_age = foi_matrix.sum(axis=0)  # Shape: (NAG_receiving, time)
    # Reshape and sum by season
    foi_experienced_by_age_by_season = foi_experienced_by_age.reshape((NAG, n_seasons, 365)).sum(axis=2).T
    # summed version for consistency
    foi_experienced_summed = foi_experienced_by_age.sum(axis=0)
    foi_experienced_summed_by_season = foi_experienced_summed.reshape((n_seasons, 365)).sum(axis=1)
    foi_experienced_by_age_by_season = jnp.concatenate([foi_experienced_by_age_by_season,
        foi_experienced_summed_by_season[:, None]], axis=1)
    
    # return as a 3d array
    return jnp.stack([
        obs_per_season,
        peak_times,
        proportion_first_infectious,
        proportion_infectious,
        infections_caused_by_age_by_season,
        foi_experienced_by_age_by_season,
        hospitalizations_caused_by_age_by_season,
        population_by_age_by_season_wsum,
        suppression_duration_by_season,
    ], axis=0)

def simulate_samples_chunked(samples, lockdown, POINTS, STATE0, p_time_to_obs, option1, option2,
                             base_save_path, chunk_size, skip_existing=False, NAG=7):
    """
    Simulates samples in chunks and saves each chunk to disk to avoid OOM errors.
    """
    n_samples = len(samples)
    num_chunks = int(np.ceil(n_samples / chunk_size))
    
    print(f"Total samples: {n_samples}. Chunk size: {chunk_size}. Total chunks: {num_chunks}.")
    
    # Create the directory for this run's results
    os.makedirs(base_save_path, exist_ok=True)
    all_chunk_paths = []

    def partial_worker(sample):
        return worker((sample, lockdown, POINTS, STATE0, p_time_to_obs, option1, option2, NAG))
    vmapped_worker = jax.jit(jax.vmap(partial_worker))
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_samples)
        sample_chunk = samples[start_idx:end_idx]
        
        # Define file path for this chunk
        chunk_filename = f"chunk_{i:04d}.pickle"
        chunk_path = os.path.join(base_save_path, chunk_filename)
        
        # Check if chunk already exists and skip_existing is True
        if skip_existing and os.path.exists(chunk_path):
            print(f"  Skipping chunk {i+1}/{num_chunks} (already exists): {chunk_path}")
            all_chunk_paths.append(chunk_path)
            continue
        
        print(f"  Processing chunk {i+1}/{num_chunks} (samples {start_idx} to {end_idx})...")
        
        # Run vmap on the smaller chunk
        chunk_results = vmapped_worker(jnp.array(sample_chunk))
        
        # Block until computation is done and data is on host
        chunk_results.block_until_ready()
        
        # Save chunk to disk (convert to NumPy array for robust pickling)
        with open(chunk_path, "wb") as f:
            pickle.dump(np.asarray(chunk_results), f)
        # Save sample chunk as well
        samples_chunk_path = os.path.join(base_save_path, f"samples_chunk_{i:04d}.pickle")
        with open(samples_chunk_path, "wb") as f:
            pickle.dump(np.asarray(sample_chunk), f)
            
        all_chunk_paths.append(chunk_path)
        print(f"  Saved {chunk_path}")
    print(f"All chunks saved to {base_save_path}")
    return all_chunk_paths

def load_and_recombine_results(run_save_path):
    chunk_files = sorted(glob.glob(os.path.join(run_save_path, "chunk_*.pickle")), key=lambda f: int(re.search(r'chunk_(\d+)\.pickle', f).group(1)))
    sample_files = sorted(glob.glob(os.path.join(run_save_path, "samples_chunk_*.pickle")), key=lambda f: int(re.search(r'samples_chunk_(\d+)\.pickle', f).group(1)))
    
    if not chunk_files:
        return None, None
        
    all_chunks = [pickle.load(open(f, "rb")) for f in chunk_files]
    all_samples = [pickle.load(open(f, "rb")) for f in sample_files]
    
    return np.concatenate(all_chunks, axis=0), np.concatenate(all_samples, axis=0) if all_samples else None

# ==========================================
# EPIDEMIOLOGICAL OUTCOMES
# ==========================================
def time_to_rebound(x, threshold_factor=1/2, include_years=True):
    obs_summed, peak_times_summed = x[0, :, -1], x[1, :, -1]
    threshold = threshold_factor * jnp.median(obs_summed[:5])
    last_pre_pandemic_peak_time = peak_times_summed[4] + include_years * (4 * 365)
    
    post_pandemic_obs = obs_summed[5:]
    rebound_mask = post_pandemic_obs >= threshold
    rebound_found = jnp.any(rebound_mask)

    # JAX-safe first index lookup (argmax on booleans gives first True; 0 if none found)
    first_rebound_idx = jnp.argmax(rebound_mask)
    season_idx = 5 + first_rebound_idx
    first_post_pandemic_peak_time = peak_times_summed[season_idx] + include_years * (season_idx * 365)

    return jnp.where(rebound_found, first_post_pandemic_peak_time - last_pre_pandemic_peak_time, jnp.nan)

def outbreak_in_season(x, threshold_factor=0.1, season_idx=6):
    obs_summed = x[0, :, -1]
    pre_pandemic_median = jnp.median(obs_summed[:5])
    threshold = threshold_factor * pre_pandemic_median
    return obs_summed[season_idx] >= threshold

def relative_size_of_rebound(x, threshold_factor=1/2):
    obs_summed = x[0, :, -1]
    pre_pandemic_median = jnp.median(obs_summed[:5])
    post_pandemic_max = jnp.max(obs_summed[5:])
    return jnp.where(post_pandemic_max < threshold_factor*pre_pandemic_median, jnp.nan, post_pandemic_max / pre_pandemic_median)

def age_ratio_of_rebound(x, idx_num=2, idx_den=1, threshold_factor=1/2, season_idx=None):
    """Calculate the ratio of infections in one age group to another during the rebound season, relative to the pre-pandemic ratio.
    Args:
        x: The 3D array of metrics returned by the worker function, with shape (n_metrics, n_seasons, NAG+1). The last age group (index -1) corresponds to the summed age group.
        idx_num: The index of the age group to use as the numerator in the ratio (default is 2, which corresponds to 1-4y).
        idx_den: The index of the age group to use as the denominator in the ratio (default is 1, which corresponds to 3-12m).
        threshold_factor: Threshold factor for rebound detection (default 1/2, consistent with time_to_rebound).
    Returns: The ratio of the specified age groups during the rebound season, relative to the pre-pandemic ratio. If there is no rebound, returns NaN.
    """
    obs_per_season = x[0, :, :]
    pre_pandemic_obs = obs_per_season[:5]
    if idx_den is None:
        has_valid_pre_pandemic = (pre_pandemic_obs[:, idx_num] > 0).all() & jnp.isfinite(pre_pandemic_obs[:, idx_num]).all()
        pre_pandemic_mean_ratio = jnp.mean(obs_per_season[:5, idx_num])
    else:
        has_valid_pre_pandemic = (pre_pandemic_obs[:, idx_num] > 0).all() & (pre_pandemic_obs[:, idx_den] > 0).all()
        pre_pandemic_mean_ratio = jnp.mean(obs_per_season[:5, idx_num] / obs_per_season[:5, idx_den])

    post_pandemic_obs = obs_per_season[5:, -1]
    if season_idx is None:
        threshold = threshold_factor * jnp.mean(obs_per_season[:5, -1])
        rebound_mask = post_pandemic_obs >= threshold
        rebound_found = jnp.any(rebound_mask)
        first_rebound_idx = jnp.argmax(rebound_mask)
    else:
        rebound_found = True
        first_rebound_idx = season_idx - 5  # Adjust for indexing after the first 5 seasons
    
    # Get first rebound season index
    rebound_season_idx = 5 + first_rebound_idx
    rebound_obs = obs_per_season[rebound_season_idx]
    
    # Return NaN if numerator or denominator is zero or NaN

    if idx_den is None:
        has_valid_rebound = (rebound_obs[idx_num] > 0) & jnp.isfinite(rebound_obs[idx_num])
        rebound_ratio = rebound_obs[idx_num]
    else:
        has_valid_rebound = (rebound_obs[idx_num] > 0) & (rebound_obs[idx_den] > 0) & jnp.isfinite(rebound_obs[idx_num]) & jnp.isfinite(rebound_obs[idx_den])
        rebound_ratio = rebound_obs[idx_num] / rebound_obs[idx_den]
    
    has_valid_obs = has_valid_pre_pandemic & has_valid_rebound
    return jnp.where(rebound_found & has_valid_obs, rebound_ratio / pre_pandemic_mean_ratio, jnp.nan)

def age_time_shift(x, idx_foc=2, idx_ref=1, threshold_factor=1/2, season_idx=None):
    obs_per_season = x[0, :, :]
    peak_times = x[1, :, :]
    pre_pandemic_obs = obs_per_season[:5]
    if idx_ref is None:
        has_valid_pre_pandemic = (pre_pandemic_obs[:, idx_foc] > 0).all() & jnp.isfinite(pre_pandemic_obs[:, idx_foc]).all()
        pre_pandemic_mean_peak_time = jnp.mean(peak_times[:5, idx_foc])
    else:
        has_valid_pre_pandemic = (pre_pandemic_obs[:, idx_foc] > 0).all() & (pre_pandemic_obs[:, idx_ref] > 0).all()
        pre_pandemic_mean_peak_time = jnp.mean(peak_times[:5, idx_foc] - peak_times[:5, idx_ref])

    post_pandemic_obs = obs_per_season[5:, -1]
    if season_idx is None:
        threshold = threshold_factor * jnp.mean(obs_per_season[:5, -1])
        rebound_mask = post_pandemic_obs >= threshold
        rebound_found = jnp.any(rebound_mask)
        first_rebound_idx = jnp.argmax(rebound_mask)
    else:
        rebound_found = True
        first_rebound_idx = season_idx - 5  # Adjust for indexing after the first 5 seasons
    
    rebound_season_idx = 5 + first_rebound_idx
    rebound_obs = obs_per_season[rebound_season_idx]
    
    if idx_ref is None:
        has_valid_rebound = (rebound_obs[idx_foc] > 0) & jnp.isfinite(rebound_obs[idx_foc])
        rebound_peak_time = peak_times[rebound_season_idx, idx_foc]
    else:
        has_valid_rebound = (rebound_obs[idx_foc] > 0) & (rebound_obs[idx_ref] > 0) & jnp.isfinite(rebound_obs[idx_foc]) & jnp.isfinite(rebound_obs[idx_ref])
        rebound_peak_time = (peak_times[rebound_season_idx, idx_foc] - peak_times[rebound_season_idx, idx_ref])
    
    has_valid_obs = has_valid_pre_pandemic & has_valid_rebound
    return jnp.where(rebound_found & has_valid_obs, rebound_peak_time - pre_pandemic_mean_peak_time, jnp.nan)

def age_of_first_infection(x):
    return jnp.mean(jnp.sum(x[2, :5, :-1] * jnp.array(MEDIAN_AGE), axis=1)) / 12

def age_of_infector(x):
    return jnp.mean(jnp.sum(x[4, :5, :-1] * jnp.array(MEDIAN_AGE), axis=1) / jnp.sum(x[4, :5, :-1], axis=1)) / 12

def foi_weighted_age(x):
    return jnp.mean(jnp.sum(x[5, :5, :-1] * jnp.array(MEDIAN_AGE), axis=1) / jnp.sum(x[5, :5, :-1], axis=1)) / 12


def suppression_length(x, idx=-1):
    # Suppression row is repeated across seasons for shape compatibility, so take season 0.
    return x[8, 0, idx]

# ==========================================
# ANALYSIS & PROCESSING
# ==========================================
def extract_target_values(all_results, outcome, **kwargs):
    if outcome == "relative_size": return jax.jit(jax.vmap(relative_size_of_rebound))(all_results)
    if outcome == "age_ratio": return jax.jit(jax.vmap(lambda x: age_ratio_of_rebound(x, kwargs.get('idx_num', 2), kwargs.get('idx_den', 1), kwargs.get('threshold_factor', 1/2))))(all_results)
    if outcome == "age_of_first_infection": return jax.jit(jax.vmap(age_of_first_infection))(all_results)
    if outcome == "suppression_length": return jax.jit(jax.vmap(lambda x: suppression_length(x, kwargs.get('idx', -1))))(all_results)
    if outcome == "outbreak_in_season": return jax.jit(jax.vmap(lambda x: outbreak_in_season(x, kwargs.get('threshold_factor', 1/2), kwargs.get('season_idx', 6))))(all_results)
    if outcome == "age_shift": return jax.jit(jax.vmap(lambda x: age_ratio_of_rebound(x, kwargs.get('idx_num', 2), kwargs.get('idx_den', 1), kwargs.get('threshold_factor', 1/2)) > 1))(all_results)
    if outcome == "age_time_shift": return jax.jit(jax.vmap(lambda x: age_time_shift(x, kwargs.get('idx_foc', 2), kwargs.get('idx_ref', 1), kwargs.get('threshold_factor', 1/2), kwargs.get('season_idx', None))))(all_results)
    if "suppression_length_in_group_" in outcome:
        return jax.jit(jax.vmap(lambda x: suppression_length(x, int(outcome.split("_")[-1]))))(all_results)
    if "infectors_in_group_" in outcome:
        digits = [int(d) for d in outcome.split("_")[-1]]
        if "proportional" in outcome:
            return jax.vmap(lambda x: jnp.sum(jnp.array([x[4, :5, i]/x[7, :5, i] for i in digits]), axis=0).mean(axis=0))(all_results)
        else:
            return jax.vmap(lambda x: jnp.sum(jnp.array([x[4, :5, i] for i in digits]), axis=0).sum(axis=0)/x[4, :5, :-1].sum())(all_results)
    if "hospitalizors_in_group_" in outcome:
        digits = [int(d) for d in outcome.split("_")[-1]]
        if "proportional" in outcome:
            return jax.vmap(lambda x: jnp.sum(jnp.array([x[6, :5, i]/x[7, :5, i] for i in digits]), axis=0).mean(axis=0))(all_results)
        else:
            return jax.vmap(lambda x: jnp.sum(jnp.array([x[6, :5, i] for i in digits]), axis=0).sum(axis=0)/x[6, :5, :-1].sum())(all_results)
    if "abs_foi_in_group_" in outcome:
        return jax.vmap(lambda x: x[5, :5, int(outcome.split("_")[-1])].sum(axis=0))(all_results)
    if "foi_in_group_" in outcome:
        return jax.vmap(lambda x: x[5, :5, int(outcome.split("_")[-1])].sum(axis=0)/x[5, :5, :-1].sum())(all_results) / kwargs.get('foi_scaling', 0.00016)
    if "population_test_" in outcome:
        return jax.vmap(lambda x: x[7, :5, int(outcome.split("_")[-1])].mean(axis=0))(all_results)
    return jax.jit(jax.vmap(lambda x: time_to_rebound(x, kwargs.get('threshold_factor', 1/2))))(all_results) / 365

def extract_target_value_from_data(pathogen, outcome, aggregation="D", NAG=7):
    print("pathogen:", pathogen)
    incidence = jnp.array(calculate_proportion_positive_incidence(pathogen, aggregation=aggregation, window_size=1, weighting_factor=0, sum_age_groups=False, save_counts=False, hosp=True, return_counts=True, pp_only=False, dedup=True, sac=True, NAG=NAG).values)
    incidence_summed_age = jnp.array(calculate_proportion_positive_incidence(pathogen, aggregation=aggregation, window_size=1, weighting_factor=0, sum_age_groups=True, save_counts=False, hosp=True, return_counts=True, pp_only=False, dedup=True, sac=True, NAG=NAG)["Total"].values)
    # pad with zeros: 14 days at the start, then enough at the end to complete full years
    pad_start = 14
    pad_end = (365 - ((len(incidence) + pad_start) % 365)) % 365
    incidence = jnp.pad(incidence,((pad_start, pad_end), (0, 0)),mode="constant",constant_values=0,)
    incidence_summed_age = jnp.pad(incidence_summed_age,(pad_start, pad_end),mode="constant",constant_values=0,)
    # replace nans with zeros
    incidence = jnp.nan_to_num(incidence)
    incidence_summed_age = jnp.nan_to_num(incidence_summed_age)
    PERIOD = pd.date_range(start=pd.to_datetime('2015-10-01'), end=pd.to_datetime('2025-10-01'), freq='D')
    POINTS = np.array(date_to_t(PERIOD))
    n_seasons = int((POINTS[-1] - POINTS[0]) / 365)
    days_to_keep = n_seasons * 365
    obs_curtailed = incidence[:days_to_keep, :]
    obs_summed_age_curtailed = incidence_summed_age[:days_to_keep]

    # Aggregate daily obs to monthly for suppression duration, matching the worker convention.
    days_per_month = 30
    n_months = days_to_keep // days_per_month
    DATES = pd.date_range(start=pd.to_datetime('2015-10-01'), periods=days_to_keep, freq='D')
    obs_df = pd.DataFrame(np.asarray(obs_curtailed), index=DATES)
    obs_sum_df = pd.Series(np.asarray(obs_summed_age_curtailed), index=DATES)
    obs_monthly_np = obs_df.resample('MS').sum().values
    obs_sum_monthly_np = obs_sum_df.resample('MS').sum().values
    obs_monthly = jnp.array(obs_monthly_np)
    obs_summed_age_monthly = jnp.array(obs_sum_monthly_np)
    anchor_date = pd.to_datetime("2020-01-01")
    monthly_dates = obs_df.resample('MS').sum().index
    anchor_month_idx = int(np.searchsorted(monthly_dates, anchor_date, side='right')) - 1
    suppression_duration = suppression_duration_by_age(
        obs_monthly,
        obs_summed_age_monthly,
        anchor_idx=anchor_month_idx,
        threshold_divisor=20.0,
    )  # duration in months

    # calculate obs per season by summing over each 365-day period, then concatenate the summed age version
    obs_per_season = obs_curtailed.reshape((n_seasons, 365, NAG)).sum(axis=1)
    obs_summed_age_per_season = obs_summed_age_curtailed.reshape((n_seasons, 365)).sum(axis=1)
    obs_per_season = jnp.concatenate([obs_per_season, obs_summed_age_per_season[:, None]], axis=1)
    # smooth the data over two weeks before calculating peaks to avoid noise causing spurious peaks
    kernel = jnp.ones(14)/14
    obs_curtailed = jax.vmap(lambda x: jnp.convolve(x, kernel, mode='same'), in_axes=1, out_axes=1)(obs_curtailed)
    obs_summed_age_curtailed = jnp.convolve(obs_summed_age_curtailed, kernel, mode='same')
    obs_reshaped = obs_curtailed.reshape((n_seasons, 365, NAG))
    days = jnp.arange(365)
    peak_times = jnp.sum(days[None, :, None] * obs_reshaped, axis=1) / (jnp.sum(obs_reshaped, axis=1) + 1e-10)
    obs_summed_age_reshaped = obs_summed_age_curtailed.reshape((n_seasons, 365))
    peak_times_summed_age = jnp.sum(days[None, :] * obs_summed_age_reshaped, axis=1) / (jnp.sum(obs_summed_age_reshaped, axis=1) + 1e-10)
    peak_times = jnp.concatenate([peak_times, peak_times_summed_age[:, None]], axis=1)
    # season info
    seasons = jnp.stack([obs_per_season, peak_times], axis=0)
    if outcome == "relative_size":
        value = relative_size_of_rebound(seasons)
    elif outcome == "age_ratio":
        value = age_ratio_of_rebound(seasons)
    elif outcome == "time_to_rebound":
        value = time_to_rebound(seasons)/365
    elif outcome == "age_time_shift":
        value = age_time_shift(seasons)
    elif outcome == "suppression_length":
        value = suppression_duration[-1]
    elif "suppression_length_in_group_" in outcome:
        value = suppression_duration[int(outcome.split("_")[-1])]
    elif outcome == "peak_times":
        value = peak_times
    return value

def get_parameter_values(samples, p_idx, r0_base=15):
    """Extracts and properly scales the specific parameter from the sample arrays."""
    scaled_samples = samples / PARAM_SCALING
    
    if p_idx == 0:  # R0
        return r0_base * scaled_samples[:, 2] / scaled_samples[:, 0]
    else:
        return scaled_samples[:, p_idx-1]

def extract_valid_data(samples, all_results, outcome="time_to_rebound", target_bounds=None, **kwargs):
    """Extracts targets and matching valid samples, filtering out NaNs or out-of-bounds results."""
    target_values = extract_target_values(all_results, outcome, **kwargs)
    valid_mask = ~jnp.isnan(target_values)
    
    valid_samples = np.asarray(samples[valid_mask])
    valid_targets = np.asarray(target_values[valid_mask])

    if target_bounds:
        low, high = target_bounds
        bound_mask = (valid_targets >= low) & (valid_targets <= high)
        valid_samples, valid_targets = valid_samples[bound_mask], valid_targets[bound_mask]

    return valid_samples, valid_targets

# ==========================================
# PLOTTING UTILITIES
# ==========================================
def add_2d_heatmap_figure(ax, x_vals, y_vals, valid_targets, vmin=None, vmax=None, outcome="time_to_rebound", use_scatter=False):
    if use_scatter:
        hist = ax.scatter(x_vals, y_vals, c=valid_targets, s=1, alpha=1, cmap=cm.viridis, vmin=vmin, vmax=vmax)
    else:
        ret = binned_statistic_2d(x_vals, y_vals, valid_targets, statistic='mean', bins=100, 
                                 range=[[x_vals.min(), x_vals.max()], [y_vals.min(), y_vals.max()]])
        hist = ax.imshow(ret.statistic.T, origin='lower', extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], 
                         cmap=cm.viridis, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.grid(True, alpha=0.3)
    return hist

def add_line_of_best_fit(ax, x_vals, y_vals):
    model = LinearRegression(fit_intercept=True)
    X_transformed = np.log(x_vals).reshape(-1,1)
    y_transformed = y_vals + 1
    model.fit(X_transformed, y_transformed)
    
    x_fit = np.linspace(1, 50, 1000)
    y_fit = model.predict(np.log(x_fit).reshape(-1,1)) - 1
    
    n = len(y_vals)
    x_mean = np.mean(np.log(x_vals))
    residuals = y_vals - (model.predict(X_transformed) - 1)
    mse = np.sum(residuals**2) / (n - 2)
    se = np.sqrt(mse * (1/n + (np.log(x_fit) - x_mean)**2 / np.sum((X_transformed- x_mean)**2)))
    t_val = stats.t.ppf(0.975, n-2)  # 95% CI
    ci_upper = np.minimum(y_fit + t_val * se, 0)
    ci_lower = np.maximum(y_fit - t_val * se, -1)

    valid_mask = (y_fit >= -1) & (y_fit <= 0)
    x_fit_masked, y_fit_masked = x_fit[valid_mask], y_fit[valid_mask]

    ci_valid_mask = (ci_upper >= -1) & (ci_lower <= 0)
    ci_upper_masked = ci_upper[ci_valid_mask]
    ci_lower_masked = ci_lower[ci_valid_mask]
    x_fit_ci_masked = x_fit[ci_valid_mask]
    
    ax.fill_between(x_fit_ci_masked, ci_lower_masked, ci_upper_masked, alpha=0.3, color='gray', label='95% CI')
    ax.plot(x_fit_masked, y_fit_masked, color='black', linestyle='--', label='Line of best fit')

def add_pathogen_labels(ax, good_simulations, p1=0, p2=8, NAG=7, color=None, r0_base=15):
    if color is None:
        color = ["white"] * len(good_simulations)
    for i, pathogen_info in enumerate(good_simulations):
        pathogen, seed, lockdown, option1, option2 = pathogen_info
        x = consistent_x_from_DE(pathogen, lockdown, option1, option2, seed, NAG=NAG, prefix="emcee_median_")
        if p1 == 0:
            val1 = r0_base * x[2] / x[0]
        else:
            val1 = x[p1-1] / PARAM_SCALING[p1-1]
        val2 = x[p2-1] / PARAM_SCALING[p2-1]
        ax.scatter(val1, val2, s=50, color=color[i], edgecolor='black', zorder=5)
        ax.annotate(PATHOGEN_SHORT_NAMES.get(pathogen, pathogen), (val1, val2), xytext=(5, -5),
                textcoords='offset points',  ha="left", color='black', zorder=4,
                path_effects=[pe.Stroke(linewidth=2, foreground='white'), pe.Normal()])

def add_extra_pathogens(ax):
    extra_names = ["Rotavirus", "Norovirus", "Measles", "Herpes"]
    extra_R0s, extra_R0s_upper, extra_R0s_lower = [17.5, 2, 13.2, 2.07], [18.2, 7.2, 44.4, 3], [5.03, 1.1, 4.6, 2]
    extra_immunity1, extra_immunity1_upper, extra_immunity1_lower = [-0.62, -0.74, 0, -1], [-0.83, -0.95, 0, -1], [-0.5, -0.57, 0, -1]
    
    for i, name in enumerate(extra_names):
        ax.scatter(extra_R0s[i], extra_immunity1[i], s=50, color='grey', edgecolor='black', zorder=4)
        ax.errorbar(extra_R0s[i], extra_immunity1[i], xerr=[[extra_R0s[i]-extra_R0s_lower[i]], [extra_R0s_upper[i]-extra_R0s[i]]],
                    yerr=[[extra_immunity1[i]-extra_immunity1_upper[i]], [extra_immunity1_lower[i]-extra_immunity1[i]]], fmt='o', color='grey', ecolor='black', zorder=3)
        ax.annotate(name, (extra_R0s[i], extra_immunity1[i]), xytext=(5, 5), textcoords='offset points', fontsize=18, ha="center", color='black', zorder=5,
                    path_effects=[pe.Stroke(linewidth=2, foreground='white'), pe.Normal()])

# ==========================================
# PIPELINE FUNCTIONS
# ==========================================
def run_simulation_pipeline(good_simulations, lockdown, POINTS, STATE0, p_time_to_obs, option1, option2, NAG=7,
                            seed=251118, n_samples=80000, dimension=2, chunk_size=40000, run_save_path=None):
    """Handles parameter sampling, environment setup, and executes simulation chunks."""
    print(f"Running simulation pipeline with lockdown={lockdown}, option1={option1}, option2={option2}, NAG={NAG}, seed={seed}")
    start_time = time.time()
    parameter_sets = parameter_space(good_simulations, NAG=NAG, n_samples=n_samples)
    print(f"Parameter space generated in {time.time() - start_time} seconds. Sampling {n_samples} points in {dimension}D space...")
    start_time = time.time()
    samples = lh_sampling(parameter_sets, n_samples, dimension=dimension)
    print(f"Samples generated in {time.time() - start_time} seconds.")

    if run_save_path is None:
        run_save_path = f"Outputs/sim_grid_lh_n{n_samples}_chunk{chunk_size}_seed{seed}_lockdown{lockdown}_{dimension}d"
    os.makedirs(run_save_path, exist_ok=True)
    with open(os.path.join(run_save_path, "samples.pickle"), "wb") as f: pickle.dump(np.asarray(samples), f)

    start_time = time.time()  
    _ = simulate_samples_chunked(samples, lockdown, POINTS, STATE0, p_time_to_obs, option1, option2, NAG=NAG,
                                           base_save_path=run_save_path, chunk_size=chunk_size, skip_existing=True)
    print(f"Simulations completed in {time.time() - start_time} seconds. Saved to {run_save_path}")
    
    return run_save_path

def generate_2d_heatmap_plot(ax, run_save_path, good_simulations, NAG=7, p1=0, p2=8, vmin=None, vmax=None, log=False, label=None, outcome="time_to_rebound", cbar=True, pathogen_vals=None, r0_base=15, **kwargs):
    """Generates the heatmap/histogram background with simply the main pathogens overlaid."""
    all_results, all_samples = load_and_recombine_results(run_save_path)
    if all_results is None:
        print("No simulation results found. Please run the simulation pipeline first.")
        return
        
    valid_samples, valid_targets = extract_valid_data(all_samples, all_results, outcome=outcome, **kwargs)

    x_vals = get_parameter_values(valid_samples, p1, r0_base=r0_base)
    y_vals = get_parameter_values(valid_samples, p2, r0_base=r0_base)

    if log:
        valid_targets = jnp.log(valid_targets)

    # Heatmap
    scatter = add_2d_heatmap_figure(ax, x_vals, y_vals, valid_targets, outcome=outcome, vmin=vmin, vmax=vmax)
    if label is None:
        if outcome == "time_to_rebound":
            label = "Time to re-emergence (years)"
        elif outcome == "outbreak_in_season":
            label = "Outbreak in 2022-23 season"
        elif outcome == "age_shift":
            label = r"Age shift in re-emergence"
        elif outcome == "age_time_shift":
            label = r"Relative peak shift in re-emergence (days)"
        elif outcome == "relative_size":
            label = "Relative size of rebound"
        elif outcome == "age_ratio":
            label = "Ratio of 1-4y to 3-12m in rebound vs pre-pandemic"
        elif outcome == "suppression_length":
            label = "Suppression length (months)"
        elif "suppression_length_in_group_" in outcome:
            age_group = int(outcome.split("_")[-1])
            label = f"Suppression length in age group {age_group} (months)"
        elif "abs_foi_in_group_" in outcome:
            age_group = int(outcome.split("_")[-1])
            label = f"Force of infection in age group {age_group}"
        elif "foi_in_group_" in outcome:
            age_group = int(outcome.split("_")[-1])
            label = f"Proportion of force of infection in age group {age_group}"
        elif "infectors_in_group_" in outcome:
            age_group = int(outcome.split("_")[-1])
            label = f"Proportion of infectors in age group {age_group}"
        elif "hospitalizors_in_group_" in outcome:
            age_group = int(outcome.split("_")[-1])
            label = f"Proportion of hospitalizations caused by infections from age group {age_group}"
    if cbar:
        plt.colorbar(scatter, ax=ax, label=label)

    if outcome in ["time_to_rebound", "relative_size", "age_ratio", "age_time_shift", "suppression_length"] or "suppression_length_in_group_" in outcome:
        if pathogen_vals is None:
            pathogen_vals = jnp.asarray([extract_target_value_from_data(pathogen, outcome=outcome, NAG=NAG) for pathogen in [good_simulations[i][0] for i in range(len(good_simulations))]])
        if log:
            pathogen_vals = jnp.log(pathogen_vals)
        pathogen_colors = cm.viridis((pathogen_vals - np.nanmin(valid_targets)) / (np.nanmax(valid_targets) - np.nanmin(valid_targets)))
        # where pathogen_vals is NA, set color to white
        pathogen_colors = [pathogen_colors[i] if not np.isnan(pathogen_vals[i]) else (1,1,1,1) for i in range(len(good_simulations))]
    else:
        pathogen_colors = ["white"] * len(good_simulations)
    add_pathogen_labels(ax, good_simulations, p1=p1, p2=p2, color=pathogen_colors, NAG=NAG, r0_base=r0_base)

    ax.set_xlabel(PARAMETER_NAMES[p1])
    ax.set_ylabel(PARAMETER_NAMES[p2])

    if "Immunity" in PARAMETER_NAMES[p1]: ax.set_xticklabels([f'{1.01+tick:.1f}' for tick in ax.get_xticks()])
    if "Immunity" in PARAMETER_NAMES[p2]: ax.set_yticklabels([f'{1.01+tick:.1f}' for tick in ax.get_yticks()])

    return np.min(valid_targets), np.max(valid_targets)

def generate_best_fit_plot(ax, good_simulations, p1=0, p2=8, NAG=7, r0_base=15):
    """Generates the scatter plot of pathogens + line of best fit."""
    
    val1, val2 = np.zeros(len(good_simulations)), np.zeros(len(good_simulations))
    for i, pathogen_info in enumerate(good_simulations):
        pathogen, seed, lockdown, option1, option2 = pathogen_info
        x = consistent_x_from_DE(pathogen, lockdown, option1, option2, seed)
        if p1 == 0:
            val1[i] = r0_base * x[2] / x[0]
            ax.set_xscale('log')
            ax.set_xticks([1,2,3,4,5,6,7,8,9])
            ax.set_xlim([0.901,10])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_xlabel("Basic reproduction number (log scale)")
        else:
            val1[i] = x[p1-1] * PARAM_SCALING[p1-1]
        val2[i] = x[p2-1] * PARAM_SCALING[p2-1]

    # Extra Pathogens & Line of Best Fit
    add_line_of_best_fit(ax, val1, val2)
    add_pathogen_labels(ax, good_simulations, p1=p1, p2=p2, NAG=NAG, r0_base=r0_base)
    add_extra_pathogens(ax)
    ax.set_xlabel(PARAMETER_NAMES[p1])
    ax.set_ylabel(PARAMETER_NAMES[p2])

    if "Immunity" in PARAMETER_NAMES[p1]: ax.set_xticklabels([f'{1.01+tick:.1f}' for tick in ax.get_xticks()])
    if "Immunity" in PARAMETER_NAMES[p2]: ax.set_yticklabels([f'{1.01+tick:.1f}' for tick in ax.get_yticks()])

    return ax

# function to plot a grid of the nine closest simulations to target parameter values
def plot_time_series_for_parameters(args, run_save_path, target_p1, target_p2, axes=None, p1=0, p2=8, r0_base=15):
    all_results, all_samples = load_and_recombine_results(run_save_path)

    valid_samples, _ = extract_valid_data(all_samples, all_results)
    
    parameter_values_p1 = get_parameter_values(valid_samples, p1, r0_base=r0_base)
    parameter_values_p2 = get_parameter_values(valid_samples, p2, r0_base=r0_base)
    
    # Calculate distance from target parameters
    distances = np.sqrt((parameter_values_p1 - target_p1)**2 + (parameter_values_p2 - target_p2)**2)
    
    # Find indices of 9 closest simulations
    closest_indices = np.argsort(distances)[:9]
    
    # Plot grid of 9 subplots
    if axes is None:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for plot_idx, closest_idx in enumerate(closest_indices):
        row, col = divmod(plot_idx, 3)
        current_ax = axes[row, col]
        
        # Load the time series data for this simulation
        closest_sample = valid_samples[closest_idx]
        print(closest_sample)
        lockdown_param, POINTS, STATE0, p_time_to_obs, option1, option2, NAG = args
        params = x_to_params(closest_sample, "sim", lockdown_param, option1+"mimmwane", option2+"nr", NAG=NAG, print_params=True)
        solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
        
        values = solution.ys.T
        trajectory = jnp.diff(values[-NAG:,:],axis=1).T
            # convolution of trajectory with probability of detection at each day after infection to get expected observations on each day
        p_time_to_obs_flipped = jnp.flip(p_time_to_obs.flatten())
        def obs_convolution(x):
            return jnp.convolve(x, p_time_to_obs_flipped, mode='same')
            # the expected observations for a given date are the observations on each day i days prvious multiplied by the probability of detection i days after infection
        expected_obs = jax.vmap(obs_convolution, in_axes=1, out_axes=1)(trajectory)
        expected_obs = jax.nn.softplus(expected_obs*100)/100

        color = cm.hsv(-0.02+np.arange(NAG)/NAG)
        color[3] = cm.hsv((3/NAG)+0.28/NAG)

        for i in range(NAG):
            current_ax.plot(POINTS[1:], expected_obs[:, i], color=color[i], linewidth=1.5)
        current_ax.set_title(f"Rank {plot_idx+1}: p1={parameter_values_p1[closest_idx]:.2f}, p2={parameter_values_p2[closest_idx]:.2f}")
        current_ax.set_xlabel("Time (days)")
        current_ax.set_ylabel("Observed infections")
        current_ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"9 Closest Simulations to Target: {PARAMETER_NAMES[p1]}={target_p1:.2f}, {PARAMETER_NAMES[p2]}={target_p2:.2f}")
    return axes

def plot_outcome_along_linear_combination(
    ax, 
    run_save_path, 
    good_simulations, 
    p1=0, 
    p2=8,
    outcome="time_to_rebound",
    direction='parallel', 
    method='projection', 
    num_bins=20, 
    slice_width=0.1, 
    log_target=False,
    r0_base=15,
    **kwargs
):
    """
    Plots the average value of an outcome along a specific linear combination 
    of log(R0) and Immunity.
    
    Parameters:
    - ax: matplotlib Axes object to plot on.
    - run_save_path: Path to LHS grid results.
    - good_simulations: List of pathogens defining the parameter space fit.
    - p1, p2: Parameter indices (default 0 for R0, 8 for Immunity).
    - outcome: Name of the outcome variable to load.
    - direction: 'parallel' (along line of best fit) or 'perpendicular'.
    - method: 'projection' (project all points) or 'slice' (filter points close to the line).
    - num_bins: Number of bins for averaging the outcome along the 1D path.
    - slice_width: If method='slice', the width of the band around the line to include points.
    - log_target: Whether to log-transform the outcome values.
    """
    
    # ==========================================
    # 1. Load Parameter Space Data (LHS Simulations)
    # ==========================================
    all_results, all_samples = load_and_recombine_results(run_save_path)
    if all_results is None:
        print("No simulation results found. Please run the simulation pipeline first.")
        return ax
        
    valid_samples, valid_targets = extract_valid_data(all_samples, all_results, outcome=outcome, **kwargs)

    x_vals_sim = get_parameter_values(valid_samples, p1, r0_base=r0_base)
    y_vals_sim = get_parameter_values(valid_samples, p2, r0_base=r0_base)
    outcome_vals_sim = jnp.log(valid_targets) if log_target else valid_targets
    
    x_sim_log = np.log(x_vals_sim)

    # ==========================================
    # 2. Load Pathogen Fits
    # ==========================================
    # Using the same extraction method as generate_best_fit_plot
    val1, val2 = np.zeros(len(good_simulations)), np.zeros(len(good_simulations))
    for i, pathogen_info in enumerate(good_simulations):
        pathogen, seed, lockdown, option1, option2 = pathogen_info
        x = consistent_x_from_DE(pathogen, lockdown, option1, option2, seed)
        if p1 == 0:
            val1[i] = 13.88 * x[2] / x[0]
        else:
            val1[i] = x[p1-1] * PARAM_SCALING[p1-1]
        val2[i] = x[p2-1] * PARAM_SCALING[p2-1]
        
    pathogen_x = val1
    pathogen_y = val2
    x_path_log = np.log(pathogen_x)

    # ==========================================
    # 3. Calculate Vectors & Projection
    # ==========================================
    # Fit the linear regression on the known pathogens
    model = LinearRegression()
    model.fit(x_path_log.reshape(-1, 1), pathogen_y)
    m = model.coef_[0]  # The slope
    
    # Define the direction vector in the 2D space (log(R0), Immunity)
    if direction == 'parallel':
        v = np.array([1.0, m])
    elif direction == 'perpendicular':
        v = np.array([m, -1.0])
    else:
        raise ValueError("Direction must be 'parallel' or 'perpendicular'.")
        
    v_norm = v / np.linalg.norm(v)
    n_norm = np.array([-v_norm[1], v_norm[0]]) 
    
    # Prepare coordinates
    points = np.column_stack([x_sim_log, y_vals_sim])
    positions_along_line = points @ v_norm
    perpendicular_positions = points @ n_norm
    
    # Find empirically the widest section of the simplex to drop the anchor
    perp_bins = np.linspace(perpendicular_positions.min(), perpendicular_positions.max(), 100)
    bin_indices = np.digitize(perpendicular_positions, perp_bins)
    
    max_range = 0
    optimal_perp_pos = perpendicular_positions.mean()
    
    for i in range(1, len(perp_bins)):
        in_bin = (bin_indices == i)
        if np.any(in_bin):
            current_range = positions_along_line[in_bin].max() - positions_along_line[in_bin].min()
            if current_range > max_range:
                max_range = current_range
                optimal_perp_pos = (perp_bins[i] + perp_bins[i-1]) / 2.0
                
    distances_from_line = np.abs(perpendicular_positions - optimal_perp_pos)
    
    # ==========================================
    # 4. Bin Data and Plot
    # ==========================================
    if method == 'slice':
        mask = distances_from_line <= (slice_width / 2.0)
        positions_along_line = positions_along_line[mask]
        outcome_vals_sim = outcome_vals_sim[mask]
        
        if len(positions_along_line) == 0:
            print(f"Warning: Slice width {slice_width} too narrow, no points found!")
            return ax
            
    # Bin the data along the optimal 1D axis
    bin_means, bin_edges, _ = stats.binned_statistic(
        positions_along_line, outcome_vals_sim, statistic='mean', bins=num_bins
    )
    
    bin_stds, _, _ = stats.binned_statistic(
        positions_along_line, outcome_vals_sim, statistic='std', bins=num_bins
    )
    bin_counts, _, _ = stats.binned_statistic(
        positions_along_line, outcome_vals_sim, statistic='count', bins=num_bins
    )
    
    with np.errstate(divide='ignore', invalid='ignore'):
        bin_sems = bin_stds / np.sqrt(bin_counts)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid_bins = ~np.isnan(bin_means)
    
    # Center the X axis relative to the middle of the longest path segment
    path_center = (np.max(positions_along_line) + np.min(positions_along_line)) / 2.0
    centered_x = bin_centers[valid_bins] - path_center

    ax.plot(
        centered_x, 
        bin_means[valid_bins], 
        color='blue' if direction == 'parallel' else 'purple', 
        linewidth=2, 
        label=f'{direction.title()} ({method})'
    )
    
    ax.fill_between(
        centered_x, 
        bin_means[valid_bins] - bin_sems[valid_bins], 
        bin_means[valid_bins] + bin_sems[valid_bins], 
        color='blue' if direction == 'parallel' else 'purple', 
        alpha=0.2
    )
    
    ax.set_xlabel(f"Distance from widest center along {direction} line")
    
    label_outcome = outcome.replace("_", " ").title()
    ax.set_ylabel(f"Average {label_outcome}" + (" (log)" if log_target else ""))
    ax.grid(True, alpha=0.3)
    # ax.legend()
    
    return ax

PARAM_SCALING = np.array([1, 1, 1, 1, 1, 1e-2, 1e-2, -1, -1, -1, -1, 1, 1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])
if __name__ == "__main__":
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif', 'font.serif': ['Palatino']})

    seed = 260612
    option1 = "dedupsac"
    NAG = 7 + ("split" in option1)
    if "split" in option1:
        from Parameters.census_population import CENSUS_AGE_POP_split as CENSUS_AGE_POP
        from Parameters.census_population import MEDIAN_AGE_split as MEDIAN_AGE
    if "sac" in option1:
        from Parameters.census_population import CENSUS_AGE_POP_sac as CENSUS_AGE_POP
        from Parameters.census_population import MEDIAN_AGE_sac as MEDIAN_AGE
    else:
        from Parameters.census_population import CENSUS_AGE_POP
        from Parameters.census_population import MEDIAN_AGE
    option2 = "flexagep05"
    lockdown = "ExponentialODipp25"
    CONTACT_MATRIX = jnp.asarray(pd.read_csv('Data/Processed/contact_matrices/KP'+['', '_mod']["cmod" in option2]+['', '_split'][NAG>7]+['', '_sac']["sac" in option1]+'_contact_all_US_Census.csv', delimiter=',', header=None).values)
    p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/Influenza_A_incubation_admittance_distribution.csv", delimiter=',', header=None).values)
    PERIOD = pd.date_range(start=pd.to_datetime('2015-09-17'), end=pd.to_datetime('2025-09-17'), freq='D')
    POINTS = np.array(date_to_t(PERIOD))
    ## Initial conditions
    STATE0 = jnp.zeros((2*N_S+1,NAG))
    STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
    STATE0 = STATE0.at[1,:].set(1)
    # # flatten initial state and add maternal immunity compartment
    STATE0 = STATE0.flatten()
    STATE0 = jnp.concatenate((jnp.array([0]), STATE0))
    good_simulations = [
        ["RSV", seed, lockdown, option1, "maxagep028"],["Metapneumovirus", seed, lockdown, option1, "maxagep015"],
        ["InfluenzaA", seed, lockdown, option1, "maxagep035"], 
        ["Adenovirus", seed, lockdown, option1, "maxagep003"],["Parainfluenza3", seed, lockdown, option1, "maxagep004"],
    ]
    
    r0_base = calculate_R0_from_values(1, 1, CONTACT_MATRIX, CENSUS_AGE_POP, jnp.zeros(NAG))
    # Parameter scaling factors used in the model
    if "Exponential" in lockdown:
        PARAM_SCALING = np.array([1, 1, 1, 1, 1, 1e-2, 1e-2, -1, -1, -1, -1, 1, 1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])
    elif lockdown == "RSV0415":
        PARAM_SCALING = np.array([1, 1, 1, 1, 1, 1e-2, 1e-2, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])
    if "split" in option1:
        PARAM_SCALING = np.concatenate((PARAM_SCALING, np.array([1e-2])))

    # # fig, ax = plt.subplots(figsize=(12, 6))
    # # generate_best_fit_plot(ax, good_simulations, p1=0, p2=8, r0_base=r0_base)
    # # plt.tight_layout()
    # # plt.savefig("Figures/line_of_best_fit_ExponentialODipLinearsac.png", dpi=300)
    # # # print("NAG", NAG)
    # run_save_path = "Outputs/sim_grid_lh_n80612_chunk10612_seed260612_lockdownExponentialODipp25_2d"
    run_save_path = run_simulation_pipeline(good_simulations, lockdown, POINTS, STATE0, p_time_to_obs, option1, option2, NAG=NAG,
                                            seed=seed, n_samples=10000, dimension=2, chunk_size=5000,
                                            # run_save_path=run_save_path
    )           
    # # print(run_save_path)

    # # # perpendicular / parallel plots
    # # fig, ax = plt.subplots(1, 2, figsize=(6.5,4))
    # # ax[0] = plot_outcome_along_linear_combination(ax[0], run_save_path, good_simulations, p1=0, p2=8, outcome="age_of_first_infection", direction='parallel', method='projection', num_bins=20, log_target=False)
    # # ax[1] = plot_outcome_along_linear_combination(ax[1], run_save_path, good_simulations, p1=0, p2=8, outcome="age_of_first_infection", direction='perpendicular', method='projection', num_bins=20, log_target=False)
    # # # ax[0].set_ylabel("Proportion of hospitalizations from <3m")
    # # # ax[1].set_ylabel("Proportion of hospitalizations from >65y")
    # # ax[0].set_xlabel("Relative sum of R0 and immunity")
    # # ax[1].set_xlabel("Relative excess in R0 vs immunity")
    # # plt.tight_layout()
    # # plt.savefig(f"Figures/along_linear_combination_age_of_first_infection_ExponentialODipEqualdedupsplit_AdVPIV3hMPV_lowerIHR2_{SHORT_PNAMES[0]}_{SHORT_PNAMES[8]}_projection.png", dpi=300)

    # ## single panel outcome heatmap
    fig, ax = plt.subplots(figsize=(4, 4))
    generate_2d_heatmap_plot(ax, run_save_path, good_simulations, NAG=NAG, p1=0, p2=8, outcome="age_ratio", cbar=True)
    plt.tight_layout()
    plt.savefig(f"Figures/heatmap_age_ratio_ExponentialODipp25_{SHORT_PNAMES[0]}_{SHORT_PNAMES[8]}.png", dpi=300)

    # # ## single panel outcome heatmap
    # # fig, ax = plt.subplots(figsize=(4, 4))
    # # generate_2d_heatmap_plot(ax, run_save_path, good_simulations, NAG=NAG, p1=0, p2=8, outcome="time_to_rebound", cbar=True)
    # # plt.tight_layout()
    # # plt.savefig(f"Figures/heatmap_time_to_rebound_ExponentialODipLinearsac_{SHORT_PNAMES[0]}_{SHORT_PNAMES[8]}.png", dpi=300)

    # # ## single panel outcome heatmap
    # # fig, ax = plt.subplots(figsize=(4, 4))
    # # generate_2d_heatmap_plot(ax, run_save_path, good_simulations, NAG=NAG, p1=0, p2=8, outcome="time_to_rebound", cbar=True, threshold_factor=1/3)
    # # plt.tight_layout()
    # # plt.savefig(f"Figures/heatmap_time_to_rebound_ExponentialODipLinearsac_{SHORT_PNAMES[0]}_{SHORT_PNAMES[8]}_threshold_third.png", dpi=300)
    
    # # two panel heatmap
    # fig, ax = plt.subplots(1, 2, figsize=(6.5, 4), sharex=True, sharey=True)
    # min, max = generate_2d_heatmap_plot(ax[0], run_save_path, good_simulations, NAG=NAG, p1=0, p2=8, outcome="hospitalizors_in_group_3", cbar=True, r0_base=r0_base)
    # print("Min and max for 5 to 17 years:", min, max)
    # min, max = generate_2d_heatmap_plot(ax[1], run_save_path, good_simulations, NAG=NAG, p1=0, p2=8, outcome="hospitalizors_in_group_6", cbar=True, r0_base=r0_base)
    # print("Min and max for >65y:", min, max)
    # ax[1].set_ylabel("")
    # ax[0].set_title("5 to 17 years")
    # ax[1].set_title("Over 65 years")
    # cbar0 = [c for c in ax[0].get_figure().axes if c != ax[0] and c != ax[1]][0] if len([c for c in ax[0].get_figure().axes if c != ax[0] and c != ax[1]]) > 0 else None
    # if cbar0 is not None:
    #     cbar0.set_ylabel("")
    # cbar1 = [c for c in ax[1].get_figure().axes if c != ax[0] and c != ax[1]][-1] if len([c for c in ax[1].get_figure().axes if c != ax[0] and c != ax[1]]) > 0 else None
    # if cbar1 is not None:
    #     cbar1.set_ylabel("Proportion of hospitalizations caused by\n infections from age group")
    # plt.savefig(f"Figures/heatmaps_hospitalizors_SACvs>65y_ExponentialODipLinearsac_{SHORT_PNAMES[0]}_{SHORT_PNAMES[8]}_medians.png", dpi=300)
    
    # # two panel heatmap
    # fig, ax = plt.subplots(1, 2, figsize=(6.5, 4), sharex=True, sharey=True)
    # min, max = generate_2d_heatmap_plot(ax[0], run_save_path, good_simulations, NAG=NAG, p1=0, p2=8, outcome="hospitalizors_in_group_4", cbar=False, r0_base=r0_base, vmin=0.32, vmax=0.43)
    # print("Min and max for 18 to 39 years:", min, max)
    # min, max = generate_2d_heatmap_plot(ax[1], run_save_path, good_simulations, NAG=NAG, p1=0, p2=8, outcome="hospitalizors_in_group_5", cbar=False, r0_base=r0_base, vmin=0.32, vmax=0.43)
    # print("Min and max for 40 to 64 years:", min, max)
    # ax[1].set_ylabel("")
    # ax[0].set_title("18 to 39 years")
    # ax[1].set_title("40 to 64 years")
    # fig.subplots_adjust(right=0.85)
    # cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    # norm = plt.Normalize(vmin=0.32, vmax=0.43)
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, cax=cbar_ax, label="Relative proportion of hospitalizations caused")
    # # plt.tight_layout()
    # plt.savefig(f"Figures/heatmaps_hospitalizors_18to39_vs_40to64_ExponentialODipLinearsac_{SHORT_PNAMES[0]}_{SHORT_PNAMES[8]}_test.png", dpi=300)
    

    # fig, ax = plt.subplots(4, 2, figsize=(6.5,8.5), sharex=True, sharey=True)
    # from Parameters.census_population import AGE_GROUP_NAMES_sac as AGE_GROUP_NAMES
    # overall_min, overall_max = float('inf'), float('-inf')
    # for age_group in range(NAG):
    #     current_ax = ax[age_group//2, age_group%2]
    #     min, max = generate_2d_heatmap_plot(current_ax, run_save_path, good_simulations, label="", NAG=NAG, p1=0, p2=8,
    #                                         cbar=True, vmin=None, vmax=None, log = False,
    #                                         outcome=f"infectors_in_group_{age_group}", foi_scaling=1)
    #     current_ax.set_title(AGE_GROUP_NAMES[age_group])
    #     overall_min = min if min < overall_min else overall_min
    #     overall_max = max if max > overall_max else overall_max
    #     current_ax.set_xlabel("")
    #     current_ax.set_ylabel("")
    # print("Overall min:", overall_min, "Overall max:", overall_max)
    # fig.supxlabel(f"{PARAMETER_NAMES[0]}", fontsize=11)
    # fig.supylabel(f"{PARAMETER_NAMES[8]}", fontsize=11)

    # # # # add label on right for colorbars
    # fig.text(0.95, 0.5, "Proportion of hospitalization-causing infectors in age group", va='center', rotation='vertical', fontsize=11)

    # # move figure to make space for colorbar
    # fig.subplots_adjust(right=0.8)
    # # add overall colorbar
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    # norm = plt.Normalize(vmin=overall_min, vmax=overall_max)
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, cax=cbar_ax, label="Proportion of infectors in age group")

    # plt.tight_layout()
    # plt.savefig(f"Figures/heatmaps_infectors_ExponentialDipLinearsac_{SHORT_PNAMES[0]}_{SHORT_PNAMES[8]}.png", dpi=300)
    
    # # # args = (lockdown, POINTS, STATE0, p_time_to_obs, option1, option2, NAG)
    # # # plot_time_series_for_parameters(args, run_save_path, target_p1=0.250, target_p2=-0.299, p1=2, p2=8)
    # # # plt.tight_layout()
    # # # plt.savefig(f"Figures/close_to_RSV_260415_{SHORT_PNAMES[0]}_{SHORT_PNAMES[8]}.png", dpi=300)


    # pathogen_vals = jnp.asarray([extract_target_value_from_data(pathogen, outcome="suppression_length", NAG=NAG) for pathogen in [good_simulations[i][0] for i in range(len(good_simulations))]])
    # # for p1 in range(15,len(PARAMETER_NAMES)):
    # for p1 in range(len(PARAMETER_NAMES)):
    #     for p2 in range(p1+1, len(PARAMETER_NAMES)):
    #         fig, ax = plt.subplots(figsize=(10, 8))
    #         generate_2d_heatmap_plot(ax, run_save_path, good_simulations, p1=p1, p2=p2, NAG=NAG, outcome="suppression_length", threshold_factor=1/3, pathogen_vals=pathogen_vals)
    #         plt.tight_layout()
    #         plt.savefig(f"Figures/sim_grids260609/heatmap_suppression_length_{SHORT_PNAMES[p1]}_{SHORT_PNAMES[p2]}_thresholdthird.png", dpi=300)
    # for p1 in range(len(PARAMETER_NAMES)):
    #     for p2 in range(p1+1, len(PARAMETER_NAMES)):
    #         fig, ax = plt.subplots(figsize=(10, 8))
    #         generate_2d_heatmap_plot(ax, run_save_path, good_simulations, p1=p1, p2=p2, outcome="hospitalizors_in_group_0")
    #         plt.tight_layout()
    #         plt.savefig(f"Figures/sim_grids260609/heatmap_hospitalizors_in_group_0_{SHORT_PNAMES[p1]}_{SHORT_PNAMES[p2]}_thresholdthird.png", dpi=300)
