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

from utils import consistent_x_from_DE, x_to_params, date_to_t
from fit_MCMC import run_simulation
from data_processing import calculate_proportion_positive_incidence
from Parameters.census_population import CENSUS_AGE_POP, MEDIAN_AGE
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
NAG = 7  # Number of age groups
N_S = 3  # Susceptibility classes

PERIOD = pd.date_range(start=pd.to_datetime('2015-09-17'), end=pd.to_datetime('2025-09-17'), freq='D')
POINTS = np.array(date_to_t(PERIOD))
## Initial conditions
STATE0 = jnp.zeros((2*N_S+1,NAG))
STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
STATE0 = STATE0.at[1,:].set(1)
# # flatten initial state and add maternal immunity compartment
STATE0 = STATE0.flatten()
STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

PARAMETER_NAMES = [
    "Reproduction number", "First infection duration", "Second infection duration", 
    "Transmissibility", "Seasonality", "Phase", "Waning (per 100 days after first infection)", 
    "Waning (per 100 days after second infection)", "Immunity from first infection", 
    "Immunity from second infection", "Immunity to severe disease after first infection",
    "Immunity to severe disease after second infection", "Maternal immunity",
    "Contact reduction", "Contact recovery",
    "Disease susceptibility <3m", "Disease susceptibility 3–11m", "Disease susceptibility 1–4y",
    "Disease susceptibility 5–7y", "Disease susceptibility 8–49y", "Disease susceptibility 50-64y", 
    "Disease susceptibility 65+y"
]

SHORT_PNAMES = [
    "R0", "rec_up", "rec_same", "transmissibility", "seasonality", "phase", "wane1", "waning", 
    "immunity1", "immunity2", "disease_immunity1", "disease_immunity2", "maternal_immunity", 
    "contact_reduction", "contact_recovery",
    "susceptibility_0_3m", "susceptibility_3_11m", "susceptibility_1_4y", "susceptibility_5_7y", 
    "susceptibility_8_49y", "susceptibility_50_64y", "susceptibility_65y"
]

PATHOGEN_SHORT_NAMES = {
    "RSV": "RSV", "Metapneumovirus": "hMPV", "InfluenzaA": "FluA", 
    "InfluenzaB": "FluB", "Adenovirus": "AdV", "Parainfluenza3": "PIV3"
}

# ==========================================
# DEFINE SAMPLING SPACE
# ==========================================
def parameter_space(good_simulations):
    """Extracts parameter sets from good simulations for use in sampling."""
    parameter_sets = []
    for pathogen_info in good_simulations:
        pathogen, seed, lockdown, option1, option2 = pathogen_info
        x = consistent_x_from_DE(pathogen, lockdown, option1, option2, seed)
        parameter_sets.append(x)
    parameter_sets = jnp.array(parameter_sets)
    return(parameter_sets)

def lh_sampling(parameter_sets, n_samples, dimension=None):
    """Generates samples from the parameter space using Latin Hypercube Sampling.
    Args:
        parameter_sets: Array of shape (n_pathogens, n_parameters) containing parameter sets from good simulations.
        n_samples: Number of samples to generate.
        dimension: The dimension of the simplex to sample from. If None, samples from the full simplex.
                   If 0, samples from vertices; if 1, samples from edges; if k, samples from k-dimensional faces.
    Returns:
        samples: Array of shape (n_samples, n_parameters) containing the sampled parameter sets.
    """
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
        samples = parameter_sets[vertex_indices]
        return samples
    
    elif dimension == 1:
        # Sample from edges (between pairs of pathogens)
        # Choose random pairs of pathogens
        # pairs = np.random.choice(n_pathogens, size=(n_samples, 2), replace=True)
        pairs = np.array(list(it.combinations(range(n_pathogens), 2)))
        pairs = np.tile(pairs, (int(np.ceil(n_samples / pairs.shape[0])), 1))[:n_samples]
        # Generate weights for each pair
        weights = jnp.arange(n_samples)/(n_samples-1)
        
        samples = []
        for i in range(n_samples):
            idx1, idx2 = pairs[i]
            w = weights[i]
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
            # For lower-dimensional faces, we need to select which subset of pathogens to use
            # and set the rest to zero weights
            
            # For each sample, randomly select which (dimension+1) pathogens to use
            samples = []
            for i in range(n_samples):
                # Choose (dimension+1) pathogens for this sample
                selected_pathogens = np.random.choice(n_pathogens, size=dimension+1, replace=False)
                
                # Create barycentric coordinates for selected pathogens
                weights = jnp.zeros(n_pathogens)
                selected_exp = exp_samples[i]
                # Add one more dimension to complete the simplex for selected pathogens
                last_coord = np.random.exponential(1.0)
                full_exp = jnp.concatenate([selected_exp, jnp.array([last_coord])])
                
                # Normalize and assign to selected pathogens
                normalized_weights = full_exp / jnp.sum(full_exp)
                weights = weights.at[selected_pathogens].set(normalized_weights)
                
                # Compute sample as convex combination
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
            samples = lh_samples @ parameter_sets
            
            return samples

# ==========================================
# SIMULATION FUNCTIONS AND BATCHING SYSTEM
# ==========================================

def worker(args):
    """Worker function to run a single simulation and extract relevant metrics.
    Args:
        args: A tuple containing (sample, lockdown, POINTS, STATE0, p_time_to_obs)
    Returns:
        A 3D array containing the following metrics for each season and age group:
        - Observed infections per season (shape: n_seasons x NAG)
        - Time of peak incidence per season (shape: n_seasons x NAG)
        - Proportion of first infectious compartment per season (shape: n_seasons x NAG)
        - Proportion of all infectious compartments per season (shape: n_seasons x NAG)
        - Infections caused by each age group per season (shape: n_seasons x NAG)
        - Force of infection experienced by each age group per season (shape: n_seasons x NAG)
    """
    sample, lockdown, POINTS, STATE0, p_time_to_obs, option1, option2 = args
    params = x_to_params(sample, "sim", lockdown, option1+"mimmwane", option2+"nr")
    solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS)
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
    # Curtail obs to fit exact seasons (ignore partial days due to leap years)
    days_to_keep = n_seasons * 365
    obs_curtailed = obs[:days_to_keep, :]
    obs_summed_age_curtailed = obs_summed_age[:days_to_keep]
    obs_per_season = obs_curtailed.reshape((n_seasons, 365, NAG)).sum(axis=1)
    obs_summed_age_per_season = obs_summed_age_curtailed.reshape((n_seasons, 365)).sum(axis=1)
    # concatenate to obs_per_season
    obs_per_season = jnp.concatenate([obs_per_season, obs_summed_age_per_season[:, None]], axis=1)
    # find the time of peak incidence in each age group for each season
    peak_times = jnp.argmax(obs_curtailed.reshape((n_seasons, -1, NAG)), axis=1)
    peak_times_summed_age = jnp.argmax(obs_summed_age_curtailed.reshape((n_seasons, -1)), axis=1)
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
    # Shape: (NAG_causing, NAG_receiving, time)
    foi_matrix = params[4] * params[3][:, :, None] * all_infectious[None, :, :]
    # Calculate new infections: susceptible * foi * susceptibility by class
    # susceptible shape: (N_S, NAG, time)
    # foi_matrix shape: (NAG_causing, NAG_receiving, time)
    # params[6] shape: (N_S,)
    infections_matrix = params[6][:, None, None, None] * foi_matrix[None, :, :, :] * susceptible[:, None, :, :]
    # Sum over susceptibility classes and receiving age groups to get infections caused by each age group
    # Shape: (NAG_causing, time)
    infections_caused_by_age = infections_matrix.sum(axis=(0, 2))
    # Reshape and sum by season
    infections_caused_by_age_by_season = infections_caused_by_age.reshape((NAG, n_seasons, 365)).sum(axis=2).T
    # summed version for consistency
    infections_caused_by_age_summed = infections_caused_by_age.sum(axis=0)
    infections_caused_by_age_summed_by_season = infections_caused_by_age_summed.reshape((n_seasons, 365)).sum(axis=1)
    infections_caused_by_age_by_season = jnp.concatenate([infections_caused_by_age_by_season,
    infections_caused_by_age_summed_by_season[:, None]], axis=1)
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
    return jnp.stack([obs_per_season, peak_times, proportion_first_infectious, proportion_infectious, infections_caused_by_age_by_season, foi_experienced_by_age_by_season], axis=0)

def simulate_samples_chunked(samples, lockdown, POINTS, STATE0, p_time_to_obs, option1, option2,
                             base_save_path, chunk_size, skip_existing=False):
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
        return worker((sample, lockdown, POINTS, STATE0, p_time_to_obs, option1, option2))
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

def relative_size_of_rebound(x, threshold_factor=1):
    obs_summed = x[0, :, -1]
    pre_pandemic_median = jnp.median(obs_summed[:5])
    post_pandemic_max = jnp.max(obs_summed[5:])
    return jnp.where(post_pandemic_max < threshold_factor*pre_pandemic_median, jnp.nan, post_pandemic_max / pre_pandemic_median)

def age_ratio_of_rebound(x, idx_num=2, idx_den=1):
    """Calculate the ratio of infections in one age group to another during the rebound season, relative to the pre-pandemic ratio.
    Args:
        x: The 3D array of metrics returned by the worker function, with shape (n_metrics, n_seasons, NAG+1). The last age group (index -1) corresponds to the summed age group.
        idx_num: The index of the age group to use as the numerator in the ratio (default is 2, which corresponds to 1-4y).
        idx_den: The index of the age group to use as the denominator in the ratio (default is 1, which corresponds to 3-12m).
    Returns: The ratio of the specified age groups during the rebound season, relative to the pre-pandemic ratio. If there is no rebound, returns NaN.
    """
    obs_per_season = x[0, :, :]
    pre_pandemic_median_ratio = jnp.median(obs_per_season[:5, idx_num] / obs_per_season[:5, idx_den])
    post_pandemic_obs = obs_per_season[5:, :]
    rebound_season_idx = jnp.argmax(jnp.sum(post_pandemic_obs, axis=1))
    rebound_obs = post_pandemic_obs[rebound_season_idx]
    
    return (rebound_obs[idx_num] / rebound_obs[idx_den]) / pre_pandemic_median_ratio

def age_of_first_infection(x):
    return jnp.mean(jnp.sum(x[2, :5, :-1] * jnp.array(MEDIAN_AGE), axis=1)) / 12

def age_of_infector(x):
    return jnp.mean(jnp.sum(x[4, :5, :-1] * jnp.array(MEDIAN_AGE), axis=1) / jnp.sum(x[4, :5, :-1], axis=1)) / 12

def foi_weighted_age(x):
    return jnp.mean(jnp.sum(x[5, :5, :-1] * jnp.array(MEDIAN_AGE), axis=1) / jnp.sum(x[5, :5, :-1], axis=1)) / 12

# ==========================================
# ANALYSIS & PROCESSING
# ==========================================
def extract_target_values(all_results, outcome, **kwargs):
    if outcome == "relative_size": return jax.jit(jax.vmap(relative_size_of_rebound))(all_results)
    if outcome == "age_ratio": return jax.jit(jax.vmap(lambda x: age_ratio_of_rebound(x, kwargs.get('idx_num', 2), kwargs.get('idx_den', 1))))(all_results)
    if outcome == "age_of_first_infection": return jax.jit(jax.vmap(age_of_first_infection))(all_results)
    if "infectors_in_class_" in outcome:
        return jax.vmap(lambda x: x[4, :5, int(outcome.split("_")[-1])].sum(axis=0)/x[4, :5, :-1].sum())(all_results)
    if "abs_foi_in_class_" in outcome:
        return jax.vmap(lambda x: x[5, :5, int(outcome.split("_")[-1])].sum(axis=0))(all_results)
    if "foi_in_class_" in outcome:
        return jax.vmap(lambda x: x[5, :5, int(outcome.split("_")[-1])].sum(axis=0)/x[5, :5, :-1].sum())(all_results) / kwargs.get('foi_scaling', 0.00016)
    return jax.jit(jax.vmap(time_to_rebound))(all_results) / 365

def extract_target_value_from_data(pathogen, outcome, aggregation="D"):
    print("pathogen:", pathogen)
    incidence = jnp.array(calculate_proportion_positive_incidence(pathogen, aggregation=aggregation, window_size=1, weighting_factor=0, sum_age_groups=False, save_counts=False, pp_only=False).values)
    incidence_summed_age = jnp.array(calculate_proportion_positive_incidence(pathogen, aggregation=aggregation, window_size=1, weighting_factor=0, sum_age_groups=True, save_counts=False, pp_only=False)["Total"].values)
    # pad with zeros: 14 days at the start, then enough at the end to complete full years
    pad_start = 14
    pad_end = (365 - ((len(incidence) + pad_start) % 365)) % 365
    incidence = jnp.pad(incidence,((pad_start, pad_end), (0, 0)),mode="constant",constant_values=0,)
    incidence_summed_age = jnp.pad(incidence_summed_age,(pad_start, pad_end),mode="constant",constant_values=0,)
    # replace nans with zeros
    incidence = jnp.nan_to_num(incidence)
    incidence_summed_age = jnp.nan_to_num(incidence_summed_age)
    n_seasons = int((POINTS[-1] - POINTS[0]) / 365)
    days_to_keep = n_seasons * 365
    obs_curtailed = incidence[:days_to_keep, :]
    obs_summed_age_curtailed = incidence_summed_age[:days_to_keep]
    # calculate obs per season by summing over each 365-day period, then concatenate the summed age version
    obs_per_season = obs_curtailed.reshape((n_seasons, 365, NAG)).sum(axis=1)
    obs_summed_age_per_season = obs_summed_age_curtailed.reshape((n_seasons, 365)).sum(axis=1)
    obs_per_season = jnp.concatenate([obs_per_season, obs_summed_age_per_season[:, None]], axis=1)
    # smooth the data over two weeks before calculating peaks to avoid noise causing spurious peaks
    kernel = jnp.ones(14)/14
    obs_curtailed = jax.vmap(lambda x: jnp.convolve(x, kernel, mode='same'), in_axes=1, out_axes=1)(obs_curtailed)
    obs_summed_age_curtailed = jnp.convolve(obs_summed_age_curtailed, kernel, mode='same')
    peak_times = jnp.argmax(obs_curtailed.reshape((n_seasons, -1, NAG)), axis=1)
    peak_times_summed_age = jnp.argmax(obs_summed_age_curtailed.reshape((n_seasons, -1)), axis=1)
    peak_times = jnp.concatenate([peak_times, peak_times_summed_age[:, None]], axis=1)
    # season info
    seasons = jnp.stack([obs_per_season, peak_times], axis=0)
    if outcome == "relative_size":
        return relative_size_of_rebound(seasons)
    elif outcome == "age_ratio":
        return age_ratio_of_rebound(seasons)
    elif outcome == "time_to_rebound":
        return time_to_rebound(seasons)/365

def get_parameter_values(samples, p_idx):
    """Extracts and properly scales the specific parameter from the sample arrays."""
    scaled_samples = samples / PARAM_SCALING
    
    if p_idx == 0:  # R0
        return 13.88 * scaled_samples[:, 2] / scaled_samples[:, 0]
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
    
    ax.fill_between(x_fit, ci_lower, ci_upper, alpha=0.3, color='gray', label='95% CI')
    ax.plot(x_fit_masked, y_fit_masked, color='black', linestyle='--', label='Line of best fit')

def add_pathogen_labels(ax, good_simulations, p1=0, p2=8, color=None):
    if color is None:
        color = ["white"] * len(good_simulations)
    for i, pathogen_info in enumerate(good_simulations):
        pathogen, seed, lockdown, option1, option2 = pathogen_info
        x = consistent_x_from_DE(pathogen, lockdown, option1, option2, seed)
        if p1 == 0:
            val1 = 13.88 * x[2] / x[0]
        else:
            val1 = x[p1-1] / PARAM_SCALING[p1-1]
        val2 = x[p2-1] / PARAM_SCALING[p2-1]
        ax.scatter(val1, val2, s=50, color=color[i], edgecolor='black', zorder=5)
        ax.annotate(PATHOGEN_SHORT_NAMES.get(pathogen, pathogen), (val1, val2), xytext=(5, -15),
                textcoords='offset points', fontsize=18, ha="center", color='black', zorder=4,
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
def run_simulation_pipeline(good_simulations, lockdown, POINTS, STATE0, p_time_to_obs, option1, option2,
                            seed=251118, n_samples=80000, dimension=2, chunk_size=40000):
    """Handles parameter sampling, environment setup, and executes simulation chunks."""
    parameter_sets = parameter_space(good_simulations)
    samples = lh_sampling(parameter_sets, n_samples, dimension=dimension)

    run_save_path = f"Outputs/sim_grid_lh_n{n_samples}_chunk{chunk_size}_seed{seed}_lockdown{lockdown}_{dimension}d"
    os.makedirs(run_save_path, exist_ok=True)
    with open(os.path.join(run_save_path, "samples.pickle"), "wb") as f: pickle.dump(np.asarray(samples), f)

    start_time = time.time()  
    _ = simulate_samples_chunked(samples, lockdown, POINTS, STATE0, p_time_to_obs, option1, option2,
                                           base_save_path=run_save_path, chunk_size=chunk_size, skip_existing=True)
    print(f"Simulations completed in {time.time() - start_time} seconds. Saved to {run_save_path}")
    
    return run_save_path

def generate_2d_heatmap_plot(ax, run_save_path, good_simulations, p1=0, p2=8, outcome="time_to_rebound"):
    """Generates the heatmap/histogram background with simply the main pathogens overlaid."""
    all_results, all_samples = load_and_recombine_results(run_save_path)
    if all_results is None:
        print("No simulation results found. Please run the simulation pipeline first.")
        return
        
    valid_samples, valid_targets = extract_valid_data(all_samples, all_results, outcome=outcome)

    x_vals = get_parameter_values(valid_samples, p1)
    y_vals = get_parameter_values(valid_samples, p2)

    # Heatmap
    scatter = add_2d_heatmap_figure(ax, x_vals, y_vals, valid_targets, outcome=outcome)
    if outcome == "time_to_rebound":
        label = "Time to re-emergence (years)"
    elif outcome == "relative_size":
        label = "Relative size of rebound"
    elif outcome == "age_ratio":
        label = "Ratio of 1-4y to 3-12m in rebound vs pre-pandemic"
    elif "abs_foi_in_class_" in outcome:
        age_group = int(outcome.split("_")[-1])
        label = f"Force of infection in age group {age_group}"
    elif "foi_in_class_" in outcome:
        age_group = int(outcome.split("_")[-1])
        label = f"Proportion of force of infection in age group {age_group}"
    elif "infectors_in_class_" in outcome:
        age_group = int(outcome.split("_")[-1])
        label = f"Proportion of infectors in age group {age_group}"
    plt.colorbar(scatter, ax=ax, label=label)

    if outcome in ["time_to_rebound", "relative_size", "age_ratio"]:
        pathogen_vals = [extract_target_value_from_data(pathogen, outcome=outcome) for pathogen in [good_simulations[i][0] for i in range(len(good_simulations))]]
        pathogen_colors = cm.viridis((jnp.array(pathogen_vals) - np.nanmin(valid_targets)) / (np.nanmax(valid_targets) - np.nanmin(valid_targets)))
        # where pathogen_vals is NA, set color to white
        pathogen_colors = [pathogen_colors[i] if not np.isnan(pathogen_vals[i]) else (1,1,1,1) for i in range(len(good_simulations))]
    else:
        pathogen_colors = ["white"] * len(good_simulations)
    add_pathogen_labels(ax, good_simulations, p1=p1, p2=p2, color=pathogen_colors)

    ax.set_xlabel(PARAMETER_NAMES[p1])
    ax.set_ylabel(PARAMETER_NAMES[p2])

    if "Immunity" in PARAMETER_NAMES[p1]: ax.set_xticklabels([f'{1.01+tick:.1f}' for tick in ax.get_xticks()])
    if "Immunity" in PARAMETER_NAMES[p2]: ax.set_yticklabels([f'{1.01+tick:.1f}' for tick in ax.get_yticks()])

    return ax

def generate_best_fit_plot(ax, good_simulations, p1=0, p2=8):
    """Generates the scatter plot of pathogens + line of best fit."""
    
    val1, val2 = np.zeros(len(good_simulations)), np.zeros(len(good_simulations))
    for i, pathogen_info in enumerate(good_simulations):
        pathogen, seed, lockdown, option1, option2 = pathogen_info
        x = consistent_x_from_DE(pathogen, lockdown, option1, option2, seed)
        if p1 == 0:
            val1[i] = 13.88 * x[2] / x[0]
            ax.set_xscale('log')
            ax.set_xticks([1,2,5,10,20,40])
            ax.set_xlim([0.901,60])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_xlabel("Basic reproduction number (log scale)")
        else:
            val1[i] = x[p1-1] * PARAM_SCALING[p1-1]
        val2[i] = x[p2-1] * PARAM_SCALING[p2-1]

    # Extra Pathogens & Line of Best Fit
    add_line_of_best_fit(ax, val1, val2)
    add_pathogen_labels(ax, good_simulations, p1=p1, p2=p2)
    add_extra_pathogens(ax)
        
    ax.set_xlabel(PARAMETER_NAMES[p1])
    ax.set_ylabel(PARAMETER_NAMES[p2])

    if "Immunity" in PARAMETER_NAMES[p1]: ax.set_xticklabels([f'{1.01+tick:.1f}' for tick in ax.get_xticks()])
    if "Immunity" in PARAMETER_NAMES[p2]: ax.set_yticklabels([f'{1.01+tick:.1f}' for tick in ax.get_yticks()])

    return ax


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 18, 'font.family': 'serif', 'font.serif': ['Palatino']})

    seed = 2604154
    option1 = "maxmimmsplit"
    option2 = "daycarep5maxagep02"
    lockdown = "Exponential"
    p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/Influenza_A_incubation_admittance_distribution.csv", delimiter=',', header=None).values)
    good_simulations = [
        ["RSV", 260415, lockdown, option1, "daycarep5maxagep028"], ["Metapneumovirus", 260415, lockdown, option1, option2], 
        ["InfluenzaA", 260415, lockdown, option1, "daycarep5maxagep05"], ["InfluenzaB", 260415, lockdown, option1, "daycarep5maxagep05"], 
        ["Adenovirus", 260415, lockdown, option1, option2], ["Parainfluenza3", 260415, lockdown, option1, option2]
    ]
    
    # Parameter scaling factors used in the model
    if lockdown == "Exponential":
        PARAM_SCALING = np.array([1, 1, 1, 1, 1, 1e-2, 1e-2, -1, -1, -1, -1, 1, 1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])
    elif lockdown == "RSV0415":
        PARAM_SCALING = np.array([1, 1, 1, 1, 1, 1e-2, 1e-2, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])

    fig, ax = plt.subplots(figsize=(12, 6))
    generate_best_fit_plot(ax, good_simulations, p1=0, p2=8)
    plt.tight_layout()
    plt.savefig("Figures/line_of_best_fit_Exponential_maxmimmsplit_daycarep5maxagep02.png", dpi=300)

    run_save_path = run_simulation_pipeline(good_simulations, lockdown, POINTS, STATE0, p_time_to_obs, option1, option2,
                                            seed=seed, n_samples=80000, dimension=2, chunk_size=40000)
    
    run_save_path = "Outputs/sim_grid_lh_n80000_chunk40000_seed2604153_lockdownExponential_2d"
    fig, ax = plt.subplots(1, 2, figsize=(13,6.5))
    generate_2d_heatmap_plot(ax[0], run_save_path, good_simulations, p1=0, p2=8, outcome="time_to_rebound")
    generate_2d_heatmap_plot(ax[1], run_save_path, good_simulations, p1=0, p2=8, outcome="age_ratio")
    plt.tight_layout()
    plt.savefig(f"Figures/heatmaps_time_age_Exponential_RSVmaxmimm_daycarep5maxagep02_{SHORT_PNAMES[0]}_{SHORT_PNAMES[8]}_thresholdp5.png", dpi=300)
    # for p1 in range(len(PARAMETER_NAMES)):
    #     for p2 in range(p1+1, len(PARAMETER_NAMES)):
    #         fig, ax = plt.subplots(figsize=(10, 8))
    #         generate_2d_heatmap_plot(ax, run_save_path, good_simulations, p1=p1, p2=p2, outcome="time_to_rebound")
    #         plt.tight_layout()
    #         plt.savefig(f"Figures/sim_grids260415/heatmap_time_to_rebound_daycare_factortwothirds_{SHORT_PNAMES[p1]}_{SHORT_PNAMES[p2]}.png", dpi=300)
