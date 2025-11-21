import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
import itertools as it
import pandas as pd
import time
import pickle
import sys
import os
import re
import glob
from utils import consistent_x_from_DE, x_to_params, date_to_t
from fit_MCMC import run_simulation
from Parameters.census_population import CENSUS_AGE_POP, AGE_GROUP_NAMES, MEDIAN_AGE
from Parameters.times_and_contacts import PERIOD
from plotting import lockdown_incidence_plot
from scipy.stats import qmc
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import binned_statistic_2d
from scipy.spatial import ConvexHull

# Load opt.x for each pathogen and find a polygon that captures the parameter space
def parameter_space(good_simulations):
    parameter_sets = []
    for pathogen_info in good_simulations:
        pathogen, seed, option1 = pathogen_info
        x = consistent_x_from_DE(pathogen, option1, seed)
        parameter_sets.append(x)
    parameter_sets = jnp.array(parameter_sets)
    return(parameter_sets)


def lh_sampling(parameter_sets, n_samples, dimension=None):
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

# Worker function defined at module level for pickling
NAG = 7  # Number of age groups
def worker(args):
    sample, lockdown, POINTS, STATE0, p_time_to_obs = args
    params = x_to_params(sample, "sim", lockdown, "mimmwane", "flexage")
    solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS)
    # find total observed infections each season in each age group
    values = solution.ys.T
    trajectory = jnp.diff(values[-NAG:,:], axis=1).T
    p_time_to_obs_flipped = jnp.flip(p_time_to_obs.flatten())
    @jax.jit
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

def simulate_samples_chunked(samples, lockdown, POINTS, STATE0, p_time_to_obs, 
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
    # --- JAX vmap path ---
    print("Using JAX vmap (n_workers=1) with chunking.")
    @jax.jit
    def partial_worker(sample):
        return worker((sample, lockdown, POINTS, STATE0, p_time_to_obs))
    
    vmapped_worker = jax.vmap(partial_worker)
    
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
    """
    Loads all pickled chunks from a specific run directory,
    recombines them, and returns both results and samples arrays.
    """
    
    # Find all chunk files in the directory
    search_path = os.path.join(run_save_path, "chunk_*.pickle")
    chunk_files = glob.glob(search_path)
    
    # Find all sample chunk files
    samples_search_path = os.path.join(run_save_path, "samples_chunk_*.pickle")
    sample_chunk_files = glob.glob(samples_search_path)
    
    if not chunk_files:
        print(f"No chunk files found in: {run_save_path}")
        return None
        
    # --- Sort the files numerically ---
    def get_chunk_number(filepath):
        # Extracts the number '0001' from '.../chunk_0001.pickle' or '.../samples_chunk_0001.pickle'
        filename = os.path.basename(filepath)
        match = re.search(r'chunk_(\d+)\.pickle', filename)
        if match:
            return int(match.group(1))
        return -1 # Should not happen if files are named correctly

    chunk_files.sort(key=get_chunk_number)
    sample_chunk_files.sort(key=get_chunk_number)
    
    print(f"Found {len(chunk_files)} result chunks and {len(sample_chunk_files)} sample chunks. Recombining...")
    
    # Load result chunks
    all_chunks_list = []
    for f_path in chunk_files:
        try:
            with open(f_path, "rb") as f:
                chunk_data = pickle.load(f)
                all_chunks_list.append(chunk_data)
        except Exception as e:
            print(f"Error loading {f_path}: {e}")
    
    # Load sample chunks
    all_sample_chunks_list = []
    for f_path in sample_chunk_files:
        try:
            with open(f_path, "rb") as f:
                sample_chunk_data = pickle.load(f)
                all_sample_chunks_list.append(sample_chunk_data)
        except Exception as e:
            print(f"Error loading {f_path}: {e}")
            
    if not all_chunks_list:
        print("No result data loaded.")
        return None

    # Concatenate all loaded chunks along the first axis (the samples axis)
    try:
        combined_results = np.concatenate(all_chunks_list, axis=0)
        combined_samples = np.concatenate(all_sample_chunks_list, axis=0) if all_sample_chunks_list else None
        return combined_results, combined_samples
    except ValueError as e:
        print(f"Error concatenating chunks. Are they all the same shape? {e}")
        print("Dumping individual chunk shapes:")
        for i, chunk in enumerate(all_chunks_list):
            print(f"  Result chunk {i} shape: {chunk.shape}")
        if all_sample_chunks_list:
            for i, chunk in enumerate(all_sample_chunks_list):
                print(f"  Sample chunk {i} shape: {chunk.shape}")
        return None

# Find the first season after the sixth season where the peak incidence rebounds to at least 70% of the average pre-pandemic peak, and measure the time of the peak of that season relative to 2020-03-19, i.e. the 170th day of the fifth season
def time_to_rebound(x, threshold_factor=0.5):
    """
    Find the time from the last pre-pandemic peak to the first post-pandemic peak.
    
    Args:
        x: Array of shape (2, n_seasons, n_age_groups+1) containing [obs_per_season, peak_times]
           where the last column contains age-summed data
        threshold_factor: Minimum fraction of pre-pandemic average for rebound detection
    
    Returns:
        Time in days from last pre-pandemic peak to first post-pandemic peak
    """
    # Use the age-summed data (last column)
    obs_summed = x[0, :, -1]  # Shape: (n_seasons,)
    peak_times_summed = x[1, :, -1]  # Shape: (n_seasons,)

    typical_pre_pandemic_season = jnp.median(obs_summed[:5])
    threshold = threshold_factor * typical_pre_pandemic_season

    # Find last pre-pandemic peak time (season 5, which is index 4)
    last_pre_pandemic_peak_time = peak_times_summed[4] + (4 * 365)
    
    # Find first season >= threshold in post-pandemic period
    post_pandemic_obs = obs_summed[5:]
    rebound_season_idx = jnp.argmax(post_pandemic_obs >= threshold)
    
    # Check if rebound was actually found
    rebound_found = post_pandemic_obs[rebound_season_idx] >= threshold
    
    # Calculate first post-pandemic peak time
    season_idx = 5 + rebound_season_idx
    first_post_pandemic_peak_time = peak_times_summed[season_idx] + (season_idx * 365)
    
    # Calculate time difference
    time_difference = first_post_pandemic_peak_time - last_pre_pandemic_peak_time
    
    return jnp.where(rebound_found, time_difference, jnp.nan)
@jax.jit
def vectorized_time_to_rebound(all_results):
    return jax.vmap(time_to_rebound)(all_results)

# find relative size of rebound season compared to pre-pandemic average
def relative_size_of_rebound(x):
    """
    Find the relative size of the rebound season compared to pre-pandemic average.
    
    Args:
        x: Array of shape (2, n_seasons, n_age_groups+1) containing [obs_per_season, peak_times]
           where the last column contains age-summed data
        sum_ages: If True, use age-summed data. If False, analyze individual age groups.
    
    Returns:
        Array of relative sizes (either per age group or summed)
    """
    # Use the age-summed data (last column)
    obs_summed = x[0, :, -1]  # Shape: (n_seasons,)
    pre_pandemic_median = jnp.median(obs_summed[:5])
    
    # Find largest post-pandemic season
    post_pandemic_max = jnp.max(obs_summed[5:])
    
    relative_size = post_pandemic_max / pre_pandemic_median
    return relative_size
    # else:
    #     # Per age group analysis (excluding the summed column)
    #     obs_per_age = x[0, :, :-1]  # Shape: (n_seasons, n_age_groups)
    #     pre_pandemic_maxs = jnp.max(obs_per_age[:6, :], axis=0)  # Shape: (n_age_groups,)
        
    #     # Find largest post-pandemic season for each age group
    #     post_pandemic_maxs = jnp.max(obs_per_age[6:, :], axis=0)  # Shape: (n_age_groups,)
        
    #     relative_sizes = post_pandemic_maxs / pre_pandemic_maxs
    #     return relative_sizes
@jax.jit
def vectorized_relative_size_of_rebound(all_results):
    return jax.vmap(relative_size_of_rebound)(all_results)

# Find the rato of cases in 1-4y to 3-11m in the rebound season compared to the median pre-pandemic season
def age_ratio_of_rebound(x):
    """
    Find the age ratio of the rebound season compared to pre-pandemic median.
    
    Args:
        x: Array of shape (2, n_seasons, n_age_groups+1) containing [obs_per_season, peak_times]
           where the last column contains age-summed data
    
    Returns:
        Age ratio in the rebound season compared to pre-pandemic median
    """
    # Use the obs_per_season data
    obs_per_season = x[0, :, :]  # Shape: (n_seasons, n_age_groups+1)
    
    # Define age group indices
    idx_3_11m = 1  # 3-11 months
    idx_1_4y = 2   # 1-4 years

    # Calculate pre-pandemic median ratios
    pre_pandemic_ratios = obs_per_season[:5, idx_1_4y] / obs_per_season[:5, idx_3_11m]
    pre_pandemic_median_ratio = jnp.median(pre_pandemic_ratios)

    # Find largest post-pandemic season
    post_pandemic_obs = obs_per_season[5:, :]
    rebound_season_idx = jnp.argmax(jnp.sum(post_pandemic_obs, axis=1))
    
    # Calculate age ratio in rebound season
    rebound_obs = post_pandemic_obs[rebound_season_idx]
    rebound_ratio = rebound_obs[idx_1_4y] / rebound_obs[idx_3_11m]
    
    age_ratio_of_rebound = rebound_ratio / pre_pandemic_median_ratio
    
    return age_ratio_of_rebound
@jax.jit
def vectorized_age_ratio_of_rebound(all_results):
    return jax.vmap(age_ratio_of_rebound)(all_results)

def age_of_first_infection(x):
    """
    Find the average age of first infection pre-pandemic.
    
    Args:
        x: Array of shape (3, n_seasons, n_age_groups+1) containing [obs_per_season, peak_times, proportion_first_infectious]
           where the last column contains age-summed data
    Returns:
        Average age of first infection pre-pandemic
    """
    proportion_first_infectious = x[2, :, :-1]  # Shape: (n_seasons, n_age_groups)
    pre_pandemic_ages = jnp.sum(proportion_first_infectious[:5, :] * jnp.array(MEDIAN_AGE), axis=1)  # Weight by median ages
    average_age = jnp.mean(pre_pandemic_ages)/12
    
    return average_age
@jax.jit
def vectorized_age_of_first_infection(all_results):
    return jax.vmap(age_of_first_infection)(all_results)

def age_of_infection(x):
    proportion_infections = x[3, :, :-1]  # Shape: (n_seasons, n_age_groups)
    pre_pandemic_ages = jnp.sum(proportion_infections[:5, :] * jnp.array(MEDIAN_AGE), axis=1)  # Weight by median ages
    average_age = jnp.mean(pre_pandemic_ages)/12
    
    return average_age
@jax.jit
def vectorized_age_of_infection(all_results):
    return jax.vmap(age_of_infection)(all_results)

def age_of_infector(x):
    infections_caused = x[4, :, :-1]  # Shape: (n_seasons, n_age_groups)
    pre_pandemic_ages = jnp.sum(infections_caused[:5, :] * jnp.array(MEDIAN_AGE), axis=1) / jnp.sum(infections_caused[:5, :], axis=1)  # Weight by median ages
    average_age = jnp.mean(pre_pandemic_ages)/12
    
    return average_age
@jax.jit
def vectorized_age_of_infector(all_results):
    return jax.vmap(age_of_infector)(all_results)

def foi_weighted_age(x):
    foi_experienced = x[5, :, :-1]  # Shape: (n_seasons, n_age_groups)
    pre_pandemic_ages = jnp.sum(foi_experienced[:5, :] * jnp.array(MEDIAN_AGE), axis=1)  / jnp.sum(foi_experienced[:5, :], axis=1)  # Weight by median ages
    average_age = jnp.mean(pre_pandemic_ages)/12
    
    return average_age
@jax.jit
def vectorized_foi_weighted_age(all_results):
    return jax.vmap(foi_weighted_age)(all_results)

def load_samples_and_results(run_save_path):
    # Load and recombine simulation results
    all_results, all_samples = load_and_recombine_results(run_save_path)
    if all_results is None:
        raise ValueError(f"Could not load results from {run_save_path}")
    return all_samples, all_results

def valid_samples_targets_array(samples, target_values, target_bounds=None):
    """
    Remove invalid samples and targets.
    """
    # Remove samples that have NaN
    valid_mask = ~jnp.isnan(target_values)
    valid_samples = samples[valid_mask]
    valid_targets = target_values[valid_mask]
    
    # Convert to numpy
    valid_samples = np.asarray(valid_samples)
    valid_targets = np.asarray(valid_targets)

    if target_bounds is not None:
        low, high = target_bounds
        bound_mask = (valid_targets >= low) & (valid_targets <= high)
        valid_samples = valid_samples[bound_mask]
        valid_targets = valid_targets[bound_mask]
        print(f"After applying target bounds {target_bounds}, {valid_samples.shape[0]} valid samples remain.")
        print(f"Target range after bounds: {valid_targets.min()} to {valid_targets.max()}")
    return valid_samples, valid_targets

# use Lasso regression to find parameters that predict time to rebound, then run lasso regression again on the errors of that model to create a 2d space
def lasso_analysis(run_save_path, outcome="time_to_rebound", alpha=0.01, target_bounds=None):
    """
    Perform Lasso analysis on simulation results.
    
    Args:
        run_save_path: Path to directory containing chunk files and samples
        analyze_size: If True, analyze relative size of rebound instead of time to rebound
        alpha: Regularization parameter for Lasso regression
    
    Returns:
        tuple: (lasso1_coef, lasso2_coef) - coefficients from both Lasso regressions
    """
    # Load samples
    samples, all_results = load_samples_and_results(run_save_path)

    # Calculate target variable (time to rebound or relative size) with age groups summed
    if outcome == "relative_size":
        target_values = vectorized_relative_size_of_rebound(all_results)
    elif outcome == "age_ratio":
        target_values = vectorized_age_ratio_of_rebound(all_results)
    elif outcome == "age_of_first_infection":
        target_values = vectorized_age_of_first_infection(all_results)
    elif outcome == "age_of_infection":
        target_values = vectorized_age_of_infection(all_results)
    elif outcome == "age_of_infector":
        target_values = vectorized_age_of_infector(all_results)
    elif "infectors_in_class_" in outcome:
        target_values = jax.vmap(lambda x: x[4, :5, int(outcome.split("_")[-1])].sum(axis=0)/x[4, :5, :-1].sum())(all_results)
    elif outcome == "foi_weighted_age":
        target_values = vectorized_foi_weighted_age(all_results)
    elif "abs_foi_in_class_" in outcome:
        target_values = jax.vmap(lambda x: x[5, :5, int(outcome.split("_")[-1])].sum(axis=0))(all_results)
    elif "foi_in_class_" in outcome:
        target_values = jax.vmap(lambda x: x[5, :5, int(outcome.split("_")[-1])].sum(axis=0)/x[5, :5, :-1].sum())(all_results)
    else:
        target_values = vectorized_time_to_rebound(all_results)
    print(np.min(target_values), np.max(target_values))
    
    # Convert to numpy for sklearn
    valid_samples, valid_targets = valid_samples_targets_array(samples, target_values, target_bounds=target_bounds)

    untransformed_targets = valid_targets.copy()

    valid_samples *= 1/np.array([1,1,1,1,1,1e-2,1e-2,-1,-1,-1,-1,1,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3])  # scale parameters for better lasso performance

    # First Lasso regression
    lasso1 = Lasso(alpha=alpha)
    lasso1.fit(valid_samples, valid_targets)
    predictions1 = lasso1.predict(valid_samples)
    residuals1 = valid_targets - predictions1
    
    # Second Lasso regression on residuals
    lasso2 = Lasso(alpha=alpha)
    lasso2.fit(valid_samples, residuals1)

    return lasso1.coef_, lasso2.coef_, valid_samples, untransformed_targets

# plot the time to rebound or relative size as a heatmap with contours on the 2d space defined by the two lasso components
def plot_lasso_heatmap(ax, valid_samples, valid_targets, lasso1_coef, lasso2_coef, outcome="time_to_rebound", use_scatter=False):
    """
    Plot heatmap using actual simulation data projected onto 2D Lasso space.
    
    Args:
        run_save_path: Path to directory containing chunk files and samples
        analyze_size: If True, analyze relative size of rebound instead of time to rebound
        alpha: Regularization parameter for Lasso regression
    """
    title = "Relative Size of Rebound Season" if outcome == "relative_size" else ("Age Ratio of Rebound Season" if outcome == "age_ratio" else "Time to Rebound (days)")

    # Project samples onto the 2D Lasso space using precomputed coefficients
    if lasso1_coef == "R0":
        lasso1_projections = 13.88 * valid_samples[:, 2]/valid_samples[:, 0]
    else:
        lasso1_projections = valid_samples @ lasso1_coef
    lasso2_projections = valid_samples @ lasso2_coef

    if all(lasso2_projections == 0):
        lasso2_projections = np.random.normal(0, 1e-6, size=lasso2_projections.shape)

    if outcome == "relative_size":
        vmin, vmax = 0.155, 1.82
    elif outcome == "time_to_rebound":
        vmin, vmax = 657, 1095
    else:
        vmin, vmax = None, None

    if use_scatter:
        # Scatter plot with tiny points
        hist = ax.scatter(lasso1_projections, lasso2_projections, c=valid_targets, 
                         s=1, alpha=1, cmap=cm.viridis, vmin=vmin, vmax=vmax)

    else:
        # Create a 2D histogram/heatmap instead of scatter plot
        # Define grid resolution
        grid_size = 100
        # Create bins for the 2D histogram
        x_range = [lasso1_projections.min(), lasso1_projections.max()]
        y_range = [lasso2_projections.min(), lasso2_projections.max()]

        # Compute the average target value in each bin
        ret = binned_statistic_2d(lasso1_projections, lasso2_projections, valid_targets, 
                                 statistic='mean', bins=grid_size, 
                                 range=[x_range, y_range])
        # Create the heatmap
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
        hist = ax.imshow(ret.statistic.T, origin='lower', extent=extent, 
                           cmap=cm.viridis, aspect='auto', interpolation='nearest',
                           vmin=vmin, vmax=vmax
                           )
    
    ax.set_xlabel('Lasso Component 1 Projection')
    ax.set_ylabel('Lasso Component 2 Projection')
    # ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return hist, ax


if __name__ == "__main__":
    seed = 251118
    np.random.seed(seed)
    p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
    # Define good simulations
    RSV = ["RSV", "251103", "NA"]
    Metapneumovirus = ["Metapneumovirus", "2511032", "NA"]
    InfluenzaA = ["InfluenzaA", "251103", "NA"]
    InfluenzaB = ["InfluenzaB", "2511042", "NA"]
    Adenovirus = ["Adenovirus", "2511032", "NA"]
    Parainfluenza3 = ["Parainfluenza3", "2511032", "NA"]

    short_names = {"RSV":"RSV", "Metapneumovirus":"hMPV", "InfluenzaA":"FluA", "InfluenzaB":"FluB", "Adenovirus":"AdV", "Parainfluenza3":"PIV3"}

    good_simulations = [RSV, Metapneumovirus, InfluenzaA, InfluenzaB, Adenovirus, Parainfluenza3]

    class FakeOpt:
        def __init__(self, x):
            self.x = x
    lockdown = "251110intermediate"

    parameter_sets = parameter_space(good_simulations)
    # for i, param_set in enumerate(parameter_sets):
    #     print(f"Parameters for {good_simulations[i][0]}:")
    #     x_to_params(param_set, "test", lockdown, "mimmwane", "flexage", print_params=True)

    # colors = ["#648FFF", "#DC267F", "#FFB000", "#785EF0", "#FF832B", "#000000"]
    # for i in range(parameter_sets.shape[0]):
    #     plt.plot(np.arange(7),parameter_sets[i,-7:], label=good_simulations[i][0], color=colors[i])
    # plt.xticks(np.arange(7), labels=[f"AGE_OBS_{i+1}" for i in range(7)])
    # plt.ylabel("Detection Probability")
    # plt.title("Parameter Sets from Good Simulations: Detection Probabilities by Age Group")
    # plt.legend()
    # plt.show()
    # # print third value of each parameter set
    # print(np.min(parameter_sets,axis=0), np.max(parameter_sets,axis=0))

    POINTS = np.array(date_to_t(PERIOD))
    N_S, NAG = 3, 7
    ## Initial conditions
    STATE0 = jnp.zeros((2*N_S+1,NAG))
    STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
    STATE0 = STATE0.at[1,:].set(1)
    # # flatten initial state and add maternal immunity compartment
    STATE0 = STATE0.flatten()
    STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

    n_samples = 80000
    dimension = 2
    samples = lh_sampling(parameter_sets, n_samples, dimension=dimension)

    # # plot first and sixth dimensions of parameter sets, labelled by pathogen
    # pathogen_colors = ["#648FFF", "#DC267F", "#FFB000", "#785EF0", "#FF832B", "#000000"]
    # plt.figure(figsize=(6,5))
    # for i in range(parameter_sets.shape[0]):
    #     plt.scatter(parameter_sets[i,0], parameter_sets[i,5], label=good_simulations[i][0], color=pathogen_colors[i],
    #                 edgecolor='black', linewidth=1, s=50)
    #     plt.annotate(good_simulations[i][0], 
    #                 (parameter_sets[i,0], parameter_sets[i,5]), 
    #                 xytext=(5, 5), textcoords='offset points', 
    #                 fontsize=8, ha='left')
    # plt.xlabel("Transmissibility")
    # plt.ylabel("Immunity from first infection")
    # plt.savefig("Figures/parameter_sets_transmissibility_immunity.png", dpi=300)
    # plt.close()

    # plt.figure(figsize=(6,5))
    # plt.scatter(samples[:,0], samples[:,5], color='gray', alpha=0.1, s=1)
    # for i in range(parameter_sets.shape[0]):
    #     plt.scatter(parameter_sets[i,0], parameter_sets[i,5], label=good_simulations[i][0], 
    #                edgecolor='black', linewidth=1, s=50, color=pathogen_colors[i])
    #     plt.annotate(good_simulations[i][0], 
    #                 (parameter_sets[i,0], parameter_sets[i,5]), 
    #                 xytext=(5, 5), textcoords='offset points', 
    #                 fontsize=8, ha='left')
    
    # # Plot convex hull
    # points = parameter_sets[:, [0, 5]]  # Extract transmissibility and immunity columns
    # hull = ConvexHull(points)
    # for simplex in hull.simplices:
    #     plt.plot(points[simplex, 0], points[simplex, 1], 'k-', alpha=0.5, linewidth=1)
    
    # plt.xlabel("Transmissibility")
    # plt.ylabel("Immunity from first infection")
    # plt.savefig("Figures/parameter_sets_and_samples_transmissibility_immunity.png", dpi=300)
    # plt.close()



    print(np.min(samples,axis=0)-np.min(parameter_sets,axis=0), np.max(parameter_sets,axis=0)-np.max(samples,axis=0))
    CHUNK_SIZE = 40000
    run_save_path = f"Outputs/sim_grid_lh_n{n_samples}_chunk{CHUNK_SIZE}_seed{seed}_lockdown{lockdown}_{dimension}d"
    # save samples
    os.makedirs(run_save_path, exist_ok=True)
    with open(os.path.join(run_save_path, "samples.pickle"), "wb") as f:
        pickle.dump(np.asarray(samples), f)

    start_time = time.time()  
    # Call the new chunked simulation function
    saved_files = simulate_samples_chunked(
        samples, lockdown, POINTS, STATE0, p_time_to_obs,
        base_save_path=run_save_path,
        chunk_size=CHUNK_SIZE,
        skip_existing=True,
    )
    end_time = time.time()
    print(f"Simulations for n_samples={n_samples} completed in {end_time - start_time} seconds.")
    print(f"Results saved in directory: {run_save_path}")

    # # # Example of plotting
    # run_save_path = f"Outputs/sim_grid_lh_n80000_chunk40000_seed251117_2d"
    outcome = "NA"
    print(f"Performing Lasso analysis for outcome: {outcome}")
    lasso1_coef, lasso2_coef, valid_samples, valid_targets = lasso_analysis(run_save_path, outcome=outcome, alpha=0.001, target_bounds=None)
    # print("Limits of valid targets: ", valid_targets.min(), valid_targets.max())
    # print(lasso1_coef)
    # print(lasso2_coef)

    parameter_names = ["Basic reproduction number", "First infection duration", "Second infection duration", "Transmissibility", "Seasonality", "Phase", "Waning (per 100 days after first infection)", "Waning (per 100 days after second infection)",
                        "Immunity from first infection", "Immunity from second infection", "Immunity to severe disease after first infection","Immunity to severe disease after second infection",
                        "Maternal immunity","Disease susceptibility <3m", "Disease susceptibility 3–11m", "Disease susceptibility 1–4y",
                        "Disease susceptibility 5–7y", "Disease susceptibility 8–49y", "Disease susceptibility 50-64y", "Disease susceptibility 65+y"]
    short_pnames = ["R0", "rec_up", "rec_same", "transmissibility", "seasonality", "phase", "wane1", "wane2", "immunity1", "immunity2", "disease_immunity1", "disease_immunity2",
                    "maternal_immunity", "susceptibility_0_3m", "susceptibility_3_11m", "susceptibility_1_4y",
                    "susceptibility_5_7y", "susceptibility_8_49y", "susceptibility_50_64y", "susceptibility_65y"]

    # # random forest variable importance analysis on valid samples and targets
    # from sklearn.ensemble import RandomForestRegressor
    # # add a column for R0 = transmissibility / seasonality
    # valid_samples = np.hstack(((valid_samples[:,2]/valid_samples[:,0]).reshape(-1,1),valid_samples))
    # rf = RandomForestRegressor(n_estimators=100, random_state=42)
    # rf.fit(valid_samples, valid_targets)
    # importances = rf.feature_importances_
    # indices = np.argsort(importances)[::-1]
    # print("Random Forest Feature Importances:")
    # for f in range(valid_samples.shape[1]):
    #     print(f"{f + 1}. {parameter_names[indices[f]]}: {importances[indices[f]]:.4f}")
    # # plot feature importances
    # fig, ax = plt.subplots(figsize=(10,6))
    # ax.bar(range(valid_samples.shape[1]), importances[indices], align='center')
    # ax.set_xticks(range(valid_samples.shape[1]))
    # ax.set_xticklabels([short_pnames[i] for i in indices], rotation=45, ha='right')
    # ax.set_ylabel("Importance")
    # ax.set_title("Random Forest Feature Importances for "+("Relative Size of Rebound Season" if outcome=="relative_size" else "Time to Rebound"))
    # plt.tight_layout()
    # plt.savefig(f"Figures/random_forest_feature_importances_{outcome}.png", dpi=300)
    # plt.close()

    # for p1 in range(len(lasso1_coef)+1):
    #     for p2 in range(p1+1,len(lasso2_coef)+1):
    p1, p2 = 0, 8
    if p1 == 0:
        lasso1_coef = "R0"
    else:
        lasso1_coef = jnp.zeros(19)
        lasso1_coef = lasso1_coef.at[p1-1].set(1)
    lasso2_coef = jnp.zeros(19)
    lasso2_coef = lasso2_coef.at[p2-1].set(1)

    # # find valid samples with lasso 1 projection
    # if lasso1_coef == "R0":
    #     lasso1_proj = valid_samples[:, 2] / valid_samples[:, 0]
    # else:
    #     lasso1_proj = valid_samples @ lasso1_coef
    # print(lasso1_proj.min(), lasso1_proj.max())
    # valid_samples_lasso1 = valid_samples[np.abs(lasso1_proj - (0.668-0.2/15)) < 2*0.2/15]
    # lasso2_proj = valid_samples_lasso1 @ lasso2_coef
    # print(lasso2_proj.min(), lasso2_proj.max())
    # valid_samples_lasso2 = valid_samples_lasso1[np.abs(lasso2_proj - (0.315+0.2/24)) < 2*0.2/24]
    # print("Samples around target: ", valid_samples_lasso2.shape[0])
    # # Create a 3x3 subplot for the first 9 valid samples
    # fig, axes = plt.subplots(10, 10, figsize=(13.3, 7.5))
    # axes = axes.flatten()  # Flatten for easier indexing
    
    # for i in range(min(100, len(valid_samples_lasso2))):
    #     sample = valid_samples_lasso2[i]
    #     sim_params = x_to_params(sample * np.array([1,1,1,1,1,1e-2,1e-2,-1,-1,-1,-1,1,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3]), "sim", lockdown, "mimmwane", "flexage", print_params=False)
    #     ax = axes[i]
    #     mx = lockdown_incidence_plot(ax, STATE0, sim_params, POINTS, date_to_t('2020-03-19'), by_age = True, AGE_GROUP_NAMES=AGE_GROUP_NAMES)
    #     ax.set_title(f"Sample {i+1}")
    #     print(f"Plotted sample {i+1}/9")
    
    # # Hide any unused subplots
    # for i in range(len(valid_samples_lasso2), 9):
    #     axes[i].set_visible(False)
    
    # plt.tight_layout()
    # plt.savefig(f"Figures/selected_simulations_yellow_near_PIV3_3x3_grid.png", dpi=1000)
    # plt.close()  # Close the figure to free memory

    plt.rcParams.update({'font.size':18})
    # text type is palatino
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Palatino']
    fig, ax = plt.subplots(figsize=(12,5))
    # scatter, ax = plot_lasso_heatmap(ax, valid_samples, valid_targets, lasso1_coef, lasso2_coef, outcome=outcome, use_scatter=False)
    # cbar = plt.colorbar(scatter, ax=ax)
    # ## Make scatter points invisible by setting alpha to 0
    # # scatter.set_alpha(0)
    # cbar.set_label('Time to re-emergence (years)' if outcome=="time_to_rebound" else ("Relative size of rebound" if outcome=="relative_size" else ("Age ratio of rebound" if outcome=="age_ratio" else "Age of infection (years)")))
    # if outcome == "foi_weighted_age":
    #     cbar.set_label("Age weighted by FOI experienced (years)")
    # elif outcome == "age_of_infector":
    #     cbar.set_label("Average age of infectors (years)")
    # elif "infectors_in_class_" in outcome:
    #     class_idx = int(outcome.split("_")[-1])
    #     cbar.set_label(f"Proportion of infectors in ages {AGE_GROUP_NAMES[class_idx]}")
    # elif "abs_foi_in_class_" in outcome:
    #     class_idx = int(outcome.split("_")[-1])
    #     cbar.set_label(f"Absolute FOI in ages {AGE_GROUP_NAMES[class_idx]}")
    # elif "foi_in_class_" in outcome:
    #     class_idx = int(outcome.split("_")[-1])
    #     cbar.set_label(f"Relative FOI in ages {AGE_GROUP_NAMES[class_idx]}")
    # # # Convert colorbar ticks from days to years
    # if outcome == "time_to_rebound":
    #     ticks = cbar.get_ticks()
    #     cbar.set_ticks(ticks)
    #     cbar.set_ticklabels([f'{tick/365:.1f}' for tick in ticks])
    # if "foi_in_class_0" in outcome:
    #     ticks = cbar.get_ticks()
    #     cbar.set_ticks(ticks)
    #     cbar.set_ticklabels([f'{tick/0.00016:.2f}' for tick in ticks])
    # plot points from good simulations parameter sets, projected onto lasso space
    # textcolors = ["white", "black", "black", "black", "white", "white"]
    # edgecolors = ["white", "black", "black", "black", "white", "black"]
    textcolors = edgecolors = ["black"]*6
    pathogen_proj = np.zeros((len(good_simulations), 2))
    for i, pathogen_info in enumerate(good_simulations):
        pathogen, seed, option1 = pathogen_info
        if outcome in ["time_to_rebound", "relative_size", "age_ratio"]:
            incidence = jnp.asarray(pd.read_csv("Data/Processed/KPSC_ARI_"+pathogen+"_cases_age_daily.csv",index_col=0))
            obs_summed_age = incidence.sum(axis=1)
            # sum obs over each season, starting with the first time point
            n_seasons = int((POINTS[-1] - POINTS[0]) / 365)
            # Curtail obs to fit exact seasons (ignore partial days due to leap years)
            days_to_keep = n_seasons * 365
            obs_curtailed = incidence[:days_to_keep, :]
            obs_summed_age_curtailed = obs_summed_age[:days_to_keep]
            obs_per_season = obs_curtailed.reshape((n_seasons, 365, NAG)).sum(axis=1)
            obs_summed_age_per_season = obs_summed_age_curtailed.reshape((n_seasons, 365)).sum(axis=1)
            # concatenate to obs_per_season
            obs_per_season = jnp.concatenate([obs_per_season, obs_summed_age_per_season[:, None]], axis=1)
            # print(obs_per_season[:,-1])
            # find the time of peak incidence in each age group for each season
            peak_times = jnp.argmax(obs_curtailed.reshape((n_seasons, -1, NAG)), axis=1)
            peak_times_summed_age = jnp.argmax(obs_summed_age_curtailed.reshape((n_seasons, -1)), axis=1)
            peak_times = jnp.concatenate([peak_times, peak_times_summed_age[:, None]], axis=1)
            seasons = jnp.stack([obs_per_season, peak_times], axis=0)
            # print(f"{pathogen} time to rebound: {time_to_rebound_value} days, relative size: {rebound_size_value}")
        # map to color using same scheme as scatter plot
        if outcome == "relative_size":
            rebound_size_value = relative_size_of_rebound(seasons)
            pathogen_color = cm.viridis(( rebound_size_value - 0.155) / (1.82 - 0.155))
        elif outcome == "age_ratio":
            age_ratio_value = age_ratio_of_rebound(seasons)
            pathogen_color = cm.viridis(( age_ratio_value - np.min(valid_targets)) / (np.max(valid_targets) - np.min(valid_targets)))
        elif outcome == "time_to_rebound":
            time_to_rebound_value = time_to_rebound(seasons)
            pathogen_color = cm.viridis(( time_to_rebound_value - 657) / (1095 - 657))
        else:
            pathogen_color = "white"            

        x = consistent_x_from_DE(pathogen, option1, seed)/np.array([1,1,1,1,1,1e-2,1e-2,-1,-1,-1,-1,1,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3])  # scale parameters for better lasso performance
        
        if lasso1_coef == "R0":
            lasso1_proj = 13.88 * x[2]/x[0]
        else:
            lasso1_proj = x @ lasso1_coef
        lasso2_proj = x @ lasso2_coef
        pathogen_proj[i, :] = jnp.array([lasso1_proj, lasso2_proj])
        print(f"{pathogen} projections: {lasso1_proj}, {lasso2_proj}")
        ax.scatter(lasso1_proj, lasso2_proj, color=pathogen_color, s=50, edgecolor=edgecolors[i])
        ax.annotate(short_names[pathogen], (lasso1_proj, lasso2_proj), xytext=(5, 5), 
            textcoords='offset points', fontsize=18, ha='left', color=textcolors[i])
    # plot line of best fit with CIs through the good simulation points
    from sklearn.linear_model import LinearRegression
    from scipy import stats
    model = LinearRegression()
    model.fit(np.log(pathogen_proj[:,0]).reshape(-1,1), pathogen_proj[:,1])
    print(model.coef_, model.intercept_)
    x_fit = np.linspace(1, 44.4, 1000)
    y_fit = model.predict(np.log(x_fit).reshape(-1,1))
    
    # Filter out points where y > 0 or y < -1
    valid_mask = (y_fit >= -1) & (y_fit <= 0)
    x_fit = x_fit[valid_mask]
    y_fit = y_fit[valid_mask]
    
    # Calculate 95% confidence intervals
    n = len(pathogen_proj)
    x_mean = np.mean(np.log(pathogen_proj[:,0]))
    residuals = pathogen_proj[:,1] - model.predict(np.log(pathogen_proj[:,0]).reshape(-1,1))
    mse = np.sum(residuals**2) / (n - 2)
    se = np.sqrt(mse * (1/n + (np.log(x_fit) - x_mean)**2 / np.sum((np.log(pathogen_proj[:,0]) - x_mean)**2)))
    t_val = stats.t.ppf(0.975, n-2)  # 95% CI
    ci_upper = np.minimum(y_fit + t_val * se, 0)
    ci_lower = np.maximum(y_fit - t_val * se, -1)
    
    ax.plot(x_fit, y_fit, color='black', linestyle='--', label ='Line of best fit')
    ax.fill_between(x_fit, ci_lower, ci_upper, alpha=0.3, color='gray', label='95% confidence interval')

    # extra pathogen data
    extra_names = ["Rotavirus", "Norovirus", "Measles"]
    extra_R0s = [17.5, 2, 13.2]
    extra_R0s_upper = [18.2, 7.2, 44.4]
    extra_R0s_lower = [5.03, 1.1, 4.6]
    extra_immunity1 = [-0.62, -0.74, 0]
    extra_immunity1_upper = [-0.83, -0.95, 0]
    extra_immunity1_lower = [-0.5, -0.57, 0]
    # plot these points with error bars
    for i, name in enumerate(extra_names):
        ax.scatter(extra_R0s[i], extra_immunity1[i], s=50, color='grey', edgecolor='black')
        ax.errorbar(extra_R0s[i], extra_immunity1[i], 
                    xerr=[[extra_R0s[i]-extra_R0s_lower[i]], [extra_R0s_upper[i]-extra_R0s[i]]],
                    yerr=[[extra_immunity1[i]-extra_immunity1_upper[i]], [extra_immunity1_lower[i]-extra_immunity1[i]]],
                    fmt='o', color='grey', ecolor='black', elinewidth=1, capsize=6)
        ax.annotate(name, (extra_R0s[i], extra_immunity1[i]), xytext=(5, 5), 
            textcoords='offset points', fontsize=18, ha='left', color='black')

    ax.set_xlim([0.901,45])
    ax.set_ylim([-1.1,0.1])
    ax.set_xscale('log')
    ax.set_xticks([1,2,5,10,20,40])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Basic reproduction number (log scale)")
    ax.legend()
    
    # ax.set_title("Fit Parameter Sets")
    # put label on x-axis saying 1e-2
    # ax.set_xlabel(parameter_names[p1])
    ax.set_ylabel(parameter_names[p2])
    # Get current y-ticks and set new labels as 1 - original value
    if "Immunity" in parameter_names[p1]:
        y_ticks = ax.get_yticks()
        ax.set_yticklabels([f'{1.01+tick:.1f}' for tick in y_ticks])
    if "Immunity" in parameter_names[p2]:
        y_ticks = ax.get_yticks()
        ax.set_yticklabels([f'{1.01+tick:.1f}' for tick in y_ticks])
    if "duration" in parameter_names[p1]:
        y_ticks = ax.get_yticks()
        ax.set_yticklabels([f'{1/tick:.1f}' for tick in y_ticks])
    if "duration" in parameter_names[p2]:
        y_ticks = ax.get_yticks()
        ax.set_yticklabels([f'{1/tick:.1f}' for tick in y_ticks])

    plt.tight_layout()
    plt.savefig(f"Figures/parameter_sets_{short_pnames[p1]}_{short_pnames[p2]}_{outcome}_2d_extras.png", dpi=1000)