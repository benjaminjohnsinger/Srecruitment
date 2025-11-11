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
from Parameters.census_population import CENSUS_AGE_POP
from Parameters.times_and_contacts import PERIOD
from scipy.stats import qmc
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from matplotlib import cm

# Load opt.x for each pathogen and find a polygon that captures the parameter space
def parameter_space(good_simulations):
    parameter_sets = []
    for pathogen_info in good_simulations:
        pathogen, seed, option1 = pathogen_info
        x = consistent_x_from_DE(pathogen, option1, seed)
        parameter_sets.append(x)
    parameter_sets = jnp.array(parameter_sets)
    return(parameter_sets)


def lh_sampling(parameter_sets, n_samples):
    n_pathogens = parameter_sets.shape[0]
    
    # Use Latin Hypercube Sampling for better space-filling properties
    
    # Create Latin Hypercube sampler
    sampler = qmc.LatinHypercube(d=n_pathogens-1)
    
    # Generate samples in simplex coordinates
    lhs_samples = sampler.random(n=n_samples)
    
    # Convert to Dirichlet-like weights (ensure they sum to 1)
    # Transform uniform samples to exponential, then normalize
    exp_samples = -jnp.log(1 - lhs_samples + 1e-10)  # Add small epsilon to avoid log(0)
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
    params = x_to_params(sample, "test", lockdown, "mimmwane", "flexage")
    solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS)
    # find total observed infections each season in each age group
    values = solution.ys.T
    trajectory = jnp.diff(values[-NAG:,:], axis=1).T
    p_time_to_obs_flipped = jnp.flip(p_time_to_obs.flatten())
    @jax.jit
    def obs_convolution(x):
        return jnp.convolve(x, p_time_to_obs_flipped, mode='same')
    obs = jax.vmap(obs_convolution, in_axes=1, out_axes=1)(trajectory)
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
    # return as a 3d array
    return jnp.stack([obs_per_season, peak_times], axis=0)

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
            
        all_chunk_paths.append(chunk_path)
        print(f"  Saved {chunk_path}")
    print(f"All chunks saved to {base_save_path}")
    return all_chunk_paths

def load_and_recombine_results(run_save_path):
    """
    Loads all pickled chunks from a specific run directory,
    recombines them, and returns a single NumPy array.
    """
    
    # Find all chunk files in the directory
    search_path = os.path.join(run_save_path, "chunk_*.pickle")
    chunk_files = glob.glob(search_path)
    
    if not chunk_files:
        print(f"No chunk files found in: {run_save_path}")
        return None
        
    # --- Sort the files numerically ---
    def get_chunk_number(filepath):
        # Extracts the number '0001' from '.../chunk_0001.pickle'
        filename = os.path.basename(filepath)
        match = re.search(r'chunk_(\d+)\.pickle', filename)
        if match:
            return int(match.group(1))
        return -1 # Should not happen if files are named correctly

    chunk_files.sort(key=get_chunk_number)
    
    print(f"Found {len(chunk_files)} chunks. Recombining...")
    
    all_chunks_list = []
    for f_path in chunk_files:
        try:
            with open(f_path, "rb") as f:
                chunk_data = pickle.load(f)
                all_chunks_list.append(chunk_data)
        except Exception as e:
            print(f"Error loading {f_path}: {e}")
            
    if not all_chunks_list:
        print("No data loaded.")
        return None

    # Concatenate all loaded chunks along the first axis (the samples axis)
    try:
        combined_results = np.concatenate(all_chunks_list, axis=0)
        return combined_results
    except ValueError as e:
        print(f"Error concatenating chunks. Are they all the same shape? {e}")
        print("Dumping individual chunk shapes:")
        for i, chunk in enumerate(all_chunks_list):
            print(f"  Chunk {i} shape: {chunk.shape}")
        return None

# Find the first season after the sixth season where the peak incidence rebounds to at least 50% of the average pre-pandemic peak, and measure the time of the peak of that season relative to 2020-03-19, i.e. the 170th day of the fifth season
def time_to_rebound(x, threshold_factor=0.3):
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
    
    pre_pandemic_peak = jnp.mean(obs_summed[:6])
    threshold = threshold_factor * pre_pandemic_peak
    
    # Find last pre-pandemic peak time (season 5, which is index 4)
    last_pre_pandemic_peak_time = peak_times_summed[4] + (4 * 365)
    
    # Find first season >= threshold in post-pandemic period
    post_pandemic_obs = obs_summed[5:]
    rebound_season_idx = jnp.argmax(post_pandemic_obs >= threshold)
    
    # Check if rebound was actually found
    rebound_found = post_pandemic_obs[rebound_season_idx] >= threshold
    
    # Calculate first post-pandemic peak time
    season_idx = 6 + rebound_season_idx
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
    pre_pandemic_max = jnp.max(obs_summed[:6])
    
    # Find largest post-pandemic season
    post_pandemic_max = jnp.max(obs_summed[6:])
    
    relative_size = post_pandemic_max / pre_pandemic_max
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

def load_samples_and_results(run_save_path):
    # Load samples
    samples_path = os.path.join(run_save_path, "samples.pickle")
    with open(samples_path, "rb") as f:
        samples = pickle.load(f)
    
    # Load and recombine simulation results
    all_results = load_and_recombine_results(run_save_path)
    if all_results is None:
        raise ValueError(f"Could not load results from {run_save_path}")
    
    # if simulation hasn't finished running, warn and trim samples
    if all_results.shape[0] < samples.shape[0]:
        print(f"Warning: Only {all_results.shape[0]} results found, but {samples.shape[0]} samples exist. Trimming samples.")
        samples = samples[:all_results.shape[0]]

    return samples, all_results

def valid_samples_targets_array(samples,target_values):
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
    return valid_samples, valid_targets

# use Lasso regression to find parameters that predict time to rebound, then run lasso regression again on the errors of that model to create a 2d space
def lasso_analysis(run_save_path, analyze_size=False, alpha=0.01):
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
    if analyze_size:
        target_values = vectorized_relative_size_of_rebound(all_results)
    else:
        target_values = vectorized_time_to_rebound(all_results)
    
    # Convert to numpy for sklearn
    valid_samples, valid_targets = valid_samples_targets_array(samples, target_values)

    # First Lasso regression
    lasso1 = Lasso(alpha=alpha)
    lasso1.fit(valid_samples, valid_targets)
    predictions1 = lasso1.predict(valid_samples)
    residuals1 = valid_targets - predictions1
    
    # Second Lasso regression on residuals
    lasso2 = Lasso(alpha=alpha)
    lasso2.fit(valid_samples, residuals1)
    
    return lasso1.coef_, lasso2.coef_, valid_samples, valid_targets

# plot the time to rebound or relative size as a heatmap with contours on the 2d space defined by the two lasso components
def plot_lasso_heatmap(ax, valid_samples, valid_targets, lasso1_coef, lasso2_coef, analyze_size=False):
    """
    Plot heatmap using actual simulation data projected onto 2D Lasso space.
    
    Args:
        run_save_path: Path to directory containing chunk files and samples
        analyze_size: If True, analyze relative size of rebound instead of time to rebound
        alpha: Regularization parameter for Lasso regression
    """
    title = "Relative Size of Rebound Season" if analyze_size else "Time to Rebound (days)"

    # Project samples onto the 2D Lasso space using precomputed coefficients
    lasso1_projections = valid_samples @ lasso1_coef
    lasso2_projections = valid_samples @ lasso2_coef

    if all(lasso2_projections == 0):
        lasso2_projections = np.random.normal(0, 1e-6, size=lasso2_projections.shape)

    # Create scatter plot of actual data points
    scatter = ax.scatter(lasso1_projections, lasso2_projections, c=valid_targets, 
                            cmap=cm.viridis, alpha=1, s=0.01)
                            
    ax.set_xlabel('Lasso Component 1 Projection')
    ax.set_ylabel('Lasso Component 2 Projection')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return scatter, ax


if __name__ == "__main__":
    seed = 251111
    np.random.seed(seed)
    p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
    # Define good simulations
    RSV = ["RSV", "251103", "NA"]
    Metapneumovirus = ["Metapneumovirus", "2511032", "NA"]
    InfluenzaA = ["InfluenzaA", "251103", "NA"]
    InfluenzaB = ["InfluenzaB", "2511042", "NA"]
    Adenovirus = ["Adenovirus", "2511032", "NA"]
    Parainfluenza3 = ["Parainfluenza3", "2511032", "NA"]

    good_simulations = [RSV, Metapneumovirus, InfluenzaA, InfluenzaB, Adenovirus, Parainfluenza3]

    lockdown = "maxmimmwaneflexage2511032"

    parameter_sets = parameter_space(good_simulations)
    # # print third value of each parameter set
    # print(np.min(parameter_sets,axis=0), np.max(parameter_sets,axis=0))

    POINTS = np.array(date_to_t(PERIOD))
    # N_S, NAG = 3, 7
    # ## Initial conditions
    # STATE0 = jnp.zeros((2*N_S+1,NAG))
    # STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
    # STATE0 = STATE0.at[1,:].set(1)
    # # # flatten initial state and add maternal immunity compartment
    # STATE0 = STATE0.flatten()
    # STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

    # n_samples = 6**8
    # samples = lh_sampling(parameter_sets, n_samples)
    # print(np.min(samples,axis=0)-np.min(parameter_sets,axis=0), np.max(parameter_sets,axis=0)-np.max(samples,axis=0))
    # CHUNK_SIZE = 40000
    # run_save_path = f"Outputs/sim_grid_lh_n{n_samples}_chunk{CHUNK_SIZE}_seed{seed}"
    # # save samples
    # os.makedirs(run_save_path, exist_ok=True)
    # with open(os.path.join(run_save_path, "samples.pickle"), "wb") as f:
    #     pickle.dump(np.asarray(samples), f)

    # start_time = time.time()  
    # # Call the new chunked simulation function
    # saved_files = simulate_samples_chunked(
    #     samples, lockdown, POINTS, STATE0, p_time_to_obs,
    #     base_save_path=run_save_path,
    #     chunk_size=CHUNK_SIZE,
    #     skip_existing=True,
    # )
    # end_time = time.time()
    # print(f"Simulations for n_samples={n_samples} completed in {end_time - start_time} seconds.")
    # print(f"Results saved in directory: {run_save_path}")

    # Example of plotting
    fig, ax = plt.subplots(figsize=(8,6))
    run_save_path = f"Outputs/sim_grid_lh_n1679616_chunk40000_seed251111"
    lasso1_coef, lasso2_coef, valid_samples, valid_targets = lasso_analysis(run_save_path, analyze_size=False, alpha=0.0019)
    print(lasso1_coef)
    print(lasso2_coef)
    scatter, ax = plot_lasso_heatmap(ax, valid_samples, valid_targets, lasso1_coef, lasso2_coef, analyze_size=False)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time to Rebound (days)')
    # plot points from good simulations parameter sets, projected onto lasso space
    colors = ["#648FFF", "#DC267F", "#FFB000", "#785EF0", "#FF832B", "#000000"]
    for i, pathogen_info in enumerate(good_simulations):
        pathogen, seed, option1 = pathogen_info
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
        print(obs_per_season[:,-1])
        # find the time of peak incidence in each age group for each season
        peak_times = jnp.argmax(obs_curtailed.reshape((n_seasons, -1, NAG)), axis=1)
        peak_times_summed_age = jnp.argmax(obs_summed_age_curtailed.reshape((n_seasons, -1)), axis=1)
        peak_times = jnp.concatenate([peak_times, peak_times_summed_age[:, None]], axis=1)
        seasons = jnp.stack([obs_per_season, peak_times], axis=0)
        time_to_rebound_value = time_to_rebound(seasons, threshold_factor=0.3)
        print(f"{pathogen} time to rebound: {time_to_rebound_value} days")
        # map to color using same scheme as scatter plot
        pathogen_color = cm.viridis((time_to_rebound_value-np.min(valid_targets))/(np.max(valid_targets)-np.min(valid_targets)))
        x = consistent_x_from_DE(pathogen, option1, seed)
        lasso1_proj = x @ lasso1_coef
        lasso2_proj = x @ lasso2_coef
        ax.scatter(lasso1_proj, lasso2_proj, color=pathogen_color, s=50, edgecolor='black')
        ax.annotate(pathogen, (lasso1_proj, lasso2_proj), xytext=(5, 5), 
               textcoords='offset points', fontsize=8, ha='left')

    plt.savefig("Figures/lasso_time_to_rebound_heatmap.png", dpi=1000)