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
from Parameters.census_population import CENSUS_AGE_POP, AGE_GROUP_NAMES
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

# Find the first season after the sixth season where the peak incidence rebounds to at least 70% of the average pre-pandemic peak, and measure the time of the peak of that season relative to 2020-03-19, i.e. the 170th day of the fifth season
def time_to_rebound(x, threshold_factor=0.4):
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
def lasso_analysis(run_save_path, analyze_size=False, alpha=0.01, target_bounds=None):
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
    valid_samples, valid_targets = valid_samples_targets_array(samples, target_values, target_bounds=target_bounds)

    untransformed_targets = valid_targets.copy()

    valid_samples *= 1/np.array([1,1,1,1e-2,1e-2,1,1,1,1,1,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3])  # scale parameters for better lasso performance

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
def plot_lasso_heatmap(ax, valid_samples, valid_targets, lasso1_coef, lasso2_coef, analyze_size=False, use_scatter=False):
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

    if analyze_size:
        vmin, vmax = 0.155, 1.82
    else:
        vmin, vmax = 1004, 1850

    if use_scatter:
        # Scatter plot with tiny points
        hist = ax.scatter(lasso1_projections, lasso2_projections, c=valid_targets, 
                         s=1, alpha=1, cmap=cm.viridis, vmin=vmin, vmax=vmax)

    else:
        # Create a 2D histogram/heatmap instead of scatter plot
        # Define grid resolution
        grid_size = 50
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
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return hist, ax


if __name__ == "__main__":
    seed = 251113
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

    lockdown = "maxmimmwaneflexage2511032"

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

    n_samples = 15000
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



    # print(np.min(samples,axis=0)-np.min(parameter_sets,axis=0), np.max(parameter_sets,axis=0)-np.max(samples,axis=0))
    CHUNK_SIZE = 40000
    run_save_path = f"Outputs/sim_grid_lh_n{n_samples}_chunk{CHUNK_SIZE}_seed{seed}_{dimension}d"
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
    run_save_path = f"Outputs/sim_grid_lh_n15000_chunk40000_seed251113_2d"
    lasso1_coef, lasso2_coef, valid_samples, valid_targets = lasso_analysis(run_save_path, analyze_size=False, alpha=0.001, target_bounds=(1004, 1850))
    print("Limits of valid targets: ", valid_targets.min(), valid_targets.max())
    print(lasso1_coef)
    print(lasso2_coef)
    lasso1_coef = jnp.zeros(lasso1_coef.shape)
    lasso1_coef = lasso1_coef.at[0].set(1)
    lasso2_coef = jnp.zeros(lasso2_coef.shape)
    lasso2_coef = lasso2_coef.at[5].set(1)

    # # find valid samples with lasso 1 projection
    # lasso1_proj = valid_samples @ lasso1_coef
    # print(lasso1_proj.min(), lasso1_proj.max())
    # valid_samples_lasso1 = valid_samples[np.abs(lasso1_proj - 4.1) < 5e-3]
    # lasso2_proj = valid_samples_lasso1 @ lasso2_coef
    # valid_samples_lasso2 = valid_samples_lasso1[np.abs(lasso2_proj + 6.2e-6) < 5e-9]
    # print("Samples around target: ", valid_samples_lasso2.shape[0])
    # # Create a 3x3 subplot for the first 9 valid samples
    # fig, axes = plt.subplots(3, 3, figsize=(13.3, 7.5))
    # axes = axes.flatten()  # Flatten for easier indexing
    
    # for i in range(min(9, len(valid_samples_lasso2))):
    #     sample = valid_samples_lasso2[i]
    #     sim_params = x_to_params(sample * np.array([1,1,1,1e-2,1e-2,1,1,1,1,1,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3]), "test", lockdown, "mimmwane", "flexage", print_params=False)
    #     ax = axes[i]
    #     mx = lockdown_incidence_plot(ax, STATE0, sim_params, POINTS, date_to_t('2020-03-19'), by_age = True, AGE_GROUP_NAMES=AGE_GROUP_NAMES)
    #     ax.set_title(f"Sample {i+1}")
    #     print(f"Plotted sample {i+1}/9")
    
    # # Hide any unused subplots
    # for i in range(len(valid_samples_lasso2), 9):
    #     axes[i].set_visible(False)
    
    # plt.tight_layout()
    # plt.savefig(f"Figures/selected_simulations_alpha0_hMPV_3x3_grid.png", dpi=1000)
    # plt.close()  # Close the figure to free memory

    fig, ax = plt.subplots(figsize=(8,6))
    scatter, ax = plot_lasso_heatmap(ax, valid_samples, valid_targets, lasso1_coef, lasso2_coef, analyze_size=False, use_scatter=True)
    cbar = plt.colorbar(scatter, ax=ax)
    ## Make scatter points invisible by setting alpha to 0
    # scatter.set_alpha(0)
    cbar.set_label('Time to Rebound (years)')
    # Convert colorbar ticks from days to years
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{tick/365:.1f}' for tick in ticks])
    # plot points from good simulations parameter sets, projected onto lasso space
    # textcolors = ["white", "black", "black", "black", "white", "white"]
    # edgecolors = ["white", "black", "black", "black", "white", "black"]
    textcolors = edgecolors = ["black"]*6
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
        time_to_rebound_value = time_to_rebound(seasons)
        rebound_size_value = relative_size_of_rebound(seasons)
        print(f"{pathogen} time to rebound: {time_to_rebound_value} days, relative size: {rebound_size_value}")
        # map to color using same scheme as scatter plot
        pathogen_color = cm.viridis(( time_to_rebound_value - 1004) / (1850 - 1004))
        x = consistent_x_from_DE(pathogen, option1, seed)/np.array([1,1,1,1e-2,1e-2,1,1,1,1,1,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3])  # scale parameters for better lasso performance
        lasso1_proj = x @ lasso1_coef
        lasso2_proj = x @ lasso2_coef
        ax.scatter(lasso1_proj, lasso2_proj, color=pathogen_color, s=50, edgecolor=edgecolors[i])
        ax.annotate(short_names[pathogen], (lasso1_proj, lasso2_proj), xytext=(5, 5), 
               textcoords='offset points', fontsize=12, ha='left', color=textcolors[i])
    ax.set_title("Fit Parameter Sets")
    # put label on x-axis saying 1e-2
    ax.set_xlabel('Transmissibility')
    ax.set_ylabel('Immunity from first infection')
    # Get current y-ticks and set new labels as 1 - original value
    y_ticks = ax.get_yticks()
    ax.set_yticklabels([f'{1-tick:.1f}' for tick in y_ticks])
    plt.savefig("Figures/parameter_sets_transmissibility_immunity_time_to_rebound_2d.png", dpi=1000)