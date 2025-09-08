import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import time

# --- Data Loading and Constants (Assumed to be defined elsewhere in your code) ---

# This setup is the same as your original code
VAX_FLU = pd.read_csv('Data/Processed/KPSC_vaccinated_proportion_ages_monthly.csv', index_col=0)
VAX_FLU = VAX_FLU[['<3m', '3-11m', '1-4y', '5-17y', '18-39y', '40-64y', '>=65y']]
VAX_FLU.index = pd.to_datetime(VAX_FLU.index, format='%Y-%m')
VAX_FLU.index = (VAX_FLU.index - pd.to_datetime('1970-01-01')).days
KPSC_RATIO = 0.455 / 0.5793469238813651
VAX_FLU_ADJUSTED = VAX_FLU * KPSC_RATIO
VAX_FLU_IDX = jnp.array(VAX_FLU_ADJUSTED.index)
VAX_FLU_NP = jnp.array(VAX_FLU_ADJUSTED)

EFF = (1/100) * jnp.array([37, 61, 51, 44, 39, 53, 7, 52, 19, 33, 25, 34, 37, 32, 23, 41, 37])
EFF_IDX = jnp.array([(pd.to_datetime('2008-10-01') + pd.DateOffset(years=i) - pd.to_datetime('1970-01-01')).days for i in range(17)])

# --- Step 1: Vectorized helper function for vaccine coverage ---

@partial(jax.jit, static_argnames=['eff_cap'])
def flu_eff_coverage_vectorized(t_arr, max_eff, eff_cap=True):
    """
    Calculates effective flu vaccine coverage for an array of time points.
    """
    # Vectorized lookup for raw efficacy using searchsorted
    # Find the index of the season for each time point in t_arr
    # eff_indices = jnp.searchsorted(EFF_IDX, t_arr, side='right')
    # # Gemini's original version has a -1 to get the correct index
    eff_indices = jnp.searchsorted(EFF_IDX, t_arr, side='right') - 1
    raw_eff = EFF[jnp.maximum(0, eff_indices)] # Use maximum to handle t before first season

    adj_eff = raw_eff / max_eff
    if eff_cap:
        adj_eff = jnp.minimum(adj_eff, 1.0)

    # Define time boundaries
    time_2015_start = 16709  # Days from 1970-01-01 to 2015-10-01
    time_2022_end = 19266    # Days from 1970-01-01 to 2022-10-01

    # --- Logic for times before the main data range ---
    day_in_season_pre = (t_arr - time_2015_start) % 365
    time_in_2015_season = time_2015_start + day_in_season_pre
    vax_indices_pre = jnp.searchsorted(VAX_FLU_IDX, time_in_2015_season)
    coverage_pre = adj_eff[:, None] * VAX_FLU_NP[vax_indices_pre]

    # --- Logic for times after the main data range ---
    day_in_season_post = (t_arr - time_2022_end) % 365
    time_in_2022_season = time_2022_end + day_in_season_post
    vax_indices_post = jnp.searchsorted(VAX_FLU_IDX, time_in_2022_season)
    coverage_post = adj_eff[:, None] * VAX_FLU_NP[vax_indices_post]

    # --- Logic for times within the main data range ---
    vax_indices_mid = jnp.searchsorted(VAX_FLU_IDX, t_arr)
    coverage_mid = adj_eff[:, None] * VAX_FLU_NP[vax_indices_mid]

    # Combine results using jnp.where for conditional logic
    coverage = jnp.where(t_arr[:, None] < time_2015_start, coverage_pre, coverage_mid)
    coverage = jnp.where(t_arr[:, None] > time_2022_end, coverage_post, coverage)

    return coverage

# --- Step 2: Pre-calculation Class ---

class FluRatePreprocessor:
    """
    A class to perform all calculations that do not depend on the 'protection' parameter.
    Initialize this once before your optimization loop.
    """
    def __init__(self, full_points, age_pops, aging_rate):
        self.full_points = full_points
        self.age_pops = age_pops
        
        # Pre-calculate population ratio (shape: n_times, n_ages)
        pop_ratio_calc = age_pops[:, :-1] / age_pops[:, 1:]
        self.pop_ratio = jnp.pad(pop_ratio_calc, ((0, 0), (0, 1)))

        # Pre-calculate shifted aging rate (shape: n_ages)
        self.age_rate_shift = jnp.pad(aging_rate[:-1], (1, 0))

# --- Step 3: Main JIT-compiled Vectorized Function ---

@partial(jax.jit, static_argnames=['preprocessor','eff_cap'])
def calculate_vax_rate_vectorized(protection, preprocessor, eff_cap=True):
    """
    Calculates the vaccination rate for all time points at once.
    
    Args:
        protection (jnp.array): The protection array (e.g., S_REL * P_OBS). Changes frequently.
        preprocessor (FluRatePreprocessor): The pre-calculated data object.
        eff_cap (bool): Whether to cap vaccine efficacy at 1.
    """
    # This calculation is fast and depends on the changing 'protection' parameter
    max_eff = (protection[-2] - protection[-1]) / protection[-2]

    # Calculate coverage for all time points and for the following month
    v = flu_eff_coverage_vectorized(preprocessor.full_points, max_eff, eff_cap)
    v_next_month = flu_eff_coverage_vectorized(preprocessor.full_points + 31, max_eff, eff_cap)
    
    # Calculate shifted vaccine coverage (v_shift)
    v_shift_calc = v[:, 1:] - v[:, :-1]
    v_shift = jnp.pad(v_shift_calc, ((0, 0), (0, 1)))

    # Calculate the rate using vectorized operations
    # This replaces the body of your original flu_rate function
    rate_numerator = (v_next_month - v) / 31 + (1 / 365.0) * v + \
                     preprocessor.age_rate_shift * preprocessor.pop_ratio * v_shift
    
    rate_denominator = 1 - v

    # Use jnp.where to handle the v >= 1 condition safely without a loop
    # This prevents division by zero and sets the rate to 0 where coverage is >= 1
    safe_rate = jnp.where(
        v >= 1,
        0,
        rate_numerator / rate_denominator
    )
    
    return jnp.maximum(0, safe_rate)

# --- Usage Example ---
if __name__ == "__main__":

    # --- Placeholder variables for demonstration ---
    # Replace these with your actual data
    NAG = 7
    FULL_POINTS = jnp.arange(0, 19266) # Example time points
    age_pops = jnp.ones((19266, NAG)) # Example age populations
    AGING_RATE = jnp.ones(NAG) / (365 * 10) # Example aging rate
    P_OBS = jnp.array([1,0.5,0.2])
    S_REL = jnp.array([1,0.5,0.2])

    # Example usage of the optimized functions
    # 1. Initialize the preprocessor ONCE outside your main loop
    print("Performing one-time pre-calculation...")
    preprocessor = FluRatePreprocessor(FULL_POINTS, age_pops, AGING_RATE)
    print("Pre-calculation complete.")

    protection_param = S_REL * P_OBS 
    VAX_RATE = calculate_vax_rate_vectorized(protection_param, preprocessor)

    S_REL = jnp.array([1, 0.51, 0.21])
    P_OBS = jnp.array([1, 0.51, 0.21])
    # 2. Inside your loop, call the fast, vectorized function
    # This is the line you would use repeatedly with different S_REL * P_OBS values
    print("\nCalculating VAX_RATE with the optimized function...")
    time_start = time.time()
    protection_param = S_REL * P_OBS
    for i in range(10):
        protection_param = protection_param*np.random.uniform(0.9,1.1,size=protection_param.shape)
        VAX_RATE = calculate_vax_rate_vectorized(protection_param, preprocessor)
    print("Calculation complete, took", (time.time() - time_start)/1000, "seconds.")
    print("Shape of VAX_RATE:", VAX_RATE.shape)

    # # save VAX_RATE
    # np.savetxt('Data/Processed/KPSC_vaccination_rate_ages_monthly_optimized.csv', VAX_RATE, delimiter=',')