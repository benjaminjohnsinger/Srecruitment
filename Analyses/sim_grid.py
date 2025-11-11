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
from utils import consistent_x_from_DE, x_to_params, date_to_t
from fit_MCMC import run_simulation
from Parameters.census_population import CENSUS_AGE_POP
from Parameters.times_and_contacts import PERIOD
import multiprocessing
from scipy.stats import qmc

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
def worker(args):
    sample, lockdown, POINTS, STATE0 = args
    params = x_to_params(sample, "test", lockdown, "mimmwane", "flexage")
    sim_result = run_simulation(params, STATE0, int(POINTS[-1]), POINTS)
    return sim_result

# for each sample, simulate the model and save the output
def simulate_samples(samples, lockdown, POINTS, STATE0, n_workers=4):
    # Prepare arguments for each worker
    args_list = [(sample, lockdown, POINTS, STATE0) for sample in samples]
    with multiprocessing.Pool(n_workers) as pool:
        all_results = pool.map(worker, args_list)
    return all_results


if __name__ == "__main__":
    seed = 251110
    np.random.seed(seed)
    # Define good simulations
    RSV = ["RSV", "251103", "NA"]
    Metapneumovirus = ["Metapneumovirus", "2511032", "NA"]
    InfluenzaA = ["InfluenzaA", "251103", "NA"]
    InfluenzaB = ["InfluenzaB", "2511042", "NA"]
    Adenovirus = ["Adenovirus", "2511032", "NA"]
    Parainfluenza3 = ["Parainfluenza3", "2511032", "NA"]

    good_simulations = [RSV, Metapneumovirus, InfluenzaA, InfluenzaB, Adenovirus, Parainfluenza3]

    parameter_sets = parameter_space(good_simulations)
    n_samples = 8**6
    samples = lh_sampling(parameter_sets, n_samples)

    lockdown = "maxmimmwaneflexage2511032"
    print(re.match(r'\d{6}',lockdown))
    POINTS = np.array(date_to_t(PERIOD))
    N_S, NAG = 3, 7
    ## Initial conditions
    STATE0 = jnp.zeros((2*N_S+1,NAG))
    STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
    STATE0 = STATE0.at[1,:].set(1)
    # # flatten initial state and add maternal immunity compartment
    STATE0 = STATE0.flatten()
    STATE0 = jnp.concatenate((jnp.array([0]), STATE0))
    start_time = time.time()
    all_results = simulate_samples(samples, lockdown, POINTS, STATE0, n_workers=8)
    end_time = time.time()
    print(f"Simulations completed in {end_time - start_time} seconds.")
    # how many results?
    print(f"Number of simulation results: {len(all_results)}")

    # Save results
    with open("Data/Processed/simulation_grid_results" + str(seed) + ".pickle", "wb") as f:
        pickle.dump(all_results, f)