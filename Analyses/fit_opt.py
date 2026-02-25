## BJS Jan 2025
## Fitting models to data using out-of-the-box optimisation tools

import jax.numpy as jnp
import numpy as np
import scipy as sp
import pandas as pd
import time
import pickle
import sys
import os
import multiprocessing

from Parameters.census_population import *
from Parameters.times_and_contacts import *

from utils import *
from fit_MCMC import SIS_likelihood
import traceback

pathogen, seed, lockdown, option1, option2, import_multiplier, desize, max_mutation, recombination = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9])

# set seed
np.random.seed(seed)

start_date = '2015-07-04'
end_date = '2025-05-01'
# check if option1 is in date format with regex
if re.match(r'\d{4}-\d{2}-\d{2}',option1):
    start_date = option1
if re.match(r'\d{4}-\d{2}-\d{2}',option2):
    end_date = option2
    option2 = "flexage" #this is super hacky sorry

START = pd.to_datetime(start_date) 
END = pd.to_datetime(end_date)
PERIOD = pd.date_range(start=START, end=END, freq='D')
POINTS = np.array(date_to_t(PERIOD))
FULL_PERIOD = pd.date_range(start=pd.to_datetime('1970-01-01'), end=END, freq='D')
FULL_POINTS = np.array(date_to_t(FULL_PERIOD))

REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, tests_full = pathogen_parameters(pathogen, import_multiplier=import_multiplier)
N_S, NAG = 3, 7

# # trim incidence so that Date is between START and END
start_idx = int(date_to_t(start_date) + 90 - date_to_t('2015-10-01'))
end_idx = int(date_to_t(end_date) - date_to_t('2015-10-01'))
tests = tests_full[start_idx:end_idx, :, :]

daily_hospitalization_rates_pd = pd.read_csv('Data/Processed/KPSC_ARI_hospitalization_rates_by_day_age_group.csv',index_col=0,parse_dates=True)
daily_hospitalization_rates_pd = daily_hospitalization_rates_pd.fillna(0)
daily_hospitalization_rates_full = jnp.asarray(daily_hospitalization_rates_pd.values)
daily_hospitalization_rates = daily_hospitalization_rates_full[start_idx:end_idx,]

## Initial conditions
STATE0 = jnp.zeros((2*N_S+1,NAG))
STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
STATE0 = STATE0.at[1,:].set(1)
# # flatten initial state and add maternal immunity compartment
STATE0 = STATE0.flatten()
STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

param_names, bounds = parameters_names_bounds(pathogen, lockdown, option1, option2)

def likelihood(x):
    sim_params = x_to_params(x, pathogen, lockdown, option1, option2)
    lh = -SIS_likelihood(tests, daily_hospitalization_rates, sim_params, POINTS, STATE0, p_time_to_obs)
    lh = lh / jnp.prod(jnp.asarray(daily_hospitalization_rates.shape)) # normalize by number of data points
    is_invalid = jnp.isnan(lh) | jnp.isinf(lh)
    return jnp.where(is_invalid, 1e10, lh)

vmap_likelihood = jax.jit(jax.vmap(likelihood))

if __name__ == '__main__':
    # #scipy version
    # def scipy_objective(x):
    #     x_transposed = x.T
    #     return np.asarray(vmap_likelihood(x_transposed))
    # multiprocessing.set_start_method('spawn', force=True)
    # opt = sp.optimize.differential_evolution(scipy_objective,bounds,popsize=desize,mutation=(0.5,max_mutation),recombination=recombination,init="halton",seed=seed,updating="deferred",
    # strategy="currenttobest1bin", vectorized=True)
    # # if there's no Data/Processed/results<seed> directory, create it
    # if not os.path.exists("Data/Processed/results"+str(seed)[:6]):
    #     os.makedirs("Data/Processed/results"+str(seed)[:6])
    # with open("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","wb") as f:
    #     pickle.dump(opt,f)

    # evosax version
    from evosax.algorithms import CMA_ES
    
    # set up random key and bounds for evosax
    rng = jax.random.PRNGKey(seed)
    lower_bounds = jnp.array([b[0] for b in bounds])
    upper_bounds = jnp.array([b[1] for b in bounds])
    nD = len(bounds)
    # n_generations = 1000
    n_generations = 40

    dummy_solution = jnp.array([(lower_bounds[i] + upper_bounds[i]) / 2 for i in range(nD)])

    # initialize CMA-ES optimizer
    strategy = CMA_ES(population_size=desize, num_dims=nD)
    # pass the bounds into the optimizer parameters
    es_params = strategy.default_params
    # es_params = es_params.replace(
    #     bounds_min=lower_bounds,
    #     bounds_max=upper_bounds,
    # )

    # create the optimization step
    @jax.jit
    def step(state, rng_key):
        # get population of candidate parameters
        x, state = strategy.ask(rng_key, state, es_params)
        # evaluate the likelihood for each candidate parameter set
        fitness = vmap_likelihood(x)
        # update the optimizer state with the fitness values
        state = strategy.tell(x, fitness, state, es_params)
        return state, x, fitness
    
    # initialize the optimizer state
    rng, init_rng = jax.random.split(rng)
    state = strategy.init(init_rng, es_params)

    # run the optimization loop
    print(f"Starting optimization for {pathogen} with seed {seed}...")
    if not os.path.exists("Data/Processed/results"+str(seed)[:6]):
        os.makedirs("Data/Processed/results"+str(seed)[:6])

    best_x = None
    best_fitness = jnp.inf

    for gen in range(n_generations):
        rng, step_rng = jax.random.split(rng)

        state, population, fitness = step(state, step_rng)

        # track the best solution found so far
        if state.best_fitness < best_fitness:
            best_fitness = state.best_fitness
            best_x = state.best_x

    # save results
    with open("Data/Processed/results"+str(seed)[:6]+"/evosax_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","wb") as f:
        pickle.dump({"x": best_x, "fun": best_fitness},f)