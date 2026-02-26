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

pathogen, seed, lockdown, option1, option2, import_multiplier, opt_size, opt_rate1, opt_rate2 = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9])

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

N = jnp.prod(jnp.asarray(daily_hospitalization_rates.shape))

pp_opt = None

def likelihood(x):
    sim_params = x_to_params(x, pathogen, lockdown, option1, option2, rescale=bounds)
    if "pp" in option2:
        pp_opt = sim_params[8]
    lh = -SIS_likelihood(tests, daily_hospitalization_rates, sim_params, POINTS, STATE0, p_time_to_obs, obs_age=pp_opt)
    lh = lh / N # normalize by number of data points
    return lh


if __name__ == '__main__':
    # #scipy version of DE
    vmap_likelihood = jax.jit(jax.vmap(likelihood))
    # def scipy_objective(x):
    #     x_transposed = x.T
    #     return np.asarray(vmap_likelihood(x_transposed))
    # multiprocessing.set_start_method('spawn', force=True)
    # opt = sp.optimize.differential_evolution(scipy_objective,bounds,popsize=opt_size,mutation=(0.5,opt_rate1),recombination=opt_rate2,init="halton",seed=seed,updating="deferred",
    # strategy="currenttobest1bin", vectorized=True)
    # # if there's no Data/Processed/results<seed> directory, create it
    # if not os.path.exists("Data/Processed/results"+str(seed)[:6]):
    #     os.makedirs("Data/Processed/results"+str(seed)[:6])
    # with open("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","wb") as f:
    #     pickle.dump(opt,f)

    # optax minimizer
    import optax

    hypercube_size = int(opt_rate2)

    # generate latin hypercube starting points within bounds
    sampling_start_time = time.time()
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    xs = jnp.array(sampler.random(n=hypercube_size))
    # xs = jnp.array(bounds[:, 0] + lhs_samples * (bounds[:, 1] - bounds[:, 0]))
    print(f"Generated {hypercube_size} Latin hypercube samples in {time.time() - sampling_start_time:.2f} seconds.")

    # if there are opt_states with likelihood over 100, resample those points
    likelihoods = vmap_likelihood(xs)
    likelihood_threshold = 2 # constrain to reasonable initial guesses
    print(f"n initial points with likelihood > {likelihood_threshold}: {jnp.sum(likelihoods > likelihood_threshold)}")
    max_resampling_iterations = 100
    for iteration in range(max_resampling_iterations):
        bad_indices = jnp.where(likelihoods > likelihood_threshold)[0]
        if len(bad_indices) == 0:
            break
        
        # Resample all bad points at once
        n_bad = len(bad_indices)
        new_samples = sampler.random(n=n_bad)
        # scaled_new_samples = jnp.array(bounds[:, 0] + new_samples * (bounds[:, 1] - bounds[:, 0]))
        
        # Replace bad samples
        xs = xs.at[bad_indices].set(new_samples)
        new_likelihoods = vmap_likelihood(new_samples)
        likelihoods = likelihoods.at[bad_indices].set(new_likelihoods)
        print(f"Iteration {iteration + 1}: {jnp.sum(likelihoods > likelihood_threshold)} points still exceed threshold")
    print(f"Resampled points with likelihood > {likelihood_threshold} in {time.time() - sampling_start_time:.2f} seconds.")

    # save initial points to disk
    if not os.path.exists("Data/Processed/results"+str(seed)[:6]):
        os.makedirs("Data/Processed/results"+str(seed)[:6])
    with open("Data/Processed/results"+str(seed)[:6]+"/optax_initial_points_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","wb") as f:
        pickle.dump(xs,f)

    schedule = optax.exponential_decay(init_value=opt_rate1, transition_steps=jnp.max(1000, opt_size/5), decay_rate=0.5, staircase=True)
    solver = optax.apply_if_finite(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adabelief(learning_rate=schedule)
        ),
        max_consecutive_errors=5,
    )

    def single_step(x, opt_state):
        neglogL, grad = jax.value_and_grad(likelihood)(x)
        update, opt_state = solver.update(grad, opt_state, x)
        x = optax.apply_updates(x, update)
        x = optax.projections.projection_box(x, 0, 1)
        return x, opt_state, neglogL
    vmapped_step = jax.jit(jax.vmap(single_step))

    def scan_body(carry, step_index):
        x, opt_state = carry
        x, opt_state, neglogL = vmapped_step(x, opt_state)
        return (x, opt_state), neglogL
    
    @jax.jit
    def run_optimization(xs):
        vmapped_init = jax.jit(jax.vmap(solver.init))
        opt_states = vmapped_init(xs)

        initial_carry = (xs, opt_states)
        final_carry, neglogL_history = jax.lax.scan(scan_body, initial_carry, jnp.arange(opt_size))
        final_xs, _ = final_carry
        return final_xs, neglogL_history
    
    final_xs, neglogL_history = run_optimization(xs)
    # save results to disk
    results_file = "Data/Processed/results"+str(seed)[:6]+"/optax_"+pathogen+lockdown+option1+option2+str(seed)+".pickle"
    with open(results_file, "wb") as f:
        pickle.dump({"final_xs": final_xs, "neglogL_history": neglogL_history}, f)
    # print best parameters and likelihood
    best_index = jnp.argmin(neglogL_history[-1])
    best_params = bounds[:,0] + final_xs[best_index] * (bounds[:,1] - bounds[:,0])
    best_likelihood = jnp.min(neglogL_history[-1])
    print(f"Best parameters: {best_params}")
    print(f"Best likelihood: {best_likelihood}")


    # vmapped_init = jax.jit(jax.vmap(solver.init))
    # opt_states = vmapped_init(xs)

    # record_likelihoods = jnp.zeros((opt_size, hypercube_size))
    # record_xs = jnp.zeros((opt_size, hypercube_size, len(bounds)))
    # sampling_start_time = time.time()
    
    # # if there's no Data/Processed/results<seed> directory, create it
    # if not os.path.exists("Data/Processed/results"+str(seed)[:6]):
    #     os.makedirs("Data/Processed/results"+str(seed)[:6])
    
    # results_file = "Data/Processed/results"+str(seed)[:6]+"/optax_"+pathogen+lockdown+option1+option2+str(seed)+".pickle"
    
    # for i in range(opt_size):
    #     xs, opt_states, neglogLs = vmapped_step(xs, opt_states)
    #     record_likelihoods = record_likelihoods.at[i].set(neglogLs)
    #     record_xs = record_xs.at[i].set(xs)
        
    #     # Write to disk after each iteration
    #     with open(results_file, "wb") as f:
    #         pickle.dump({"opt_states": opt_states, "record_likelihoods": record_likelihoods, "record_xs": record_xs, "iteration": i+1}, f)