## BJS Jan 2025
## Fitting models to data using out-of-the-box optimisation tools
# set jax to use 64 bit precision
import jax
# jax.config.update("jax_enable_x64", True)
print(f"Devices found: {jax.devices()}")
import jax.numpy as jnp
import numpy as np
import pandas as pd
import time
import pickle
import sys
import os
print(f"My PID is: {os.getpid()}")
if "Cuda" in str(jax.devices()):
    print(f"Physical GPU assigned by Slurm: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

from Parameters.census_population import *
from Parameters.times_and_contacts import *

from utils import *
from fit_MCMC import SIS_likelihood, peaks_and_times_likelihood
from plotting import calculate_observations_per_season, get_season_start
from data_processing import calculate_proportion_positive_incidence

# import scipy as sp
# import multiprocessing

def get_likelihood(pathogen, lockdown, option1, option2, import_multiplier, normalize=True, hosp=True, hessian=False):
    ### load data and parameters
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

    N_S, NAG = 3, 7
    from Parameters.census_population import AGING_RATE
    CONTACT_MATRIX = jnp.asarray(pd.read_csv('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', header=None).values)
    BIRTH_RATE = jnp.asarray(np.genfromtxt('Data/Processed/birth_rate_daily.csv', delimiter=','))
    if "incidence_data" in option1:
        if "smoothed" in option1:
            REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=import_multiplier, incidence_data=True, smoothed=True, hosp=hosp)
        else:
            REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=import_multiplier, incidence_data=True, smoothed=False, hosp=hosp)
    else:
        REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=import_multiplier, incidence_data=False, hosp=hosp)
    fixed_params = (FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX, REC_UP, REC_SAME, IMPORT_STRENGTH)

    # # trim incidence so that Date is between START and END
    start_idx = int(date_to_t(start_date) + 90 - date_to_t('2015-10-01'))
    end_idx = int(date_to_t(end_date) - date_to_t('2015-10-01'))
    data = data_full[start_idx:end_idx]

    if "incidence_data" not in option1 and "peaks_and_times" not in option1:
        daily_hospitalization_rates_pd = pd.read_csv('Data/Processed/KPSC_ARI_hospitalization_rates_by_day_age_group.csv',index_col=0,parse_dates=True)
        daily_hospitalization_rates_pd = daily_hospitalization_rates_pd.fillna(0)
        daily_hospitalization_rates_full = jnp.asarray(daily_hospitalization_rates_pd.values)
        daily_hospitalization_rates = daily_hospitalization_rates_full[start_idx:end_idx,]
    if ("peaks_and_times" in option1) or ("combo" in option1):
        incidence = calculate_proportion_positive_incidence(pathogen, aggregation="D", window_size=1, weighting_factor=0, sum_age_groups=False, save_counts=False, pp_only=False, hosp=hosp)
        incidence = incidence.fillna(0)
        obs_per_season = jnp.asarray(calculate_observations_per_season(incidence, age_groups=True))
        peak_times = incidence.groupby(incidence.index.map(get_season_start)).idxmax()
        peak_times = jnp.asarray(np.array([date_to_t(t) for t in peak_times.values]))

    ## Initial conditions
    STATE0 = jnp.zeros((2*N_S+1,NAG))
    STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
    STATE0 = STATE0.at[1,:].set(1)
    # # flatten initial state and add maternal immunity compartment
    STATE0 = STATE0.flatten()
    STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

    #### Define likelihood function for optimization
    if "incidence_data" in option1:
        N = jnp.prod(jnp.asarray(data.shape))
        def likelihood(x):
            sim_params = x_to_params(x, pathogen, lockdown, option1, option2, fixed_params=fixed_params
                                    #  , rescale=bounds
                                    )
            lh = -SIS_likelihood(data, 0, sim_params, POINTS, STATE0, p_time_to_obs, incidence_data=True, hessian=hessian)
            if normalize:
                lh = lh / N # normalize by number of data points
            return lh
    elif "peaks_and_times" in option1:
        N = 20 # number of data points is number of peaks + number of peak times
        def likelihood(x):
            sim_params = x_to_params(x, pathogen, lockdown, option1, option2, fixed_params=fixed_params
                                    #  , rescale=bounds
                                    )
            lh = -peaks_and_times_likelihood(obs_per_season, peak_times, sim_params, POINTS, STATE0, p_time_to_obs, hessian=hessian)
            if normalize:
                lh = lh / N # normalize by number of data points
            return lh
    elif "combo" in option1:
        N1 = jnp.prod(jnp.asarray(data.shape)) # number of data points is number of peaks + number of peak times + number of incidence data points
        N2 = 20
        N = (N1 + N2) / 2 # average number of data points to make comparable to other likelihoods
        def likelihood(x):
            sim_params = x_to_params(x, pathogen, lockdown, option1, option2, fixed_params=fixed_params
                                    #  , rescale=bounds
                                    )
            lh_base = -SIS_likelihood(data, daily_hospitalization_rates, sim_params, POINTS, STATE0, p_time_to_obs, hessian=hessian)
            lh_peaks_times = -peaks_and_times_likelihood(obs_per_season, peak_times, sim_params, POINTS, STATE0, p_time_to_obs, hessian=hessian)
            if normalize:
                lh = (lh_base / N1 + lh_peaks_times / N2)/2 # normalize by number of data points to make comparable
            else:
                lh = (lh_base + lh_peaks_times)/2
            return lh
    else:
        N = jnp.prod(jnp.asarray(daily_hospitalization_rates.shape))
        def likelihood(x, pp_opt=None):
            sim_params = x_to_params(x, pathogen, lockdown, option1, option2, fixed_params=fixed_params
                                    #  , rescale=bounds
                                    )
            if "pp" in option2:
                pp_opt = sim_params[8]
            lh = -SIS_likelihood(data, daily_hospitalization_rates, sim_params, POINTS, STATE0, p_time_to_obs, obs_age=pp_opt, hessian=hessian)
            if normalize:
                lh = lh / N # normalize by number of data points
            return lh
    return likelihood, N

#### Define functions for sampling initial points and resampling bad points
def latin_hypercube_sample(key, n_samples, n_dims):
    cut = jnp.linspace(0, 1, n_samples + 1)
    lower = cut[:-1]
    upper = cut[1:]
    def sample_dim(subkey):
        binned_samples = jax.random.uniform(subkey, (n_samples,)) * (upper - lower) + lower
        return jax.random.permutation(subkey, binned_samples)
    keys = jax.random.split(key, n_dims)
    samples = jax.vmap(sample_dim)(keys)
    return samples.T

likelihood_threshold = 10.0
max_resampling_iterations = 100
def apply_condition(state):
    _, likelihoods, _, iteration = state
    is_bad = (likelihoods > likelihood_threshold) | jnp.isnan(likelihoods)
    return jnp.any(is_bad) & (iteration < max_resampling_iterations)

def resample_bad_points(state):
    xs, likelihoods, key, iteration = state
    key, subkey = jax.random.split(key)

    is_bad = (likelihoods > likelihood_threshold) | jnp.isnan(likelihoods)

    uniform_samples = latin_hypercube_sample(subkey, xs.shape[0], xs.shape[1])
    new_xs = bounds[:, 0] + uniform_samples * (bounds[:, 1] - bounds[:, 0])

    new_likelihoods = vmap_likelihood(new_xs)

    updated_xs = jnp.where(is_bad[:, None], new_xs, xs)
    updated_likelihoods = jnp.where(is_bad, new_likelihoods, likelihoods)
    return updated_xs, updated_likelihoods, key, iteration + 1

@jax.jit
def run_resampling(xs, likelihoods, key):
    initial_state = (xs, likelihoods, key, 0)
    final_state = jax.lax.while_loop(apply_condition, resample_bad_points, initial_state)
    return final_state[0], final_state[1], final_state[3]

if __name__ == '__main__':
    pathogen, seed, lockdown, option1, option2, import_multiplier, opt_size, opt_rate1, opt_rate2 = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9])

    if len(sys.argv) > 10:
        algorithm = sys.argv[10]
        if len(sys.argv) > 11:
            optax_length = int(sys.argv[11])
    else:
        algorithm = "evosax" # default to evosax if not specified

    # set seed
    np.random.seed(seed)

    # # #scipy version of DE
    # vmap_likelihood = jax.jit(jax.vmap(likelihood))
    # def scipy_objective(x):
    #     x_transposed = x.T
    #     return jnp.asarray(vmap_likelihood(x_transposed))
    # multiprocessing.set_start_method('spawn', force=True)
    # opt = sp.optimize.differential_evolution(scipy_objective,bounds,popsize=opt_size,mutation=(0.5,opt_rate1),recombination=opt_rate2,init="halton",seed=seed,updating="deferred",
    # strategy="currenttobest1bin", vectorized=True)
    # # if there's no Data/Processed/results<seed> directory, create it
    # if not os.path.exists("Data/Processed/results"+str(seed)[:6]):
    #     os.makedirs("Data/Processed/results"+str(seed)[:6])
    # with open("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","wb") as f:
    #     pickle.dump(opt,f)


    _, bounds = parameters_names_bounds(pathogen, lockdown, option1, option2)
    # bounds = jnp.zeros(unlogged_bounds.shape)
    # bounds = bounds.at[:, 1].set(10)
    # bounds = bounds.at[:, 0].set(-10)
    likelihood, _ = get_likelihood(pathogen, lockdown, option1, option2, import_multiplier)
    def new_likelihood(x):
        lik =  likelihood(x)
        # remove nans
        lik = jnp.where(jnp.isnan(lik), likelihood_threshold*10, lik)
        return lik
    vmap_likelihood = jax.vmap(new_likelihood)

    ## starting population for optax or DE
    key = jax.random.PRNGKey(seed)

    hypercube_size = int(opt_size) * len(bounds)

    key, subkey = jax.random.split(key)
    sampling_start_time = time.time()
    lhs_samples = latin_hypercube_sample(subkey, hypercube_size, len(bounds))
    xs = jnp.array(bounds[:, 0] + lhs_samples * (bounds[:, 1] - bounds[:, 0]))
    print(f"Generated {hypercube_size} Latin hypercube samples with JAX in {time.time() - sampling_start_time:.2f} seconds.")

    # if there are opt_states with likelihood over 100, resample those points
    likelihoods = jax.jit(vmap_likelihood)(xs)

    if not ("skip_resampling" in algorithm):
        key, subkey = jax.random.split(key)
        print(f"Starting resampling of {jnp.sum((likelihoods > likelihood_threshold) | jnp.isnan(likelihoods))} bad initial points...")
        start_time = time.time()
        xs, likelihoods, iterations = run_resampling(xs, likelihoods, subkey)
        likelihoods.block_until_ready()
        print(f"Resampling completed in {time.time() - start_time:.2f} seconds after {iterations} iterations.")

    if "evosax" in algorithm:
        ################ evosax ##################
        
        if "diffusion" in algorithm:
            from evosax.algorithms import DiffusionEvolution
            es = DiffusionEvolution(population_size=hypercube_size, solution=xs[0])
            params = es.default_params
            name = "DiffusionEvolution"
        else:
            from evosax.algorithms import DifferentialEvolution
            es = DifferentialEvolution(population_size=hypercube_size, solution=xs[0])
            params = es.default_params
            # set crossover_rate to opt_rate2
            params = params.replace(elitism=False,crossover_rate=opt_rate2)
            name = "DE"

        key, subkey = jax.random.split(key)
        state = es.init(subkey, xs, likelihoods, params)

        def es_step(carry, _):
            key, state, params = carry
            # split keys for dithering, ask, tell, and boundary corrections
            key, subkey = jax.random.split(key)
            key_dither, key_ask, key_tell, key_b_low, key_b_up = jax.random.split(subkey, 5)
            # dither the mutation rate
            if name == "DE":
                differential_weight = jax.random.uniform(key_dither, minval=0.5, maxval=1.0)
                params = params.replace(differential_weight=differential_weight)
            # generate a population
            population, state = es.ask(key_ask, state, params)
            # scipy-style boundary correction: - if any parameter is out of bounds, resample it uniformly between current value and bound in the direction of the bound
            mask_lower = population < bounds[:, 0]
            mask_upper = population > bounds[:, 1]
            bounce_lower = jax.random.uniform(
                key_b_low, population.shape,
                minval=bounds[:, 0], maxval=state.population
            )
            bounce_upper = jax.random.uniform(
                key_b_up, population.shape,
                minval=state.population, maxval=bounds[:, 1]
            )
            population = jnp.where(mask_lower, bounce_lower, population)
            population = jnp.where(mask_upper, bounce_upper, population)
            # calculate fitness and update the population
            fitness = vmap_likelihood(population)
            state, metrics = es.tell(key_tell, population, fitness, state, params)
            return (key, state, params), metrics
        
        @jax.jit
        def run_es_optimization(key, state, params):
            initial_carry = (key, state, params)
            (_, final_state, _), metrics_log = jax.lax.scan(es_step, initial_carry, jnp.arange(opt_rate1))
            return final_state, metrics_log

        print(f"Starting {name} optimization...")
        start_time = time.time()
        state, metrics_log = run_es_optimization(key, state, params)
        state.fitness.block_until_ready()
        print(f"{opt_rate1} {name} iterations completed in {time.time() - start_time:.2f} seconds.")

        # # scale final_population, metrics_log["best_solution"], and metrics_log["best_solution_in_generation"] by bounds[:, 0] + 1 / (1 + exp(-x)) * (bounds[:, 1] - bounds[:, 0]) transformation
        # final_population = logistic_transform(state.population)
        # metrics_log["best_solution"] = logistic_transform(metrics_log["best_solution"])
        # metrics_log["best_solution_in_generation"] = logistic_transform(metrics_log["best_solution_in_generation"])

        if not os.path.exists("Data/Processed/results"+str(seed)[:6]):
            os.makedirs("Data/Processed/results"+str(seed)[:6])
        # save results to disk
        results_file = "Data/Processed/results"+str(seed)[:6]+"/evosax_"+name+"_"+pathogen+lockdown+option1+option2+str(seed)+".pickle"
        with open(results_file, "wb") as f:
            pickle.dump({"final_population": state.population, "final_fitness": state.fitness, "metrics_log": metrics_log}, f)
    elif "optax" in algorithm:
        # ################## optax ##################
        import optax

        # save initial points to disk
        if not os.path.exists("Data/Processed/results"+str(seed)[:6]):
            os.makedirs("Data/Processed/results"+str(seed)[:6])
        with open("Data/Processed/results"+str(seed)[:6]+"/optax_initial_points_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","wb") as f:
            pickle.dump(xs,f)

        # work in logistic-transformed space to stay within bounds
        def logistic_transform(x):
            return bounds[:, 0] + 1 / (1 + jnp.exp(-x)) * (bounds[:, 1] - bounds[:, 0])
        def inverse_logistic_transform(y):
            return -jnp.log((bounds[:, 1] - bounds[:, 0]) / (y - bounds[:, 0]) - 1)
        def new_likelihood(x):
            x_transformed = logistic_transform(x)
            lik =  likelihood(x_transformed)
            # remove nas
            lik = jnp.where(jnp.isnan(lik), likelihood_threshold*10, lik)
            return lik
        xs = inverse_logistic_transform(xs)

        schedule = optax.exponential_decay(init_value=opt_rate1, transition_steps=1000, decay_rate=opt_rate2, staircase=True)
        solver = optax.apply_if_finite(
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adabelief(learning_rate=schedule)
            ),
            max_consecutive_errors=5,
        )

        def single_step(x, opt_state):
            neglogL, grad = jax.value_and_grad(new_likelihood)(x)
            update, opt_state = solver.update(grad, opt_state, x)
            x = optax.apply_updates(x, update)
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
            final_carry, neglogL_history = jax.lax.scan(scan_body, initial_carry, jnp.arange(optax_length))
            final_xs, _ = final_carry
            return final_xs, neglogL_history

        print("Starting optax optimization...")
        start_time = time.time()
        final_xs, neglogL_history = run_optimization(xs)
        final_xs.block_until_ready()
        print(f"Optax optimization completed in {time.time() - start_time:.2f} seconds.")
        # print best parameters and likelihood
        best_index = jnp.argmin(neglogL_history[-1])
        best_params = logistic_transform(final_xs[best_index])
        best_likelihood = jnp.min(neglogL_history[-1])
        print(f"Best parameters: {best_params}")
        print(f"Best likelihood: {best_likelihood}")
        final_params = logistic_transform(final_xs)
        # save results to disk
        results_file = "Data/Processed/results"+str(seed)[:6]+"/optax_"+pathogen+lockdown+option1+option2+str(seed)+".pickle"
        with open(results_file, "wb") as f:
            pickle.dump({"best_params": best_params, "final_params": final_params, "neglogL_history": neglogL_history}, f)