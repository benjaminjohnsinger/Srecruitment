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

start_date = '2015-10-01'
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
start_idx = int(date_to_t(start_date) - date_to_t('2015-10-01'))
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
    try:
        lh = -SIS_likelihood(tests, daily_hospitalization_rates, sim_params, POINTS, STATE0, p_time_to_obs)
    except Exception as e: # Catch the specific exception
        worker_pid = os.getpid()
        print(f"!!! ERROR in worker {worker_pid} with params {x}")
        print(f"Error details: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return 1e10
    printstr = str(lh) + "," + pathogen + "," + lockdown + "," + str(seed) + "," + ",".join([str(value) for value in x])
    print(printstr, flush=True)
    return lh

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    # printstr = "neg_log_likelihood,pathogen,seed," + ",".join([key for key in bounds_dict.keys()])
    # print(printstr)
    opt = sp.optimize.differential_evolution(likelihood,bounds,popsize=desize,mutation=(0.5,max_mutation),recombination=recombination,init="halton",seed=seed,
    workers=int(os.getenv('SLURM_CPUS_ON_NODE')))
    # if there's no Data/Processed/results<seed> directory, create it
    if not os.path.exists("Data/Processed/results"+str(seed)[:6]):
        os.makedirs("Data/Processed/results"+str(seed)[:6])
    with open("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","wb") as f:
        pickle.dump(opt,f)