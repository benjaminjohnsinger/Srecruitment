## BJS Jan 2025
## Fitting models to data using out-of-the-box optimisation tools

# import jax.numpy as jnp
import scipy as sp
import pandas as pd
import time
import pickle
import sys
import os
import multiprocessing

import contact_model as cm
from SISn_ODEs import single_pathogen_deltas as sis_deltas
from Parameters.census_population import *
from Parameters.times_and_contacts import *

from utils import *
from demography import *
from mobility_and_import import *
from fit_MCMC import SIS_likelihood


pathogens, seed, option1, option2, desize, max_mutation, recombination = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])

# split pathogens by commas
pathogen_list = pathogens.split(",")

if "," in seed:
    seed_list = [int(s) for s in seed.split(",")]
else:
    seed_list = [int(seed)]

if "," in option1:
    option1_list = option1.split(",")
else:
    option1_list = [option1]

# set seed as sum of seed list
np.random.seed(sum(seed_list))

start_date = '2015-07-04'
end_date = '2025-05-01'

N_C, N_S, NAG = 2, 3, 7

EPOCH = pd.to_datetime('1970-01-01')
START = pd.to_datetime(start_date) 
END = pd.to_datetime(end_date)
PERIOD = pd.date_range(start=START, end=END, freq='D')
POINTS = jnp.array(date_to_t(PERIOD))

## Initial conditions
STATE0 = jnp.zeros((2*N_S+1,NAG))
STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
STATE0 = STATE0.at[1,:].set(1)
# # flatten initial state and add maternal immunity compartment
STATE0 = STATE0.flatten()
STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

bounds_dict = {}
bounds_dict["DT1"] = [0,1]
bounds_dict["DT2"] = [0,1]
bounds_dict["DT3"] = [0,1]
bounds_dict["F1"] = [0,1]
bounds_dict["F2"] = [0,1]
bounds_dict["F3"] = [0,1]
bounds_dict["F4"] = [0,1]

bounds = jnp.array([bounds_dict[key] for key in ["DT1","DT2","DT3","F1","F2","F3","F4"]
if key in bounds_dict.keys()])

def likelihood(x):
    lockdown_x = {"DT1": x[0], "DT2": x[1], "DT3": x[2], "F1": x[3], "F2": x[4], "F3": x[5], "F4": x[6]}

    lh = 0
    for i in range(len(pathogen_list)):
        seed = seed_list[i % len(seed_list)]
        option1 = option1_list[i % len(option1_list)]
        pathogen = pathogen_list[i]
        params, _, _, incidence, p_time_to_obs= parameters_from_DE(pathogen,"FlexStepwise",option1,option2,seed,lockdown_x)
        try:
            lh_pathogen = -SIS_likelihood(incidence,params,POINTS,STATE0,p_time_to_obs)
            lh += lh_pathogen
        except Exception as e: # Catch the specific exception
            worker_pid = os.getpid()
            print(f"!!! ERROR in worker {worker_pid} with params {x} for pathogen {pathogen}")
            print(f"Error details: {e}")
            return 1e10

    printstr = str(lh) + "," + ",".join([str(value) for value in x])
    print(printstr, flush=True)
    return lh

start = time.time()
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    opt = sp.optimize.differential_evolution(likelihood,bounds,popsize=desize,mutation=(0.5,max_mutation),recombination=recombination,
    workers=int(os.getenv('SLURM_CPUS_ON_NODE')))

    with open("Data/Processed/DE_cm_opt_"+option1+option2+str(seed)+".pickle","wb") as f:
        pickle.dump(opt,f)