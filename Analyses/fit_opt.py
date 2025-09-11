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

from JAX_ODEs import deltas as sis_deltas
from Parameters.census_population import *
from Parameters.times_and_contacts import *

from utils import *
from fit_MCMC import SIS_likelihood


pathogen, seed, lockdown, option1, option2, import_multiplier, desize, max_mutation, recombination = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9])

# set seed
np.random.seed(seed)

start_date = '2015-07-04'
end_date = '2023-10-01'
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

REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, incidence = pathogen_parameters(pathogen, import_multiplier=import_multiplier)
N_S, NAG = 3, 7

# trim incidence so that Date is between START and END
start_idx = int(date_to_t(start_date) - date_to_t('2015-07-04'))
end_idx = int(date_to_t(end_date) - date_to_t('2015-07-04')) + 1
incidence = incidence[start_idx:end_idx, :]

## Initial conditions
STATE0 = jnp.zeros((2*N_S+1,NAG))
STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
STATE0 = STATE0.at[1,:].set(1)
# # flatten initial state and add maternal immunity compartment
STATE0 = STATE0.flatten()
STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

bounds_dict = {"WANE": [0,1e-2], "SEASONALITY": [0,1], "OFFSET": [0,1], "BETA": [0,1]}

if option1 == "nb":
    bounds_dict["OVERDISPERSION"] = [-5,10]
elif option1 == "maternal":
    bounds_dict["MATERNAL_IMMUNITY"] = [0,10]
if ("Influenza" in pathogen) and (option2 != "nr"):
    bounds_dict["EXTRA_IMMUNITY"] = [0,1]
    bounds_dict["FIRST_IMMUNITY"] = [0.1,1]
    bounds_dict["FIRST_DIS_INF_FACTOR"] = [0,1]
elif pathogen == "RSV":
    bounds_dict["S_REL1"] = bounds_dict["S_REL2"] = [0.1,1]
else:
    bounds_dict["S_REL1"] = bounds_dict["S_REL2"] = bounds_dict["D_REL1"] = bounds_dict["D_REL2"] = [0.1,1]
if option2 == "flexage":
    bounds_dict["AGE_OBS_1"] = bounds_dict["AGE_OBS_2"] = bounds_dict["AGE_OBS_3"] = bounds_dict["AGE_OBS_4"] = bounds_dict["AGE_OBS_5"] = bounds_dict["AGE_OBS_6"] = bounds_dict["AGE_OBS_7"] = [0,0.005]
else:
    bounds_dict["P_OBS"] = [0,0.01]
    bounds_dict["AGE_OBS_YOUNG"] = bounds_dict["AGE_OBS_OLD"] = bounds_dict["AGE_OBS_YOUNG_OLD"] = [0,1]
if lockdown == "FlexStepwise":
    bounds_dict["DT1"] = bounds_dict["DT2"] = bounds_dict["DT3"] = bounds_dict["F1"] = bounds_dict["F2"] = bounds_dict["F3"] = bounds_dict["F4"] = [0,1]
elif lockdown == "Mobility":
    bounds_dict["F1"] = [0,1.5]
    bounds_dict["CBASE"] = [0,1.5]
elif lockdown == "Mobility2":
    bounds_dict["F1"] = bounds_dict["F2"] = [0,1]
    bounds_dict["CBASE"] = [0,1.5]

# reorder bounds_dict to match order in x
bounds_dict = {key: bounds_dict[key] for key in ["WANE","SEASONALITY","OFFSET","BETA","IMPORT_RATE","EXTRA_IMMUNITY","FIRST_IMMUNITY","FIRST_DIS_INF_FACTOR","S_REL1","S_REL2","D_REL1","D_REL2","P_OBS","MATERNAL_IMMUNITY","DT1","DT2","DT3","F0","F1","F2","F3","F4","CBASE","OVERDISPERSION","AGE_OBS_YOUNG","AGE_OBS_OLD","AGE_OBS_YOUNG_OLD","AGE_OBS_MATERNAL","AGE_OBS_1","AGE_OBS_2","AGE_OBS_3","AGE_OBS_4","AGE_OBS_5","AGE_OBS_6","AGE_OBS_7"]\
    if key in bounds_dict.keys()}
bounds = jnp.array(list(bounds_dict.values()))

def likelihood(x):
    print('try')
    sim_params = x_to_params(x, pathogen, lockdown, option1, option2)
    try:
        lh = -SIS_likelihood(incidence,sim_params,POINTS,STATE0,p_time_to_obs,age=True,incidence=True,overdispersion=False)
    except Exception as e: # Catch the specific exception
        worker_pid = os.getpid()
        print(f"!!! ERROR in worker {worker_pid} with params {x}")
        print(f"Error details: {e}")
        return 1e10
    printstr = str(lh) + "," + ",".join([str(value) for value in x])
    print(printstr)
    return lh

start = time.time()
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    printstr = "neg_log_likelihood," + ",".join([key for key in bounds_dict.keys()])
    print(printstr)
    opt = sp.optimize.differential_evolution(likelihood,bounds,popsize=desize,mutation=(0.5,max_mutation),recombination=recombination,init="halton",
    workers = 4)
    # workers=int(os.getenv('SLURM_CPUS_ON_NODE')))

    with open("Data/Processed/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","wb") as f:
        pickle.dump(opt,f)