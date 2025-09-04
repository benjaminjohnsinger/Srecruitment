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

from vaccination import flu_rate, flu_eff_coverage
import contact_model as cm
from SISn_ODEs import single_pathogen_deltas as sis_deltas
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
    option2 = "maternal" #this is super hacky sorry


EPOCH = pd.to_datetime('1970-01-01')
START = pd.to_datetime(start_date) 
END = pd.to_datetime(end_date)
FULL_PERIOD = pd.date_range(start=EPOCH, end=END, freq='D')
FULL_POINTS = np.array(date_to_t(FULL_PERIOD))
PERIOD = pd.date_range(start=START, end=END, freq='D')
POINTS = np.array(date_to_t(PERIOD))

REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, incidence = pathogen_parameters(pathogen, import_multiplier=import_multiplier)
N_S, NAG = 3, 7
CONTACT_MATRIX = np.asarray(pd.read_csv('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', header=None).values)
BIRTH_RATE = np.genfromtxt('Data/Processed/birth_rate_daily.csv', delimiter=',')
age_pops = np.genfromtxt('Data/Processed/age_pops_daily.csv', delimiter=',')

# trim incidence so that Date is between START and END
incidence.index = pd.to_datetime(incidence.index)
incidence = incidence.loc[START+pd.Timedelta(days=89):END]

## Initial conditions
STATE0_shaped = jnp.zeros((2*N_S+1,NAG))
STATE0_shaped = STATE0_shaped.at[0,:].set(CENSUS_AGE_POP-1)
STATE0_shaped = STATE0_shaped.at[1,:].set(1)
STATE0 = jnp.concatenate((jnp.array([0]), STATE0_shaped.flatten()))

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
    bounds_dict["S_REL1"] = [0.1,1]
    bounds_dict["S_REL2"] = [0.1,1]
else:
    bounds_dict["S_REL1"] = [0.1,1]
    bounds_dict["S_REL2"] = [0.1,1]
    bounds_dict["D_REL1"] = [0.1,1]
    bounds_dict["D_REL2"] = [0.1,1]
if option2 == "flexage":
    bounds_dict["AGE_OBS_1"] = [0,0.005]
    bounds_dict["AGE_OBS_2"] = [0,0.005]
    bounds_dict["AGE_OBS_3"] = [0,0.005]
    bounds_dict["AGE_OBS_4"] = [0,0.005]
    bounds_dict["AGE_OBS_5"] = [0,0.005]
    bounds_dict["AGE_OBS_6"] = [0,0.005]
    bounds_dict["AGE_OBS_7"] = [0,0.005]
else:
    bounds_dict["P_OBS"] = [0,0.01]
    bounds_dict["AGE_OBS_YOUNG"] = [0,1]
    bounds_dict["AGE_OBS_OLD"] = [0,1]
    bounds_dict["AGE_OBS_YOUNG_OLD"] = [0,1]
if option2 == "maternal":
    bounds_dict["AGE_OBS_MATERNAL"] = [0,1]
if lockdown == "FlexStepwise":
    bounds_dict["DT1"] = [0,1]
    bounds_dict["DT2"] = [0,1]
    bounds_dict["DT3"] = [0,1]
    bounds_dict["F1"] = [0,1]
    bounds_dict["F2"] = [0,1]
    bounds_dict["F3"] = [0,1]
    bounds_dict["F4"] = [0,1]

bounds = np.array([bounds_dict[key] for key in ["WANE","SEASONALITY","OFFSET","BETA","IMPORT_RATE","EXTRA_IMMUNITY","FIRST_IMMUNITY","FIRST_DIS_INF_FACTOR","S_REL1","S_REL2","D_REL1","D_REL2","P_OBS","MATERNAL_IMMUNITY","DT1","DT2","DT3","F1","F2","F3","F4","OVERDISPERSION","AGE_OBS_YOUNG","AGE_OBS_OLD","AGE_OBS_YOUNG_OLD","AGE_OBS_MATERNAL","AGE_OBS_1","AGE_OBS_2","AGE_OBS_3","AGE_OBS_4","AGE_OBS_5","AGE_OBS_6","AGE_OBS_7"]\
if key in bounds_dict.keys()])

def likelihood(x):
    # by default, no overdispersion
    overdispersion = False
    print("time: ",time.time()-start)
    print(x)
    if np.any(x < 0) or np.any(np.isnan(x)):
        print("Invalid parameters")
        return 1e10
    WANE = np.array([0.0,x[0],0.0])
    SEASONALITY = x[1]
    OFFSET = x[2]
    BETA = x[3]
    n = 4
    if ("Influenza" in pathogen) and (option2 != 'nr'):
        srel, pobsrel = constrained_immunity(x[n],x[n+1],x[n+2])
        S_REL = srel
        n += 3
    elif pathogen == 'RSV':
        S_REL = np.array([1,x[n],x[n]*x[n+1]])
        pobsrel = np.array([1,0.46,0.31]) # Henderson 1979
        n += 2
    else:
        S_REL = np.array([1,x[n],x[n]*x[n+1]])
        pobsrel = np.array([1,x[n+2],x[n+2]*x[n+3]])
        n += 4
    if option2 != 'flexage':
        P_OBS = x[n]*pobsrel
        n += 1
    else:
        P_OBS = pobsrel
    if option1 == "maternal":
        MATERNAL_IMMUNITY = x[n]
        n+=1
    elif option1 =="fixed_maternal":
        MATERNAL_IMMUNITY = 1
    else:
        MATERNAL_IMMUNITY = 0
    if lockdown == 'FlexStepwise':
        TT = np.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[n]*365,date_to_t('2020-03-19')+(x[n]+x[n+1])*365,date_to_t('2020-03-19')+(x[n]+x[n+1]+x[n+2])*365])
        # Fs - element 2 must be bigger than element 1, element 3 must be smaller than element 2, element 4 must be bigger than element 2
        F1 = x[n+3] # value between 0 and 1 (first lockdown)
        F2 = F1 + x[n+4] - F1*x[n+4] # value between x[n+3] and 1 (inter-lockdown)
        F3 = F2*x[n+5] # value less than F2 (second lockdown)
        F4 = F2 + x[n+6] - F2*x[n+6] # value between F2 and 1 (post-lockdown)
        FF = np.array([1,F1,F2,F3,F4])
        PIECEWISE_CONTACT = np.array([cm.piecewise(t, TT, FF, steepness=0.2) for t in FULL_POINTS])
        RELATIVE_CONTACT = PIECEWISE_CONTACT*(1+SEASONALITY*np.cos(2*np.pi*((FULL_POINTS-274)/365-OFFSET)))
        n += 7
    if option1 == 'nb':
        overdispersion = np.exp(x[n])
        n += 1
    if option2 == 'flexage':
        # # barycentric parameterization of the age observation probabilities
        # obs_age = np.zeros((7))
        # remaining = 1.0
        # for i in range(1,7):
        #     allocation = x[n+i-1]*remaining
        #     obs_age[i] = allocation
        #     remaining -= allocation
        # obs_age[0] = remaining
        # obs_age = obs_age/np.max(obs_age)
        OBS_AGE = np.array([x[n],x[n+1],x[n+2],x[n+3],x[n+4],x[n+5],x[n+6]])
    elif option2 == 'maternal':
        OBS_AGE = age_detection(NAG,x[n],x[n+1],x[n+2],x[n+3],min_obs=0.025,n_infant_groups=1)
    else:
        OBS_AGE = age_detection(NAG,x[n],x[n+1],x[n+2])
    if ("Influenza" in pathogen):
        time0 = time.time()
        VAX_RATE = np.array([flu_rate(t, S_REL*P_OBS, age_pops[t], AGING_RATE) for t in FULL_POINTS])
        print("vax rate time = ", time.time()-time0)
    else:
        VAX_RATE = np.zeros((len(FULL_POINTS),NAG))
    
    sim_params = (AGING_RATE, BIRTH_RATE, CONTACT_MATRIX,
              BETA, WANE, S_REL, P_OBS, OBS_AGE, RELATIVE_CONTACT, VAX_RATE, MATERNAL_IMMUNITY,
              REC_UP, REC_SAME, IMPORT_STRENGTH)

    try:
        lh = -SIS_likelihood(incidence,sim_params,POINTS,STATE0,p_time_to_obs,age=True,incidence=True,overdispersion=overdispersion)
    except Exception as e: # Catch the specific exception
        worker_pid = os.getpid()
        print(f"!!! ERROR in worker {worker_pid} with params {x}")
        print(f"Error details: {e}")
        return 1e10
    print("neg log likelihood: ",lh)
    return lh

start = time.time()
if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn', force=True)
    opt = sp.optimize.differential_evolution(likelihood,bounds,popsize=desize,mutation=(0.5,max_mutation),recombination=recombination,init="halton",
    workers=4)
    # workers=int(os.getenv('SLURM_CPUS_ON_NODE')))

    with open("Data/Processed/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","wb") as f:
        pickle.dump(opt,f)