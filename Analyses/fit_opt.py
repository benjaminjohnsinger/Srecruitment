## BJS Jan 2025
## Fitting models to data using out-of-the-box optimisation tools

import numpy as np
import scipy as sp
import pandas as pd
import time
import pickle
import sys
import os

from vaccination import birth_vax, all_vax, flu_rate, flu_eff_coverage
import contact_model as cm
from SISn_ODEs import single_pathogen_deltas as sis_deltas
from Parameters.census_population import *
from Parameters.times_and_contacts import *

from utils import *
from demography import *
from mobility_and_import import *
from fit_MCMC import SIS_likelihood


pathogen, seed, lockdown, option1, option2, import_cap, desize, max_mutation, recombination = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9])

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
PERIOD = pd.date_range(start=START, end=END, freq='D')
POINTS = np.array(date_to_t(PERIOD))

params, p_time_to_obs, incidence = pathogen_parameters(pathogen, lockdown, CONTACT, cleaned=True)
N_S, NAG = params["N_S"], params["NAG"]

# trim incidence so that Date is between START and END
incidence.index = pd.to_datetime(incidence.index)
incidence = incidence.loc[START+pd.Timedelta(days=89):END]

## Initial conditions
STATE0 = np.zeros((2*N_S+2)*NAG)
STATE0[NAG:2*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

bounds_dict = {"WANE": [0,1e-2], "SEASONALITY": [0,1], "OFFSET": [0,1], "BETA": [0,1]}

# if option 1 is a number, use it to set obsmax
if option1.replace('.','',1).isdigit():
    obsmax = float(option1)
else:
    if pathogen == "InfluenzaA":
        obsmax = 0.03
    else:
        obsmax = 0.01

if option1 == "nb":
    bounds_dict["OVERDISPERSION"] = [-5,10]
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
    bounds_dict["AGE_OBS_1"] = [0,obsmax]
    bounds_dict["AGE_OBS_2"] = [0,obsmax]
    bounds_dict["AGE_OBS_3"] = [0,obsmax]
    bounds_dict["AGE_OBS_4"] = [0,obsmax]
    bounds_dict["AGE_OBS_5"] = [0,obsmax]
    bounds_dict["AGE_OBS_6"] = [0,obsmax]
    bounds_dict["AGE_OBS_7"] = [0,obsmax]
else:
    bounds_dict["P_OBS"] = [0,obsmax]
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

bounds = np.array([bounds_dict[key] for key in ["WANE","SEASONALITY","OFFSET","BETA","IMPORT_RATE","EXTRA_IMMUNITY","FIRST_IMMUNITY","FIRST_DIS_INF_FACTOR","S_REL1","S_REL2","D_REL1","D_REL2","P_OBS","DT1","DT2","DT3","F1","F2","F3","F4","OVERDISPERSION","AGE_OBS_YOUNG","AGE_OBS_OLD","AGE_OBS_YOUNG_OLD","AGE_OBS_MATERNAL","AGE_OBS_1","AGE_OBS_2","AGE_OBS_3","AGE_OBS_4","AGE_OBS_5","AGE_OBS_6","AGE_OBS_7"]\
if key in bounds_dict.keys()])

def likelihood(x):
    # by default, no overdispersion
    overdispersion = False
    print("time: ",time.time()-start)
    print(x)
    if np.any(x < 0) or np.any(np.isnan(x)):
        print("Invalid parameters")
        return 1e10
    sim_params = params.copy()
    sim_params["WANE"] = np.array([0.0,x[0],0.0])
    sim_params["SEASONALITY"] = x[1]
    sim_params["OFFSET"] = x[2]
    sim_params["BETA"] = x[3]
    n = 4
    sim_params["IMPORT_RATE"] = import_cap
    if ("Influenza" in pathogen) and (option2 != 'nr'):
        srel, pobsrel = constrained_immunity(x[n],x[n+1],x[n+2])
        sim_params["S_REL"] = srel
        n += 3
    elif pathogen == 'RSV':
        sim_params["S_REL"] = np.array([1,x[n],x[n]*x[n+1]])
        pobsrel = np.array([1,0.46,0.31]) # Henderson 1979
        n += 2
    else:
        sim_params["S_REL"] = np.array([1,x[n],x[n]*x[n+1]])
        pobsrel = np.array([1,x[n+2],x[n+2]*x[n+3]])
        n += 4
    if option2 != 'flexage':
        sim_params["P_OBS"] = x[n]*pobsrel
        n += 1
    else:
        sim_params["P_OBS"] = pobsrel
    if lockdown == 'FlexStepwise':
        Ts = np.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[n]*365,date_to_t('2020-03-19')+(x[n]+x[n+1])*365,date_to_t('2020-03-19')+(x[n]+x[n+1]+x[n+2])*365])
        # Fs - element 2 must be bigger than element 1, element 3 must be smaller than element 2, element 4 must be bigger than element 2
        F1 = x[n+3] # value between 0 and 1 (first lockdown)
        F2 = F1 + x[n+4] - F1*x[n+4] # value between x[n+3] and 1 (inter-lockdown)
        F3 = F2*x[n+5] # value less than F2 (second lockdown)
        F4 = F2 + x[n+6] - F2*x[n+6] # value between F2 and 1 (post-lockdown)
        Fs = np.array([1,F1,F2,F3,F4])
        @jit
        def contact(t,seasonality,offset):
            return cm.piecewise(t,Ts,Fs)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
        sim_params["contact"] = contact
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
        obs_age = np.array([x[n],x[n+1],x[n+2],x[n+3],x[n+4],x[n+5],x[n+6]])
    elif option2 == 'maternal':
        obs_age = age_detection(NAG,x[n],x[n+1],x[n+2],x[n+3],min_obs=0.025,n_infant_groups=1)
    else:
        obs_age = age_detection(NAG,x[n],x[n+1],x[n+2])
    try:
        lh = -SIS_likelihood(incidence,sim_params,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True,overdispersion=overdispersion)
    except:
        print("Error")
        return 1e10
    print("neg log likelihood: ",lh)
    return lh

start = time.time()
if __name__ == '__main__':
    opt = sp.optimize.differential_evolution(likelihood,bounds,popsize=desize,mutation=(0.5,max_mutation),recombination=recombination,init="halton",
    workers=int(os.getenv('SLURM_CPUS_ON_NODE')))

    with open("Data/Processed/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","wb") as f:
        pickle.dump(opt,f)