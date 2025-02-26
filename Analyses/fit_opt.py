## BJS Jan 2025
## Fitting models to data using out-of-the-box optimisation tools

import numpy as np
import matplotlib.pyplot as plt
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
from Parameters.RSV import *

from utils import *
from demography import *
from mobility_and_import import *
from clustering import *
from sim_grid import *
from plotting import *
from fit_MCMC import *


pathogen, seed = sys.argv[1], int(sys.argv[2])

# set seed
np.random.seed(seed)

## Initial conditions
STATE0 = np.zeros((2*N_S+2)*NAG)
STATE0[NAG:2*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

if pathogen == 'RSV':
    from Parameters.RSV import *
    # Parameters for the ODE
    params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
    'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
    'contact': contact}
    p_time_to_obs = np.genfromtxt("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
    incidence = pd.read_csv("Data/Processed/KPSC_RSV_incidence_age_daily.csv",index_col=0)
elif pathogen == 'InfluenzaA':
    from Parameters.InfluenzaA import *
    # Parameters for the ODE
    params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': flu_rate, 'BCOV': BCOV,
    'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
    'contact': contact}
    p_time_to_obs = np.genfromtxt("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
    incidence = pd.read_csv("Data/Processed/KPSC_Influenza_A_incidence_age_daily.csv",index_col=0)
elif pathogen == 'InfluenzaB':
    from Parameters.InfluenzaB import *
    # Parameters for the ODE
    params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': flu_rate, 'BCOV': BCOV,
    'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
    'contact': contact}
    p_time_to_obs = np.genfromtxt("Data/Processed/Influenza_B_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
    incidence = pd.read_csv("Data/Processed/KPSC_Influenza_B_incidence_age_daily.csv",index_col=0)

# if pathogen == 'RSV':
#     bounds = np.array([[0,1e-2], # WANE
#     [0,1], # SEASONALITY
#     [0,1], # OFFSET
#     [0,1], # BETA
#     [0,1e-9], # IMPORT_RATE
#     [0,1], # P_OBS
#     [0,1], # AGE_OBS - young_immunity
#     [0,1], # AGE_OBS - old_immunity
#     [0,1]]) # AGE_OBS - young_old
#     def likelihood(x):
#         print("time: ",time.time()-start)
#         print(x)
#         if np.any(x < 0) or np.any(np.isnan(x)):
#             print("Invalid parameters")
#             return 0
#         sim_params = params.copy()
#         sim_params["WANE"] = np.array([0.0,x[0],0.0])
#         sim_params["SEASONALITY"] = x[1]
#         sim_params["OFFSET"] = x[2]
#         sim_params["BETA"] = x[3]
#         sim_params["IMPORT_RATE"] = x[4]
#         sim_params["P_OBS"] = x[5]*params["P_OBS"]/params["P_OBS"][0]
#         obs_age = age_detection(NAG,x[6],x[7],x[8])
#         try:
#             lh = -SIS_likelihood(incidence,sim_params,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)
#         except:
#             print("Error")
#             return 0
#         print("likelihood: ",lh)
#         return lh
# else:
bounds = np.array([[0,1e-2], # WANE
[0,1], # SEASONALITY
[0,1], # OFFSET
[0,1], # BETA
[0,1e-9], # IMPORT_RATE
[0,1], # S_REL - immunity after second infection above minimum
[0,1], # S_REL - immunity after first infection above minimum
[0,1], # S_REL - relative infection and disease immunity after first infection
[0,1], # P_OBS
[0,1], # AGE_OBS - young_immunity
[0,1], # AGE_OBS - old_immunity
[0,1]]) # AGE_OBS - young_old
def likelihood(x):
    print("time: ",time.time()-start)
    print(x)
    if np.any(x < 0) or np.any(np.isnan(x)):
        print("Invalid parameters")
        return 0
    sim_params = params.copy()
    sim_params["WANE"] = np.array([0.0,x[0],0.0])
    sim_params["SEASONALITY"] = x[1]
    sim_params["OFFSET"] = x[2]
    sim_params["BETA"] = x[3]
    sim_params["IMPORT_RATE"] = x[4]
    srel, pobsrel = constrained_immunity(x[5],x[6],x[7])
    sim_params["S_REL"] = srel
    sim_params["P_OBS"] = x[8]*pobsrel
    obs_age = age_detection(NAG,x[9],x[10],x[11])
    try:
        lh = -SIS_likelihood(incidence,sim_params,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)
    except:
        print("Error")
        return 0
    print("likelihood: ",lh)
    return lh

start = time.time()
if __name__ == '__main__':
    opt = sp.optimize.differential_evolution(likelihood,bounds,workers=int(os.getenv('SLURM_CPUS_ON_NODE')))

    # with open("Data/Processed/DE_opt_"+pathogen+str(seed)+"_population.pickle","wb") as f:
    #     pickle.dump(opt.population,f)
    with open("Data/Processed/DE_opt_"+pathogen+str(seed)+".pickle","wb") as f:
        pickle.dump(opt,f)

    # save optimized parameters
    opt_params = params.copy()
    opt_params["WANE"] = np.array([0.0,opt.x[0],0.0])
    opt_params["SEASONALITY"] = opt.x[1]
    opt_params["OFFSET"] = opt.x[2]
    opt_params["BETA"] = opt.x[3]
    opt_params["IMPORT_RATE"] = opt.x[4]
    srel, pobsrel = constrained_immunity(opt.x[5],opt.x[6],opt.x[7])
    opt_params["S_REL"] = srel
    opt_params["P_OBS"] = opt.x[8]*pobsrel
    obs_age = age_detection(NAG,opt.x[9],opt.x[10],opt.x[11])
    with open("Data/Processed/DE_opt_params_"+pathogen+str(seed)+".pickle","wb") as f:
        pickle.dump(opt_params,f)
    with open("Data/Processed/DE_opt_obs_age_"+pathogen+str(seed)+".pickle","wb") as f:
        pickle.dump(obs_age,f)
    print(opt_params)
    print(obs_age)

    print(time.time()-start)