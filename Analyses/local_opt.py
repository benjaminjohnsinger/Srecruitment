## Local optimization from parameters passed through the command line
## BJS Feb 2025

import time
import numpy as np
import scipy as sp
import pandas as pd

from vaccination import birth_vax, birth_vax, all_vax, flu_rate, flu_eff_coverage
import contact_model as cm
from SISn_ODEs import single_pathogen_deltas as sis_deltas
from Parameters.census_population import *
from Parameters.times_and_contacts import *

from utils import *
from demography import *
from mobility_and_import import *
from fit_MCMC import SIS_likelihood

pathogen = 'InfluenzaA'
initial_params = [[0.08,0.1],[0.085,0.1],[0.09,0.1],[0.095,0.1],
[0.08,0.2],[0.085,0.2],[0.09,0.2],[0.095,0.2]]

if pathogen == 'RSV':
    from Parameters.RSV import *
    p_time_to_obs = np.genfromtxt("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
    incidence = pd.read_csv("Data/Processed/KPSC_RSV_incidence_age_daily.csv",index_col=0)
elif pathogen == 'InfluenzaA':
    from Parameters.InfluenzaA import *
    p_time_to_obs = np.genfromtxt("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
    incidence = pd.read_csv("Data/Processed/KPSC_Influenza_A_incidence_age_daily.csv",index_col=0)
elif pathogen == 'InfluenzaB':
    from Parameters.InfluenzaB import *
    p_time_to_obs = np.genfromtxt("Data/Processed/Influenza_B_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
    incidence = pd.read_csv("Data/Processed/KPSC_Influenza_B_incidence_age_daily.csv",index_col=0)

## Initial conditions
STATE0 = np.zeros((2*N_S+2)*NAG)
STATE0[NAG:2*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

# Parameters for the ODE
params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': flu_rate, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
'contact': contact}

# nelders-mead optimization
# def likelihood(x):
#     sim_params = params.copy()
#     sim_params["WANE"] = np.array([0.0,x[0],0.0])
#     sim_params["SEASONALITY"] = x[1]
#     sim_params["OFFSET"] = x[2]
#     sim_params["BETA"] = x[3]
#     sim_params["IMPORT_RATE"] = x[4]
#     srel, pobsrel = constrained_immunity(x[5],x[6],x[7])
#     sim_params["S_REL"] = srel
#     sim_params["P_OBS"] = x[8]*pobsrel
#     obs_age = age_detection(NAG,x[9],x[10],x[11])
#     return -SIS_likelihood(incidence,sim_params,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)=

def likelihood(x):
    sim_params = params.copy()
    sim_params["BETA"] = x[0]
    return -SIS_likelihood(incidence,sim_params,POINTS,STATE0,OBS_AGE,p_time_to_obs,age=True,incidence=True)

def optimizer(start):
    opt = sp.optimize.minimize(likelihood,start,method='Nelder-Mead')
    return opt.x

from multiprocessing import Pool

start_multi = time.time()
if __name__ == '__main__':
    with Pool(4) as p:
        print(p.map(optimizer,initial_params))
end_multi = time.time()

start_single = time.time()
for start in initial_params:
    print(optimizer(start))
end_single = time.time()

print("Time for multiprocessing: ",end_multi-start_multi, "Time for single processing: ",end_single-start_single)