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


seed, desize, max_mutation, recombination = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])

# set seed
np.random.seed(seed)

start_date = '2015-07-04'
end_date = '2023-10-01'

EPOCH = pd.to_datetime('1970-01-01')
START = pd.to_datetime(start_date) 
END = pd.to_datetime(end_date)
PERIOD = pd.date_range(start=START, end=END, freq='D')
POINTS = np.array(date_to_t(PERIOD))

from Parameters.RSV import *
# Parameters from differential evolution
RSV_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': np.array([0, 0.00412833, 0]),
'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': np.array([1, 0.25137851, 0.04869096]), 'S_AGE': S_AGE, 'I_REL': I_REL,
'P_OBS': np.array([1, 0.46, 0.31]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.01,
'BETA': 0.3098011988591176, 'SEASONALITY': 0.08886274108206305, 'OFFSET': 0.18238114357668778,
'contact': contact}
RSV_OBS_AGE = np.array([1.41156737e-03,1.42463892e-03,2.78237498e-03,1.34302008e-04,8.83682889e-05,3.16598659e-04,2.38996273e-03])
RSV_p_time_to_obs = np.genfromtxt("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
RSV_incidence = pd.read_csv("Data/Processed/KPSC_RSV_incidence_age_daily.csv",index_col=0)
from Parameters.InfluenzaA import *
# Parameters from differential evolution
InfluenzaA_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': np.array([0, 0.00659556, 0]),
'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': np.array([1, 0.97095996, 0.57030858]), 'S_AGE': S_AGE, 'I_REL': I_REL,
'P_OBS': np.array([1, 0.97095996, 0.57030858]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': flu_rate, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.01,
'BETA': 0.10235488717956664, 'SEASONALITY': 0.10445395838429006, 'OFFSET': 0.14719007971299192,
'contact': contact}
InfluenzaA_OBS_AGE = np.array([0.00069735,0.00111811,0.00078817,0.00014056,0.00021106,0.00052153,0.00497795])
InfluenzaA_p_time_to_obs = np.genfromtxt("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
InfluenzaA_incidence = pd.read_csv("Data/Processed/KPSC_Influenza_A_incidence_age_daily.csv",index_col=0)
from Parameters.InfluenzaB import *
# Parameters from differential evolution
InfluenzaB_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': np.array([0, 0.00864397, 0]),
'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': np.array([1, 0.92976988, 0.66999541]), 'S_AGE': S_AGE, 'I_REL': I_REL,
'P_OBS': np.array([1, 0.86061014, 0.69854585]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': flu_rate, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.01,
'BETA': 0.09525015457616015, 'SEASONALITY': 0.1684220494009686, 'OFFSET': 0.18437652662834292,
'contact': contact}
InfluenzaB_OBS_AGE = np.array([0.0011928,0.00241971,0.00098738,0.00020903,0.00021118,0.00055036,0.00451088])
InfluenzaB_p_time_to_obs = np.genfromtxt("Data/Processed/Influenza_B_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
InfluenzaB_incidence = pd.read_csv("Data/Processed/KPSC_Influenza_B_incidence_age_daily.csv",index_col=0)
from Parameters.Metapneumovirus import *
# Parameters from differential evolution
Metapneumovirus_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': np.array([0, 0.00941287, 0]),
'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': np.array([1, 0.28313607, 0.07930067]), 'S_AGE': S_AGE, 'I_REL': I_REL,
'P_OBS': np.array([1, 0.80565999, 0.54868475]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.01,
'BETA': 0.3048358628689303, 'SEASONALITY': 0.06422389269804452, 'OFFSET': 0.23489341732013758,
'contact': contact}
Metapneumovirus_OBS_AGE = np.array([8.30090073e-04,1.69293756e-03,1.88729824e-03,1.52797497e-04,9.49137803e-05,5.20598038e-04,4.61001222e-03])
Metapneumovirus_p_time_to_obs = np.genfromtxt("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
Metapneumovirus_incidence = pd.read_csv("Data/Processed/KPSC_Metapneumovirus_incidence_age_daily.csv",index_col=0)
from Parameters.Adenovirus import *
# Parameters from differential evolution
Adenovirus_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': np.array([0, 0.00839394, 0]),
'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': np.array([1, 0.32487475, 0.16268084]), 'S_AGE': S_AGE, 'I_REL': I_REL,
'P_OBS': np.array([1, 0.29454725, 0.21554204]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.01,
'BETA': 0.17515151709090215, 'SEASONALITY': 0.043289308995968556, 'OFFSET': 0.3567702562314214,
'contact': contact}
Adenovirus_OBS_AGE = np.array([0.00061644,0.00325062,0.00441007,0.00015952,0.00020786,0.00083746,0.00320991])
Adenovirus_p_time_to_obs = np.genfromtxt("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
Adenovirus_incidence = pd.read_csv("Data/Processed/KPSC_Adenovirus_incidence_age_daily.csv",index_col=0)
from Parameters.Parainfluenza3 import *
# Parameters from differential evolution
Parainfluenza3_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': np.array([0, 0.00707944, 0]),
'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': np.array([1, 0.61294058, 0.09778999]), 'S_AGE': S_AGE, 'I_REL': I_REL,
'P_OBS': np.array([1, 0.7489466 , 0.63736202]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.01,
'BETA': 0.136723789107368, 'SEASONALITY': 0.05120171277842056, 'OFFSET': 0.3905029750372729,
'contact': contact}
Parainfluenza3_OBS_AGE = np.array([4.48328458e-04,2.89073191e-03,2.24725437e-03,7.67201177e-05,9.06454200e-05,3.14410058e-04,4.72455456e-03])
Parainfluenza3_p_time_to_obs = np.genfromtxt("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
Parainfluenza3_incidence = pd.read_csv("Data/Processed/KPSC_Parainfluenza3_incidence_age_daily.csv",index_col=0)


## Initial conditions
STATE0 = np.zeros((2*N_S+2)*NAG)
STATE0[NAG:2*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

bounds_dict = {}
bounds_dict["DT1"] = [0,1]
bounds_dict["DT2"] = [0,1]
bounds_dict["DT3"] = [0,1]
bounds_dict["F1"] = [0,1]
bounds_dict["F2"] = [0,1]
bounds_dict["F3"] = [0,1]
bounds_dict["F4"] = [0,1]

bounds = np.array([bounds_dict[key] for key in ["DT1","DT2","DT3","F1","F2","F3","F4"]
if key in bounds_dict.keys()])

def likelihood(x):
    # by default, no overdispersion
    overdispersion = False
    print("time: ",time.time()-start)
    print(x)
    if np.any(x < 0) or np.any(np.isnan(x)):
        print("Invalid parameters")
        return 1e10
    Ts = np.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[0]*365,date_to_t('2020-03-19')+(x[0]+x[1])*365,date_to_t('2020-03-19')+(x[0]+x[1]+x[2])*365])
    # Fs - element 2 must be bigger than element 1, element 3 must be smaller than element 2, element 4 must be bigger than element 2
    F1 = x[3] # value between 0 and 1 (first lockdown)
    F2 = F1 + x[4] - F1*x[4] # value between x[3] and 1 (inter-lockdown)
    F3 = F2*x[5] # value less than F2 (second lockdown)
    F4 = F2 + x[6] - F2*x[6] # value between F2 and 1 (post-lockdown)
    Fs = np.array([1,F1,F2,F3,F4])
    @jit
    def contact(t,seasonality,offset):
        return cm.piecewise(t,Ts,Fs)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
    RSV_params["contact"] = contact
    InfluenzaA_params["contact"] = contact
    InfluenzaB_params["contact"] = contact
    Metapneumovirus_params["contact"] = contact
    Adenovirus_params["contact"] = contact
    Parainfluenza3_params["contact"] = contact
    try:
        lh_RSV = -SIS_likelihood(RSV_incidence,RSV_params,POINTS,STATE0,RSV_OBS_AGE,RSV_p_time_to_obs,age=True,incidence=True,overdispersion=overdispersion)
        lh_InfluenzaA = -SIS_likelihood(InfluenzaA_incidence,InfluenzaA_params,POINTS,STATE0,InfluenzaA_OBS_AGE,InfluenzaA_p_time_to_obs,age=True,incidence=True,overdispersion=overdispersion)
        lh_InfluenzaB = -SIS_likelihood(InfluenzaB_incidence,InfluenzaB_params,POINTS,STATE0,InfluenzaB_OBS_AGE,InfluenzaB_p_time_to_obs,age=True,incidence=True,overdispersion=overdispersion)
        lh_Metapneumovirus = -SIS_likelihood(Metapneumovirus_incidence,Metapneumovirus_params,POINTS,STATE0,Metapneumovirus_OBS_AGE,Metapneumovirus_p_time_to_obs,age=True,incidence=True,overdispersion=overdispersion)
        lh_Adenovirus = -SIS_likelihood(Adenovirus_incidence,Adenovirus_params,POINTS,STATE0,Adenovirus_OBS_AGE,Adenovirus_p_time_to_obs,age=True,incidence=True,overdispersion=overdispersion)
        lh_Parainfluenza3 = -SIS_likelihood(Parainfluenza3_incidence,Parainfluenza3_params,POINTS,STATE0,Parainfluenza3_OBS_AGE,Parainfluenza3_p_time_to_obs,age=True,incidence=True,overdispersion=overdispersion)
        lh = lh_RSV + lh_InfluenzaA + lh_InfluenzaB + lh_Metapneumovirus + lh_Adenovirus + lh_Parainfluenza3
    except:
        print("Error")
        return 1e10
    print("neg log likelihood: ",lh)
    return lh

start = time.time()
if __name__ == '__main__':
    opt = sp.optimize.differential_evolution(likelihood,bounds,popsize=desize,mutation=(0.5,max_mutation),recombination=recombination,
    workers=int(os.getenv('SLURM_CPUS_ON_NODE')))

    with open("Data/Processed/DE_cm_opt_"+str(seed)+".pickle","wb") as f:
        pickle.dump(opt,f)