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
RSV_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': np.array([0.        , 0.00438131, 0.        ]), 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': np.array([1.        , 0.10871732, 0.0372482 ]), 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': np.array([0.01273131, 0.01272225, 0.00913129]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.008897365162093825, 'BETA': 0.47973501805559776, 'SEASONALITY': 0.09607721848972506, 'OFFSET': 0.12504213724959057,
'contact': contact}
RSV_OBS_AGE = np.array([1,0.68056091,0.36112182,0.05,0.05,0.09321769,0.9479402])
RSV_p_time_to_obs = np.genfromtxt("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
RSV_incidence = pd.read_csv("Data/Processed/KPSC_RSV_incidence_age_daily.csv",index_col=0)
from Parameters.InfluenzaA import *
# Parameters from differential evolution
InfluenzaA_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': np.array([0.        , 0.00727684, 0.        ]), 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': np.array([1.        , 0.99047099, 0.53477431]), 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': np.array([0.09457368, 0.09275364, 0.06399125]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': flu_rate, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.00963412937332462, 'BETA': 0.08481978933438244, 'SEASONALITY': 0.19406057909181484, 'OFFSET': 0.1302407109070668,
'contact': contact}
InfluenzaA_OBS_AGE = np.array([0.75800537,0.47379857,0.18959177,0.05,0.05,0.11364865,1])
InfluenzaA_p_time_to_obs = np.genfromtxt("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
InfluenzaA_incidence = pd.read_csv("Data/Processed/KPSC_Influenza_A_incidence_age_daily.csv",index_col=0)
from Parameters.InfluenzaB import *
# Parameters from differential evolution
InfluenzaB_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': np.array([0.        , 0.00823164, 0.        ]), 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': np.array([1.        , 0.99663903, 0.7347241 ]), 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': np.array([0.01702199, 0.01682393, 0.01384398]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': flu_rate, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.004563430080876733, 'BETA': 0.08735017404726525, 'SEASONALITY': 0.18178149815535793, 'OFFSET': 0.18116258019278764,
'contact': contact}
InfluenzaB_OBS_AGE = np.array([0.81452577,0.55606827,0.29761078,0.05,0.05,0.10119786,1])
InfluenzaB_p_time_to_obs = np.genfromtxt("Data/Processed/Influenza_B_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
InfluenzaB_incidence = pd.read_csv("Data/Processed/KPSC_Influenza_B_incidence_age_daily.csv",index_col=0)


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
    # try:
    lh_RSV = -SIS_likelihood(RSV_incidence,RSV_params,POINTS,STATE0,RSV_OBS_AGE,RSV_p_time_to_obs,age=True,incidence=True,overdispersion=overdispersion)
    lh_InfluenzaA = -SIS_likelihood(InfluenzaA_incidence,InfluenzaA_params,POINTS,STATE0,InfluenzaA_OBS_AGE,InfluenzaA_p_time_to_obs,age=True,incidence=True,overdispersion=overdispersion)
    lh_InfluenzaB = -SIS_likelihood(InfluenzaB_incidence,InfluenzaB_params,POINTS,STATE0,InfluenzaB_OBS_AGE,InfluenzaB_p_time_to_obs,age=True,incidence=True,overdispersion=overdispersion)
    lh = lh_RSV + lh_InfluenzaA + lh_InfluenzaB
    # except:
    #     print("Error")
    #     return 1e10
    print("neg log likelihood: ",lh)
    return lh

start = time.time()
if __name__ == '__main__':
    opt = sp.optimize.differential_evolution(likelihood,bounds,popsize=desize,mutation=(0.5,max_mutation),recombination=recombination,
    workers=int(os.getenv('SLURM_CPUS_ON_NODE')))

    with open("Data/Processed/DE_cm_opt_"+str(seed)+".pickle","wb") as f:
        pickle.dump(opt,f)