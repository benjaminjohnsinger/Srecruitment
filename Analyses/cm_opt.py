## BJS Jan 2025
## Fitting models to data using out-of-the-box optimisation tools

import jax.numpy as jnp
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
jnp.random.seed(seed)

start_date = '2015-07-04'
end_date = '2023-10-01'

EPOCH = pd.to_datetime('1970-01-01')
START = pd.to_datetime(start_date) 
END = pd.to_datetime(end_date)
PERIOD = pd.date_range(start=START, end=END, freq='D')
POINTS = jnp.array(date_to_t(PERIOD))

from Parameters.RSV import *
# Parameters from differential evolution
RSV_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': jnp.array([0.        , 0.00136468, 0.        ]), 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': jnp.array([1.        , 0.10032712, 0.04941595]), 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': jnp.array([0.04535184, 0.02086185, 0.01405907]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.01, 'BETA': 0.5001441612672278, 'SEASONALITY': 0.0875344346283, 'OFFSET': 0.13221726766772357,
'contact': contact}
RSV_OBS_AGE = jnp.array([0.24594159,0.1857486,0.12555562,0.02660094,0.025,0.09684663,1])
RSV_p_time_to_obs = jnp.genfromtxt("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',',dtype=jnp.float64)
RSV_incidence = pd.read_csv("Data/Processed/KPSC_RSV_incidence_age_daily.csv",index_col=0)
from Parameters.InfluenzaA import *
# Parameters from differential evolution
InfluenzaA_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': jnp.array([0.        , 0.0047965, 0.        ]), 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': jnp.array([1.        , 0.85063441, 0.48050463]), 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': jnp.array([0.0961377 , 0.08948005, 0.06323804]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': flu_rate, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.01, 'BETA': 0.10275083227366294, 'SEASONALITY': 0.1734352740318053, 'OFFSET': 0.11985684117246775,
'contact': contact}
InfluenzaA_OBS_AGE = jnp.array([0.40500959,0.25868484,0.11236009,0.02751321,0.025,0.09997146,1])
InfluenzaA_p_time_to_obs = jnp.genfromtxt("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',',dtype=jnp.float64)
InfluenzaA_incidence = pd.read_csv("Data/Processed/KPSC_Influenza_A_incidence_age_daily.csv",index_col=0)
from Parameters.Adenovirus import *
# Parameters from differential evolution
Adenovirus_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': jnp.array([0.        , 0.00528484, 0.        ]), 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': jnp.array([1.        , 0.1021544 , 0.10144828]), 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': jnp.array([0.01656661, 0.00179079, 0.00091177]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.01, 'BETA': 0.7430548452606568, 'SEASONALITY': 0.020803749446702102, 'OFFSET': 0.2548373821056173,
'contact': contact}
Adenovirus_OBS_AGE = jnp.array([0.02507071,0.06970037,0.11433003,0.025,0.025,0.16546965,1])
Adenovirus_p_time_to_obs = jnp.genfromtxt("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',',dtype=jnp.float64)
Adenovirus_incidence = pd.read_csv("Data/Processed/KPSC_Adenovirus_incidence_age_daily.csv",index_col=0)
from Parameters.Parainfluenza3 import *
# Parameters from differential evolution
Parainfluenza3_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': jnp.array([0.        , 0.00385729, 0.        ]), 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': jnp.array([1.        , 0.17012103, 0.03917217]), 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': jnp.array([0.01535462, 0.00186418, 0.00141742]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.01, 'BETA': 0.5484744900640051, 'SEASONALITY': 0.04569695421592862, 'OFFSET': 0.4600220832530794,
'contact': contact}
Parainfluenza3_OBS_AGE = jnp.array([0.025,0.025,0.10121682,0.025,0.025,0.13514337,1])
Parainfluenza3_p_time_to_obs = jnp.genfromtxt("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',',dtype=jnp.float64)
Parainfluenza3_incidence = pd.read_csv("Data/Processed/KPSC_Parainfluenza3_incidence_age_daily.csv",index_col=0)

# fit with wrong constraints, but looks ok when plotted
from Parameters.Metapneumovirus import *
# Parameters from differential evolution
Metapneumovirus_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': jnp.array([0.        , 0.00591676, 0.        ]), 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': jnp.array([1.        , 0.90218669, 0.64520669]), 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': jnp.array([0.03  , 0.015 , 0.0075]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.01, 'BETA': 0.10162683437577735, 'SEASONALITY': 0.07748324448336691, 'OFFSET': 0.24423167213822972,
'contact': contact}
Metapneumovirus_OBS_AGE = jnp.array([0.00950089,0.09476558,0.06743696,0.00351061,0.00278353,0.01220659,0.08555915])
Metapneumovirus_p_time_to_obs = jnp.genfromtxt("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',',dtype=jnp.float64)
Metapneumovirus_incidence = pd.read_csv("Data/Processed/KPSC_Metapneumovirus_incidence_age_daily.csv",index_col=0)
from Parameters.InfluenzaB import *
# Parameters from differential evolution
InfluenzaB_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': jnp.array([0.        , 0.00921108, 0.        ]), 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': jnp.array([1.        , 0.93981811, 0.54689671]), 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': jnp.array([0.03, 0.03, 0.03]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': flu_rate, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.01, 'BETA': 0.09859329005854467, 'SEASONALITY': 0.14362853724915037, 'OFFSET': 0.1752366022322981,
'contact': contact}
InfluenzaB_OBS_AGE = jnp.array([0.01513805,0.04846613,0.02036943,0.00401086,0.00387421,0.01058703,0.09288545])
InfluenzaB_p_time_to_obs = jnp.genfromtxt("Data/Processed/Influenza_B_incubation_admittance_distribution.csv",delimiter=',',dtype=jnp.float64)
InfluenzaB_incidence = pd.read_csv("Data/Processed/KPSC_Influenza_B_incidence_age_daily.csv",index_col=0)

## Initial conditions
STATE0 = jnp.zeros((2*N_S+2)*NAG)
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

bounds = jnp.array([bounds_dict[key] for key in ["DT1","DT2","DT3","F1","F2","F3","F4"]
if key in bounds_dict.keys()])

def likelihood(x):
    # by default, no overdispersion
    overdispersion = False
    print("time: ",time.time()-start)
    print(x)
    if jnp.any(x < 0) or jnp.any(jnp.isnan(x)):
        print("Invalid parameters")
        return 1e10
    Ts = jnp.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[0]*365,date_to_t('2020-03-19')+(x[0]+x[1])*365,date_to_t('2020-03-19')+(x[0]+x[1]+x[2])*365])
    # Fs - element 2 must be bigger than element 1, element 3 must be smaller than element 2, element 4 must be bigger than element 2
    F1 = x[3] # value between 0 and 1 (first lockdown)
    F2 = F1 + x[4] - F1*x[4] # value between x[3] and 1 (inter-lockdown)
    F3 = F2*x[5] # value less than F2 (second lockdown)
    F4 = F2 + x[6] - F2*x[6] # value between F2 and 1 (post-lockdown)
    Fs = jnp.array([1,F1,F2,F3,F4])
    # @jit
    def contact(t,seasonality,offset):
        return cm.piecewise(t,Ts,Fs)*(1+seasonality*jnp.cos(2*jnp.pi*((t-274)/365-offset)))*CONTACT
    RSV_params["contact"] = contact
    InfluenzaA_params["contact"] = contact
    InfluenzaB_params["contact"] = contact
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