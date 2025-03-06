## Simultaneous optimization of pathogens with shared lockdown model
## BJS March 2025

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

from utils import *
from demography import *
from mobility_and_import import *
from clustering import *
from sim_grid import *
from plotting import *
from fit_MCMC import *


seed, lockdown, desize, max_mutation, recombination = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])

# set seed
np.random.seed(seed)


from Parameters.RSV import *
# Parameters for the ODE
params_RSV = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
'contact': contact}
p_time_to_obs_RSV = np.genfromtxt("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
incidence_RSV = pd.read_csv("Data/Processed/KPSC_RSV_incidence_age_daily.csv",index_col=0)

from Parameters.InfluenzaA import *
params_InfluenzaA = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': flu_rate, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
'contact': contact}
p_time_to_obs_InfluenzaA = np.genfromtxt("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
incidence_InfluenzaA = pd.read_csv("Data/Processed/KPSC_Influenza_A_incidence_age_daily.csv",index_col=0)

from Parameters.InfluenzaB import *
params_InfluenzaB = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': flu_rate, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
'contact': contact}
p_time_to_obs_InfluenzaB = np.genfromtxt("Data/Processed/Influenza_B_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
incidence_InfluenzaB = pd.read_csv("Data/Processed/KPSC_Influenza_B_incidence_age_daily.csv",index_col=0)

## Initial conditions
STATE0 = np.zeros((2*N_S+2)*NAG)
STATE0[NAG:2*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

if (lockdown == 'FlexStepwise'):
    bounds = np.array([[0,1e-2], # WANE
    [0,1], # SEASONALITY
    [0,1], # OFFSET
    [0,1], # BETA
    [0,1e-9], # IMPORT_RATE
    [0,1], # S_REL1 - relative susceptibility to infection after first infection
    [0,1], # S_REL2/S_REL1 - relative susceptibility to infection after second infection
    [0,1], # D_REL1 - relative susceptibility to disease after first infection
    [0,1], # D_REL2/D_REL1 - relative susceptibility to disease after second infection
    [0,1], # P_OBS
    [0,1], # AGE_OBS - young_immunity
    [0,1], # AGE_OBS - old_immunity
    [0,1]]) # AGE_OBS - young - old immunity
    bounds = np.tile(bounds,(3,1))
    lockdown_bounds = np.array([[0,1], # DT1 - first lockdown duration in years
    [0,1], # DT2 - inter-lockdown duration in years
    [0,1], # DT3 - second lockdown duration in years
    [0,1], # F1 - first lockdown relative contact rate
    [0,1], # F2 - inter-lockdown relative contact rate
    [0,1], # F3 - second lockdown relative contact rate
    [0,1]]) # F4 - post-lockdown relative contact rate
    bounds = np.concatenate((bounds,lockdown_bounds),axis=0)
    def likelihood(x):
        print("time: ",time.time()-start)
        print(x)
        if np.any(x < 0) or np.any(np.isnan(x)):
            print("Invalid parameters")
            return 1e10
        sim_params_RSV = params_RSV.copy()
        sim_params_RSV["WANE"] = np.array([0.0,x[0],0.0])
        sim_params_RSV["SEASONALITY"] = x[1]
        sim_params_RSV["OFFSET"] = x[2]
        sim_params_RSV["BETA"] = x[3]
        sim_params_RSV["IMPORT_RATE"] = x[4]
        sim_params_RSV["S_REL"] = np.array([1,x[5],x[5]*x[6]])
        sim_params_RSV["P_OBS"] = x[9]*np.array([1,x[7],x[7]*x[8]])
        age_obs_RSV = age_detection(NAG,x[10],x[11],x[12])
        sim_params_InfluenzaA = params_InfluenzaA.copy()
        sim_params_InfluenzaA["WANE"] = np.array([0.0,x[13],0.0])
        sim_params_InfluenzaA["SEASONALITY"] = x[14]
        sim_params_InfluenzaA["OFFSET"] = x[15]
        sim_params_InfluenzaA["BETA"] = x[16]
        sim_params_InfluenzaA["IMPORT_RATE"] = x[17]
        sim_params_InfluenzaA["S_REL"] = np.array([1,x[18],x[18]*x[19]])
        sim_params_InfluenzaA["P_OBS"] = x[22]*np.array([1,x[20],x[20]*x[21]])
        age_obs_InfluenzaA = age_detection(NAG,x[23],x[24],x[25])
        sim_params_InfluenzaB = params_InfluenzaB.copy()
        sim_params_InfluenzaB["WANE"] = np.array([0.0,x[26],0.0])
        sim_params_InfluenzaB["SEASONALITY"] = x[27]
        sim_params_InfluenzaB["OFFSET"] = x[28]
        sim_params_InfluenzaB["BETA"] = x[29]
        sim_params_InfluenzaB["IMPORT_RATE"] = x[30]
        sim_params_InfluenzaB["S_REL"] = np.array([1,x[31],x[31]*x[32]])
        sim_params_InfluenzaB["P_OBS"] = x[35]*np.array([1,x[33],x[33]*x[34]])
        age_obs_InfluenzaB = age_detection(NAG,x[36],x[37],x[38])

        Ts = np.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[39]*365,date_to_t('2020-03-19')+(x[39]+x[40])*365,date_to_t('2020-03-19')+(x[39]+x[40]+x[41])*365])
        Fs = np.array([1,x[42],x[43],x[44],x[45]])
        @jit
        def contact(t,seasonality,offset):
            return cm.piecewise(t,Ts,Fs)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
        sim_params_RSV["contact"] = sim_params_InfluenzaA["contact"] = sim_params_InfluenzaB["contact"] = contact
        
        ls = 0
        for pathogen in ["RSV","InfluenzaA","InfluenzaB"]:
            if pathogen == "RSV":
                sim_params, obs_age, incidence, p_time_to_obs = sim_params_RSV, age_obs_RSV, incidence_RSV, p_time_to_obs_RSV
            elif pathogen == "InfluenzaA":
                sim_params, obs_age, incidence, p_time_to_obs = sim_params_InfluenzaA, age_obs_InfluenzaA, incidence_InfluenzaA, p_time_to_obs_InfluenzaA
            elif pathogen == "InfluenzaB":
                sim_params, obs_age, incidence, p_time_to_obs = sim_params_InfluenzaB, age_obs_InfluenzaB, incidence_InfluenzaB, p_time_to_obs_InfluenzaB
            try:
                lh = -SIS_likelihood(incidence,sim_params,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)
            except:
                print("Error in likelihood calculation for",pathogen)
                return 1e10
            print(pathogen, "neg log likelihood:",lh)
            ls += lh
        print("overall neg log likelihood:",ls)
        return ls

start = time.time()
if __name__ == '__main__':
    opt = sp.optimize.differential_evolution(likelihood,bounds,popsize=desize,mutation=(0.5,max_mutation),recombination=recombination,
    workers=int(os.getenv('SLURM_CPUS_ON_NODE')))

    with open("Data/Processed/DE_simultaneous_"+str(os.getenv('SLURM_JOB_NAME'))+str(seed)+".pickle","wb") as f:
        pickle.dump(opt,f)