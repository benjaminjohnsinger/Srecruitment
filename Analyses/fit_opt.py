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

from utils import *
from demography import *
from mobility_and_import import *
from clustering import *
from sim_grid import *
from plotting import *
from fit_MCMC import *


pathogen, seed, lockdown, start, end, desize, max_mutation, recombination = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])

# set seed
np.random.seed(seed)

# check if start is in date format with regex
if not re.match(r'\d{4}-\d{2}-\d{2}',start):
    print("Start date not valid, assigning to default (2015-07-04).")
    start = '2015-07-04'
if not re.match(r'\d{4}-\d{2}-\d{2}',end):
    print("End date not valid, assigning to default (2023-10-01).")
    end = '2023-10-01'

EPOCH = pd.to_datetime('1970-01-01')
START = pd.to_datetime(start) 
END = pd.to_datetime(end)
PERIOD = pd.date_range(start=START, end=END, freq='D')
POINTS = np.array(date_to_t(PERIOD))

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

# trim incidence so that Date is between START and END
incidence.index = pd.to_datetime(incidence.index)
incidence = incidence.loc[START+pd.Timedelta(days=89):END]

## Initial conditions
STATE0 = np.zeros((2*N_S+2)*NAG)
STATE0[NAG:2*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

if lockdown == 'Mobility':
    @jit
    def contact(t,seasonality,offset):
        return cm.google_prestige_work(t)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
    params["contact"] = contact

if lockdown == 'YoungEarly':
    @jit
    def contact(t,seasonality,offset):
        cont = cm.google_prestige_work(t)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
        if t > date_to_t('2021-03-16'): # date where majority of schoools returned to in-person according to burbio
            cont[0:4] = (1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT[0:4]
        return cont
    params["contact"] = contact

if pathogen == 'RSV':
    if (lockdown == 'Stepwise') or (lockdown == 'Mobility') or (lockdown == 'YoungEarly'):
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
        [0,1]]) # AGE_OBS - young_old
        def likelihood(x):
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
            sim_params["IMPORT_RATE"] = x[4]
            sim_params["S_REL"] = np.array([1,x[5],x[5]*x[6]])
            sim_params["P_OBS"] = x[9]*np.array([1,x[7],x[7]*x[8]])
            obs_age = age_detection(NAG,x[10],x[11],x[12])
            try:
                lh = -SIS_likelihood(incidence,sim_params,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)
            except:
                print("Error")
                return 1e10
            print("neg log likelihood: ",lh)
            return lh
    elif lockdown == 'FixedStepwise':
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
        [0,1], # F1 - first lockdown relative contact rate
        [0,1], # F2 - inter-lockdown relative contact rate
        [0,1], # F3 - second lockdown relative contact rate
        [0,1], # F4 - post-lockdown relative contact rate
        [0,1], # AGE_OBS - young_immunity
        [0,1], # AGE_OBS - old_immunity
        [0,1]]) # AGE_OBS - young_old
        def likelihood(x):
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
            sim_params["IMPORT_RATE"] = x[4]
            sim_params["S_REL"] = np.array([1,x[5],x[5]*x[6]])
            sim_params["P_OBS"] = x[9]*np.array([1,x[7],x[7]*x[8]])
            Fs = np.array([1,x[10],x[11],x[12],x[13]])
            @jit
            def contact(t,seasonality,offset):
                return cm.piecewise(t,Ts,Fs)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
            sim_params["contact"] = contact
            obs_age = age_detection(NAG,x[14],x[15],x[16])
            try:
                lh = -SIS_likelihood(incidence,sim_params,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)
            except:
                print("Error")
                return 1e10
            print("neg log likelihood: ",lh)
            return lh
    elif lockdown == 'MovingStepwise':
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
        [0,1], # DT1 - first lockdown duration in years
        [0,1], # DT2 - inter-lockdown duration in years
        [0,1], # DT3 - second lockdown duration in years
        [0,1], # AGE_OBS - young_immunity
        [0,1], # AGE_OBS - old_immunity
        [0,1]])
        def likelihood(x):
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
            sim_params["IMPORT_RATE"] = x[4]
            sim_params["S_REL"] = np.array([1,x[5],x[5]*x[6]])
            sim_params["P_OBS"] = x[9]*np.array([1,x[7],x[7]*x[8]])
            Ts = np.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[10]*365,date_to_t('2020-03-19')+(x[10]+x[11])*365,date_to_t('2020-03-19')+(x[10]+x[11]+x[12])*365])
            @jit
            def contact(t,seasonality,offset):
                return cm.piecewise(t,Ts,Fs)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
            sim_params["contact"] = contact
            obs_age = age_detection(NAG,x[13],x[14],x[15])
            try:
                lh = -SIS_likelihood(incidence,sim_params,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)
            except:
                print("Error")
                return 1e10
            print("neg log likelihood: ",lh)
            return lh
    elif lockdown == 'FlexStepwise':
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
        [0,1], # DT1 - first lockdown duration in years
        [0,1], # DT2 - inter-lockdown duration in years
        [0,1], # DT3 - second lockdown duration in years
        [0,1], # F1 - lockdown relative contact rate
        [0,1], # F2 - inter-lockdown relative contact rate
        [0,1], # F3 - post-lockdown relative contact rate
        [0,1], # AGE_OBS - young_immunity
        [0,1], # AGE_OBS - old_immunity
        [0,1]])
        def likelihood(x):
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
            sim_params["IMPORT_RATE"] = x[4]
            sim_params["S_REL"] = np.array([1,x[5],x[5]*x[6]])
            sim_params["P_OBS"] = x[9]*np.array([1,x[7],x[7]*x[8]])
            Ts = np.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[10]*365,date_to_t('2020-03-19')+(x[10]+x[11])*365,date_to_t('2020-03-19')+(x[10]+x[11]+x[12])*365])
            Fs = np.array([1,x[13],x[14],x[13],x[15]])
            @jit
            def contact(t,seasonality,offset):
                return cm.piecewise(t,Ts,Fs)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
            sim_params["contact"] = contact
            obs_age = age_detection(NAG,x[16],x[17],x[18])
            try:
                lh = -SIS_likelihood(incidence,sim_params,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)
            except:
                print("Error")
                return 1e10
            print("neg log likelihood: ",lh)
            return lh
else:
    bounds = np.array([[0,1e-2], # WANE
    [0,1], # SEASONALITY
    [0,1], # OFFSET
    [0,1], # BETA
    [0,1e-10], # IMPORT_RATE
    [0,1], # EXTRA_IMMUNITY - immunity after second infection above minimum
    [0,1], # FIRST_IMMUNITY - immunity after first infection above minimum
    [0,1], # FIRST_DIS_INF_FACTOR - relative infection and disease immunity after first infection
    [0,0.1], # P_OBS
    [0,1], # AGE_OBS - young_immunity
    [0,1], # AGE_OBS - old_immunity
    [0,1]]) # AGE_OBS - young_old
    def likelihood(x):
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
        sim_params["IMPORT_RATE"] = x[4]
        srel, pobsrel = constrained_immunity(x[5],x[6],x[7])
        sim_params["S_REL"] = srel
        sim_params["P_OBS"] = x[8]*pobsrel
        obs_age = age_detection(NAG,x[9],x[10],x[11])
        try:
            lh = -SIS_likelihood(incidence,sim_params,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)
        except:
            print("Error")
            return 1e10
        print("neg log likelihood: ",lh)
        return lh

start = time.time()
if __name__ == '__main__':
    opt = sp.optimize.differential_evolution(likelihood,bounds,popsize=desize,mutation=(0.5,max_mutation),recombination=recombination,
    workers=int(os.getenv('SLURM_CPUS_ON_NODE')))

    with open("Data/Processed/DE_opt_"+pathogen+"_"+str(os.getenv('SLURM_JOB_NAME'))+str(seed)+".pickle","wb") as f:
        pickle.dump(opt,f)