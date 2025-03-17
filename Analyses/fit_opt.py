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


pathogen, seed, lockdown, option1, option2, desize, max_mutation, recombination = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])

# set seed
np.random.seed(seed)

start_date = '2015-07-04'
end_date = '2023-10-01'
# check if option1 is in date format with regex
if re.match(r'\d{4}-\d{2}-\d{2}',option1):
    start_date = option1
if re.match(r'\d{4}-\d{2}-\d{2}',option2):
    end_date = option2

EPOCH = pd.to_datetime('1970-01-01')
START = pd.to_datetime(start_date) 
END = pd.to_datetime(end_date)
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
        if t > 18702: # 2021-03-16 where majority of schoools returned to in-person according to burbio
            cont[0:4] = (1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT[0:4]
        return cont
    params["contact"] = contact

bounds_dict = {"WANE": [0,1e-2], "SEASONALITY": [0,1], "OFFSET": [0,1], "BETA": [0,1], "P_OBS": [0,0.1], "AGE_OBS_YOUNG": [0,1], "AGE_OBS_OLD": [0,1], "AGE_OBS_YOUNG_OLD": [0,1]}

if option1 == "nb":
    bounds_dict["OVERDISPERSION"] = [-5,10]
if option1 != "ni":
    bounds_dict["IMPORT_RATE"] = [0,0.1]
if (pathogen == "RSV") or (option2 == "nr"):
    bounds_dict["S_REL1"] = [0.1,1]
    bounds_dict["S_REL2"] = [0.1,1]
    bounds_dict["D_REL1"] = [0.1,1]
    bounds_dict["D_REL2"] = [0.1,1]
else:
    bounds_dict["EXTRA_IMMUNITY"] = [0,1]
    bounds_dict["FIRST_IMMUNITY"] = [0,1]
    bounds_dict["FIRST_DIS_INF_FACTOR"] = [0,1]
if lockdown == "FlexStepwise":
    bounds_dict["DT1"] = [0,1]
    bounds_dict["DT2"] = [0,1]
    bounds_dict["DT3"] = [0,1]
    bounds_dict["F1"] = [0,1]
    bounds_dict["F2"] = [0,1]
    bounds_dict["F3"] = [0,1]
    bounds_dict["F4"] = [0,1]

bounds = np.array([bounds_dict[key] for key in ["WANE","SEASONALITY","OFFSET","BETA","IMPORT_RATE","EXTRA_IMMUNITY","FIRST_IMMUNITY","FIRST_DIS_INF_FACTOR","S_REL1","S_REL2","D_REL1","D_REL2","P_OBS","DT1","DT2","DT3","F1","F2","F3","F4","OVERDISPERSION","AGE_OBS_YOUNG","AGE_OBS_OLD","AGE_OBS_YOUNG_OLD"]
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
    if option1 == 'ni':
        sim_params["IMPORT_RATE"] = 0
    else:
        sim_params["IMPORT_RATE"] = x[n]
        n += 1
    if (pathogen == 'RSV') or (option2 == 'nr'):
        sim_params["S_REL"] = np.array([1,x[n],x[n]*x[n+1]])
        pobsrel = np.array([1,x[n+2],x[n+2]*x[n+3]])
        n += 4
    else:
        srel, pobsrel = constrained_immunity(x[n],x[n+1],x[n+2])
        sim_params["S_REL"] = srel
        n += 3
    sim_params["P_OBS"] = x[n]*pobsrel
    n += 1
    if lockdown == 'FlexStepwise':
        Ts = np.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[n]*365,date_to_t('2020-03-19')+(x[n]+x[n+1])*365,date_to_t('2020-03-19')+(x[n]+x[n+1]+x[n+2])*365])
        # Fs - element 2 must be bigger than element 1, element 3 must be smaller than element 2, element 4 must be bigger than element 2
        a = x[n+3]+x[n+4]-x[n+3]*x[n+4] # value between x[n+3] and 1
        Fs = np.array([1,x[n+3],a,a*x[n+5],a+x[n+6]+a*x[n+6]])
        @jit
        def contact(t,seasonality,offset):
            return cm.piecewise(t,Ts,Fs)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
        sim_params["contact"] = contact
        n += 7
    if option1 == 'nb':
        overdispersion = np.exp(x[n])
        n += 1
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
    opt = sp.optimize.differential_evolution(likelihood,bounds,popsize=desize,mutation=(0.5,max_mutation),recombination=recombination,
    workers = 4)
    # workers=int(os.getenv('SLURM_CPUS_ON_NODE')))

    with open("Data/Processed/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","wb") as f:
        pickle.dump(opt,f)