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
from SISn_ODEs import two_pathogen_deltas as sis_deltas
from Parameters.census_population import *
from Parameters.times_and_contacts import *

from utils import *
from demography import *
from mobility_and_import import *
from fit_MCMC import SIS_likelihood
N_C = 2

pathogen1, pathogen2, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
lockdown = "FlexStepwise"

# set seed
np.random.seed(seed)

start_date = '2015-07-04'
end_date = '2023-10-01'

EPOCH = pd.to_datetime('1970-01-01')
START = pd.to_datetime(start_date) 
END = pd.to_datetime(end_date)
PERIOD = pd.date_range(start=START, end=END, freq='D')
POINTS = np.array(date_to_t(PERIOD))

params1, p_time_to_obs1, incidence1 = pathogen_parameters(pathogen1, lockdown, CONTACT)
params2, p_time_to_obs2, incidence2 = pathogen_parameters(pathogen2, lockdown, CONTACT)
N_S, NAG = params1["N_S"], params1["NAG"]

# trim incidence so that Date is between START and END
incidence1.index = pd.to_datetime(incidence1.index)
incidence1 = incidence1.loc[START+pd.Timedelta(days=89):END]
incidence2.index = pd.to_datetime(incidence2.index)
incidence2 = incidence2.loc[START+pd.Timedelta(days=89):END]

## Initial conditions
STATE0 = np.zeros(2*(2*N_S+2)*NAG)
STATE0[NAG:2*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.
STATE0[NAG+(2*N_S+2)*NAG:2*NAG+(2*N_S+2)*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG+(2*N_S+2)*NAG:3*NAG+(2*N_S+2)*NAG] = 1 # one individual in each age group that is infected.

bounds_dict = {"WANE1": [0,1e-2], "SEASONALITY1": [0,1], "OFFSET1": [0,1], "BETA1": [0,1], "WANE2": [0,1e-2], "SEASONALITY2": [0,1], "OFFSET2": [0,1], "BETA2": [0,1]}

if "Influenza" in pathogen1:
    bounds_dict["EXTRA_IMMUNITY1"] = [0,1]
    bounds_dict["FIRST_IMMUNITY1"] = [0.1,1]
    bounds_dict["FIRST_DIS_INF_FACTOR1"] = [0,1]
elif pathogen1 == "RSV":
    bounds_dict["S_REL1_1"] = [0.1,1]
    bounds_dict["S_REL2_1"] = [0.1,1]
else:
    bounds_dict["S_REL1_1"] = [0.1,1]
    bounds_dict["S_REL2_1"] = [0.1,1]
    bounds_dict["D_REL1_1"] = [0.1,1]
    bounds_dict["D_REL2_1"] = [0.1,1]
if "Influenza" in pathogen2:
    bounds_dict["EXTRA_IMMUNITY2"] = [0,1]
    bounds_dict["FIRST_IMMUNITY2"] = [0.1,1]
    bounds_dict["FIRST_DIS_INF_FACTOR2"] = [0,1]
elif pathogen2 == "RSV":
    bounds_dict["S_REL1_2"] = [0.1,1]
    bounds_dict["S_REL2_2"] = [0.1,1]
else:
    bounds_dict["S_REL1_2"] = [0.1,1]
    bounds_dict["S_REL2_2"] = [0.1,1]
    bounds_dict["D_REL1_2"] = [0.1,1]
    bounds_dict["D_REL2_2"] = [0.1,1]
bounds_dict["AGE_OBS_1_1"] = [0,0.01]
bounds_dict["AGE_OBS_2_1"] = [0,0.01]
bounds_dict["AGE_OBS_3_1"] = [0,0.01]
bounds_dict["AGE_OBS_4_1"] = [0,0.01]
bounds_dict["AGE_OBS_5_1"] = [0,0.01]
bounds_dict["AGE_OBS_6_1"] = [0,0.01]
bounds_dict["AGE_OBS_7_1"] = [0,0.01]
bounds_dict["AGE_OBS_1_2"] = [0,0.01]
bounds_dict["AGE_OBS_2_2"] = [0,0.01]
bounds_dict["AGE_OBS_3_2"] = [0,0.01]
bounds_dict["AGE_OBS_4_2"] = [0,0.01]
bounds_dict["AGE_OBS_5_2"] = [0,0.01]
bounds_dict["AGE_OBS_6_2"] = [0,0.01]
bounds_dict["AGE_OBS_7_2"] = [0,0.01]
bounds_dict["DT1"] = [0,1]
bounds_dict["DT2"] = [0,1]
bounds_dict["DT3"] = [0,1]
bounds_dict["F1"] = [0,1]
bounds_dict["F2"] = [0,1]
bounds_dict["F3"] = [0,1]
bounds_dict["F4"] = [0,1]
bounds_dict["INTERFERENCE1to2"] = [0,10]
bounds_dict["INTERFERENCE2to1"] = [0,10]

# bounds = np.array([bounds_dict[key] for key in ["WANE","SEASONALITY","OFFSET","BETA","IMPORT_RATE","EXTRA_IMMUNITY","FIRST_IMMUNITY","FIRST_DIS_INF_FACTOR","S_REL1","S_REL2","D_REL1","D_REL2","P_OBS","DT1","DT2","DT3","F1","F2","F3","F4","OVERDISPERSION","P_OBS","AGE_OBS_YOUNG","AGE_OBS_OLD","AGE_OBS_YOUNG_OLD","AGE_OBS_MATERNAL","AGE_OBS_1","AGE_OBS_2","AGE_OBS_3","AGE_OBS_4","AGE_OBS_5","AGE_OBS_6","AGE_OBS_7"]\
# if key in bounds_dict.keys()])

bounds = np.array([bounds_dict[key] for key in ["WANE1","SEASONALITY1","OFFSET1","BETA1","EXTRA_IMMUNITY1","FIRST_IMMUNITY1","FIRST_DIS_INF_FACTOR1","S_REL1_1","S_REL2_1","D_REL1_1","D_REL2_1","DT1","DT2","DT3","F1","F2","F3","F4","AGE_OBS_1_1","AGE_OBS_2_1","AGE_OBS_3_1","AGE_OBS_4_1","AGE_OBS_5_1","AGE_OBS_6_1","AGE_OBS_7_1","WANE2","SEASONALITY2","OFFSET2","BETA2","EXTRA_IMMUNITY2","FIRST_IMMUNITY2","FIRST_DIS_INF_FACTOR2","S_REL1_2","S_REL2_2","D_REL1_2","D_REL2_2","AGE_OBS_1_2","AGE_OBS_2_2","AGE_OBS_3_2","AGE_OBS_4_2","AGE_OBS_5_2","AGE_OBS_6_2","AGE_OBS_7_2","INTERFERENCE1to2","INTERFERENCE2to1"]\
if key in bounds_dict.keys()])

def likelihood(x):
    # by default, no overdispersion
    overdispersion = False
    print("time: ",time.time()-start)
    print(x)
    if np.any(x < 0) or np.any(np.isnan(x)):
        print("Invalid parameters")
        return 1e10
    sim_params1 = params1.copy()
    sim_params2 = params2.copy()
    sim_params1["IMPORT_RATE"] = 0.01
    sim_params2["IMPORT_RATE"] = 0.01
    sim_params1["WANE"] = np.array([0.0,x[0],0.0])
    sim_params1["SEASONALITY"] = x[1]
    sim_params1["OFFSET"] = x[2]
    sim_params1["BETA"] = x[3]
    n = 4
    if "Influenza" in pathogen1:
        srel1, pobsrel1 = constrained_immunity(x[n],x[n+1],x[n+2])
        sim_params1["S_REL"] = srel
        n += 3
    elif pathogen1 == 'RSV':
        sim_params1["S_REL"] = np.array([1,x[n],x[n]*x[n+1]])
        pobsrel1 = np.array([1,0.46,0.31]) # Henderson 1979
        n += 2
    else:
        sim_params1["S_REL"] = np.array([1,x[n],x[n]*x[n+1]])
        pobsrel1 = np.array([1,x[n+2],x[n+2]*x[n+3]])
        n += 4
    sim_params1["P_OBS"] = pobsrel1
    Ts = np.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[n]*365,date_to_t('2020-03-19')+(x[n]+x[n+1])*365,date_to_t('2020-03-19')+(x[n]+x[n+1]+x[n+2])*365])
    # Fs - element 2 must be bigger than element 1, element 3 must be smaller than element 2, element 4 must be bigger than element 2
    F1 = x[n+3] # value between 0 and 1 (first lockdown)
    F2 = F1 + x[n+4] - F1*x[n+4] # value between x[n+3] and 1 (inter-lockdown)
    F3 = F2*x[n+5] # value less than F2 (second lockdown)
    F4 = F2 + x[n+6] - F2*x[n+6] # value between F2 and 1 (post-lockdown)
    Fs = np.array([1,F1,F2,F3,F4])
    n += 7
    @jit
    def contact(t,seasonality,offset):
        return cm.piecewise(t,Ts,Fs)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
    sim_params1["contact"] = contact
    obs_age1 = np.array([x[n],x[n+1],x[n+2],x[n+3],x[n+4],x[n+5],x[n+6]])
    n += 7

    sim_params2["WANE"] = np.array([0.0,x[0],0.0])
    sim_params2["SEASONALITY"] = x[1]
    sim_params2["OFFSET"] = x[2]
    sim_params2["BETA"] = x[3]
    n += 4
    if "Influenza" in pathogen2:
        srel2, pobsrel2 = constrained_immunity(x[n],x[n+1],x[n+2])
        sim_params2["S_REL"] = srel
        n += 3
    elif pathogen2 == 'RSV':
        sim_params2["S_REL"] = np.array([1,x[n],x[n]*x[n+1]])
        pobsrel2 = np.array([1,0.46,0.31])
        n += 2
    else:
        sim_params2["S_REL"] = np.array([1,x[n],x[n]*x[n+1]])
        pobsrel2 = np.array([1,x[n+2],x[n+2]*x[n+3]])
        n += 4
    sim_params2["P_OBS"] = pobsrel2
    obs_age2 = np.array([x[n],x[n+1],x[n+2],x[n+3],x[n+4],x[n+5],x[n+6]])
    n += 7
    interference1to2 = x[n]
    interference2to1 = x[n+1]

    try:
        arguments = [NAG, N_S, sim_params1["AGING_RATE"], sim_params1["BIRTH_RATE"], sim_params1["arrivals"], sim_params1["contact"], np.array([interference2to1,interference1to2]), \
            sim_params1["WANE"], sim_params1["REC_UP"], sim_params1["REC_SAME"], sim_params1["S_REL"], sim_params1["S_AGE"], sim_params1["I_REL"], sim_params1["P_OBS"], sim_params1["birth_vax"], sim_params1["all_vax"], sim_params1["S_VAX"], sim_params1["ACOV"], sim_params1["BCOV"], sim_params1["regional_positivity"], sim_params1["IMPORT_RATE"], sim_params1["BETA"], sim_params1["SEASONALITY"], sim_params1["OFFSET"], \
            sim_params2["WANE"], sim_params2["REC_UP"], sim_params2["REC_SAME"], sim_params2["S_REL"], sim_params2["S_AGE"], sim_params2["I_REL"], sim_params2["P_OBS"], sim_params2["birth_vax"], sim_params2["all_vax"], sim_params2["S_VAX"], sim_params2["ACOV"], sim_params2["BCOV"], sim_params2["regional_positivity"], sim_params2["IMPORT_RATE"], sim_params2["BETA"], sim_params2["SEASONALITY"], sim_params2["OFFSET"]]
        result = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=arguments,t_eval=POINTS,method='RK45')
        result1 = result.copy()
        result1["y"] = result.y[:(N_C*N_S+1+(3-N_C))*NAG]
        result2 = result.copy()
        result2["y"] = result.y[(N_C*N_S+1+(3-N_C))*NAG:]
        lh1 = -SIS_likelihood(incidence1,sim_params1,POINTS,STATE0,obs_age1,p_time_to_obs1,age=True,incidence=True,overdispersion=overdispersion, result=result1)
        lh2 = -SIS_likelihood(incidence2,sim_params2,POINTS,STATE0,obs_age2,p_time_to_obs2,age=True,incidence=True,overdispersion=overdispersion, result=result2)
    except:
        print("Error")
        return 1e10
    print("neg log likelihoods:", lh1, lh2)
    return lh1 + lh2

with open("Data/Processed/results250516/DE_opt_"+pathogen1+"FlexStepwisesetimportflexage2505163.pickle","rb") as f:
    opt1 = pickle.load(f)
with open("Data/Processed/results250516/DE_opt_"+pathogen2+"FlexStepwisesetimportflexage2505163.pickle","rb") as f:
    opt2 = pickle.load(f)
x1 = opt1.x
x2 = opt2.x\
# remove index -14 to -7 from x2
ln = len(x2)
x2 = np.delete(x2, np.s_[(ln-14):(ln-7)])
x0 = np.concatenate((x1,x2,np.ones(2)))

start = time.time()
if __name__ == '__main__':
    opt = sp.optimize.minimize(likelihood, x0=x0)
    print(opt.x[-2:])
    with open("Data/Processed/interference_test_opt.pickle","wb") as f:
        pickle.dump(opt, f)