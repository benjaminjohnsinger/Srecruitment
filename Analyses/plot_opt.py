## BJS March 2025
## Plotting results of fitting

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

pathogen, seed, lockdown, option1, option2 = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]

with open("Data/Processed/results"+str(seed)+"/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","rb") as f:
    opt = pickle.load(f)
# check that optimization converged
if opt.success:
    print("Optimization converged")
else:
    print("Optimization did not converge")
    print(opt.message)
    print(opt.x)
    sys.exit()
x = opt.x

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

params["WANE"] = np.array([0.0,x[0],0.0])
params["SEASONALITY"] = x[1]
params["OFFSET"] = x[2]
params["BETA"] = x[3]
n = 4
if option1 != 'ni':
    params["IMPORT_RATE"] = x[n]
    n += 1
if (pathogen == 'RSV') or (option2 == 'nr'):
    params["S_REL"] = np.array([1,x[n],x[n]*x[n+1]])
    pobsrel = np.array([1,x[n+2],x[n+2]*x[n+3]])
    n += 4
else:
    srel, pobsrel = constrained_immunity(x[n],x[n+1],x[n+2])
    params["S_REL"] = srel
    n += 3
params["P_OBS"] = x[n]*pobsrel
n += 1
if lockdown == 'FlexStepwise':
    Ts = np.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[n]*365,date_to_t('2020-03-19')+(x[n]+x[n+1])*365,date_to_t('2020-03-19')+(x[n]+x[n+1]+x[n+2])*365])
    a = x[n+3]+x[n+4]-x[n+3]*x[n+4] # value between x[n+3] and 1
    Fs = np.array([1,x[n+3],a,a*x[n+5],a+x[n+6]+a*x[n+6]])
    @jit
    def contact(t,seasonality,offset):
        return cm.piecewise(t,Ts,Fs)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
    params["contact"] = contact
    n += 7
if option1 == 'nb':
    overdispersion = np.exp(x[n]-5)
    print("Overdispersion",overdispersion)
    n += 1
OBS_AGE = age_detection(NAG,x[n],x[n+1],x[n+2])

print(params)
print("OBS_AGE",OBS_AGE)
if lockdown == 'FlexStepwise':
    print("Ts",[t_to_date(t) for t in Ts])
    print("Fs",Fs)

np.set_printoptions(threshold=np.inf)

result = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params.values(),t_eval=POINTS,method='RK45')
obs = observations(result,params,OBS_AGE,incidence=False,time_conversion=30.44)

WANE = params["WANE"]
SEASONALITY = params["SEASONALITY"]
OFFSET = params["OFFSET"]
BETA = params["BETA"]
S_REL = params["S_REL"]
P_OBS = params["P_OBS"]
IMPORT_RATE = params["IMPORT_RATE"]
contact = params["contact"]

# get R(t)
R0s = np.zeros(len(result.t))
Rts = np.zeros(len(result.t))
contact_ratios = np.zeros(len(result.t))
for idx in range(len(result.t)):
    pop_size = np.sum(result.y[:,idx],dtype=np.float64)
    age_pops = np.array([np.sum(result.y[range(i_age,(2*N_S+1)*NAG,NAG),idx],axis=0) for i_age in range(NAG)])
    contact_t = contact(result.t[idx],SEASONALITY,OFFSET)
    infectious_contact_equal = np.dot(contact_t,np.sum(np.array([result.y[j,idx] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=np.float64).reshape((N_S,NAG))*I_REL,axis=0))/np.sum(np.array([result.y[j,idx] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=np.float64))
    infectious_contact = np.dot(contact_t,np.sum(np.array([result.y[j,idx] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=np.float64).reshape((N_S,NAG))*I_REL,axis=0))/pop_size
    import_contact = IMPORT_RATE*regional_positivity(result.t[idx])*arrivals(result.t[idx])*np.dot(contact_t,age_pops)/pop_size
    contact_ratios[idx] = np.sum(np.repeat(S_REL,NAG*N_C)*np.tile(S_AGE,N_S*N_C)*BETA*np.tile(import_contact,N_S*N_C)*np.repeat(np.tile(np.array([0,1]+[0]*(N_C-2)),N_S),NAG)*np.array([np.tile(result.y[(2*i+1)*NAG:(2*i+2)*NAG,idx],N_C) for i in range(N_S)]).flatten())/np.sum(np.repeat(S_REL,NAG*N_C)*np.tile(S_AGE,N_S*N_C)*BETA*np.tile(infectious_contact,N_S*N_C)*np.repeat(np.tile(np.array([0,1]+[0]*(N_C-2)),N_S),NAG)*np.array([np.tile(result.y[(2*i+1)*NAG:(2*i+2)*NAG,idx],N_C) for i in range(N_S)]).flatten())
    R0s[idx] = BETA*np.sum(infectious_contact_equal)/REC_UP[0]
    Rts[idx] = (1/pop_size)*(1/REC_UP[0])*np.sum(np.repeat(S_REL,NAG*N_C)*np.tile(S_AGE,N_S*N_C)*BETA*np.tile(infectious_contact_equal,N_S*N_C)*np.repeat(np.tile(np.array([1,0]+[0]*(N_C-2)),N_S),NAG)*np.array([np.tile(result.y[(2*i+1)*NAG:(2*i+2)*NAG,idx],N_C) for i in range(N_S)]).flatten())
print("R0:",np.median(R0s),"("+str(np.min(R0s))+"–"+str(np.max(R0s))+")")
print("Rt:",np.median(Rts),"("+str(np.min(Rts))+"–"+str(np.max(Rts))+")")
print("Ratio of import-caused cases to internal transmission:",np.median(contact_ratios),"("+str(np.min(contact_ratios))+"–"+str(np.max(contact_ratios))+")")

# fig = plt.figure(figsize=(6.5,6.5))
# ax1 = fig.add_subplot(3,1,1)
# ax2 = fig.add_subplot(3,1,2, sharex=ax1, sharey=ax1)
# ax3 = fig.add_subplot(3,1,3, sharex=ax1)
# ax = [ax1,ax2,ax3]
# kpsc_positive_test_plot(ax[0],pathogen=pathogen,AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, incidence=True, legend=False,aggregation="Month")
# ax[0].set_xlabel("")
# ax[0].set_title("Observed incidence")
# ax[0].set_ylabel("Incidence per 10k")
# mx = lockdown_incidence_plot(ax[1],STATE0,params,OBS_AGE,PERIOD,POINTS,date_to_t('2020-03-19'),365,result=result,obs=obs,label="Simulation",by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs)
# lockdown_incidence_format(ax[1],date_to_t('2020-03-19'),365,mx,year_window=2)
# ax[1].set_title("Simulated incidence")
# ax[1].set_xlabel("")
# ax[1].set_ylabel("Incidence per 10k")
# lockdown_susceptibility_plot(ax[2],STATE0,params,PERIOD,POINTS,date_to_t('2020-03-19'),result=result,relative=False)
# lockdown_susceptibility_format(ax[2],date_to_t('2020-03-19'),365,year_window=2,ymax=None,ymin=None)
# ax[2].set_xlabel("Date")
# plt.tight_layout()
# plt.savefig("Figures/DE_"+pathogen+lockdown+option1+option2+str(seed)+"_test.png",dpi=300)