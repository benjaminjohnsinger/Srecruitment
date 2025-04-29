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
print(x)

params, p_time_to_obs, incidence = pathogen_parameters(pathogen, lockdown, CONTACT)

N_S, NAG = params["N_S"], params["NAG"]

## Initial conditions
STATE0 = np.zeros((2*N_S+2)*NAG)
STATE0[NAG:2*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

params["WANE"] = np.array([0.0,x[0],0.0])
params["SEASONALITY"] = x[1]
params["OFFSET"] = x[2]
params["BETA"] = x[3]
n = 4
if option1 == 'ni':
    params["IMPORT_RATE"] = 0
elif option1 == 'setimport':
    params["IMPORT_RATE"] = 0.01
else:
    params["IMPORT_RATE"] = x[n]
    n += 1
if ("Influenza" in pathogen) and (option2 != 'nr'):
    srel, pobsrel = constrained_immunity(x[n],x[n+1],x[n+2])
    params["S_REL"] = srel
    n += 3
elif pathogen == 'RSV':
    params["S_REL"] = np.array([1,x[n],x[n]*x[n+1]])
    pobsrel = np.array([1,0.46,0.31]) # Henderson 1979
    n += 2
else:
    params["S_REL"] = np.array([1,x[n],x[n]*x[n+1]])
    pobsrel = np.array([1,x[n+2],x[n+2]*x[n+3]])
    n += 4
params["P_OBS"] = x[n]*pobsrel
n += 1
if lockdown == 'FlexStepwise':
    Ts = np.array([date_to_t('1970-01-01'),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[n]*365,date_to_t('2020-03-19')+(x[n]+x[n+1])*365,date_to_t('2020-03-19')+(x[n]+x[n+1]+x[n+2])*365])
    F1 = x[n+3] # value between 0 and 1 (first lockdown)
    F2 = F1 + x[n+4] - F1*x[n+4] # value between x[n+3] and 1 (inter-lockdown)
    F3 = F2*x[n+5] # value less than F2 (second lockdown)
    F4 = F2 + x[n+6] - F2*x[n+6] # value between F2 and 1 (post-lockdown)
    Fs = np.array([1,F1,F2,F3,F4])
    @jit
    def contact(t,seasonality,offset):
        return cm.piecewise(t,Ts,Fs)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
    params["contact"] = contact
    n += 7
if option1 == 'nb':
    overdispersion = np.exp(x[n]-5)
    print("Overdispersion",overdispersion)
    n += 1
if option2 == 'flexage':
    # OBS_AGE = np.zeros((7))
    # remaining = 1.0
    # for i in range(1,7):
    #     allocation = x[n+i-1]*remaining
    #     OBS_AGE[i] = allocation
    #     remaining -= allocation
    # OBS_AGE[0] = remaining
    # OBS_AGE = OBS_AGE/np.max(OBS_AGE)
    OBS_AGE = np.array([x[n],x[n+1],x[n+2],x[n+3],x[n+4],x[n+5],x[n+6]])
elif option2 == 'maternal':
    OBS_AGE = age_detection(NAG,x[n],x[n+1],x[n+2],x[n+3],min_obs=0.025)
else:
    OBS_AGE = age_detection(NAG,x[n],x[n+1],x[n+2])

print(params)
print("OBS_AGE",OBS_AGE)
if lockdown == 'FlexStepwise':
    print("Ts",[t_to_date(t) for t in Ts])
    print("Fs",Fs)

result = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params.values(),t_eval=POINTS,method='RK45')
obs = observations(result,params,OBS_AGE,incidence=False,time_conversion=30.44)

WANE = params["WANE"]
SEASONALITY = params["SEASONALITY"]
OFFSET = params["OFFSET"]
BETA = params["BETA"]
REC_UP = params["REC_UP"]
REC_SAME = params["REC_SAME"]
S_REL = params["S_REL"]
S_AGE = params["S_AGE"]
I_REL = params["I_REL"]
P_OBS = params["P_OBS"]
IMPORT_RATE = params["IMPORT_RATE"]
contact = params["contact"]
regional_positivity = params["regional_positivity"]

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

fig = plt.figure(figsize=(13.3,7.5))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2, sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(3,1,3, sharex=ax1)
ax = [ax1,ax2,ax3]
kpsc_positive_test_plot(ax[0],pathogen=pathogen,AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, incidence=True, legend=False,aggregation="Month")
ax[0].set_xlabel("")
pnamedict = {"RSV":"RSV","InfluenzaA":"Influenza A","InfluenzaB":"Influenza B","Parainfluenza3":"Parainfluenza 3","Adenovirus":"Adenovirus","Metapneumovirus":"Metapneumovirus"}
ax[0].set_title("Observed incidence of "+pnamedict[pathogen])
ax[0].set_ylabel("Incidence per 10k")
# legend
ax[0].legend(frameon=False)
mx = lockdown_incidence_plot(ax[1],STATE0,params,OBS_AGE,PERIOD,POINTS,date_to_t('2020-03-19'),365,result=result,obs=obs,label="Simulation",by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs)
lockdown_incidence_format(ax[1],date_to_t('2020-03-19'),365,mx,year_window=2)
ax[1].set_title("Simulated incidence of "+pnamedict[pathogen])
ax[1].set_xlabel("")
ax[1].set_ylabel("Incidence per 10k")
lockdown_susceptibility_plot(ax[2],STATE0,params,PERIOD,POINTS,date_to_t('2020-03-19'),result=result,relative=False,proportion=False, by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES)
lockdown_susceptibility_format(ax[2],date_to_t('2020-03-19'),365,year_window=2,ymax=None,ymin=None)
ax[2].set_xlabel("Date")
plt.tight_layout()
plt.savefig("Figures/DE_"+pathogen+lockdown+option1+option2+str(seed)+"_test.png",dpi=300)