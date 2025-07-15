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

if re.match(r'\d{4}-\d{2}-\d{2}',option1):
    start_date = option1
if re.match(r'\d{4}-\d{2}-\d{2}',option2):
    end_date = option2
    option2 = "maternal" #this is super hacky sorry

print(pathogen, seed)
if re.match(r'\d{6}',lockdown):
    with open("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+"FlexStepwise"+option1+option2+str(seed)+".pickle","rb") as f:
        opt = pickle.load(f)
else:
    try:
        with open("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","rb") as f:
            opt = pickle.load(f)
    except FileNotFoundError:
        print('No file found')
        sys.exit()
if opt.success:
    print("Optimization converged")
else:
    print("Optimization did not converge")
    print(opt.message)
    print(opt.x)
    sys.exit()
x = opt.x
# print likelihood
print("Log-Likelihood:",-1*opt.fun)
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
elif option1 == 'setimport' or option1.replace('.','',1).isdigit():
    params["IMPORT_RATE"] = 0.01
else:
    params["IMPORT_RATE"] = x[n]
    n += 1
if (("Influenza" in pathogen) and (option2 != 'nr')) or (seed <= 250407):
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
if ((seed > 250407) and (seed <= 250514)) or (option2 != 'flexage'):
    params["P_OBS"] = x[n]*pobsrel
    n += 1
elif option2 == 'flexage':
    params["P_OBS"] = pobsrel
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
else:
    with open("Data/Processed/DE_cm_opt_"+lockdown+".pickle","rb") as f:
        lock_opt = pickle.load(f)
    y = lock_opt.x
    Ts = np.array([date_to_t('1970-01-01'),date_to_t('2020-03-19'),date_to_t('2020-03-19')+y[0]*365,date_to_t('2020-03-19')+(y[0]+y[1])*365,date_to_t('2020-03-19')+(y[0]+y[1]+y[2])*365])
    F1 = y[3] # value between 0 and 1 (first lockdown)
    F2 = F1 + y[4] - F1*y[4] # value between y[3] and 1 (inter-lockdown)
    F3 = F2*y[5] # value less than F2 (second lockdown)
    F4 = F2 + y[6] - F2*y[6] # value between F2 and 1 (post-lockdown)
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
    if (seed <= 250512) or (seed > 250514):
        OBS_AGE = np.array([x[n],x[n+1],x[n+2],x[n+3],x[n+4],x[n+5],x[n+6]])
    else:
        OBS_AGE = np.array([x[n],x[n+1],x[n+2],x[n+3],x[n+4],x[n+5],1])
elif option2 == 'maternal':
    if seed <= 250512:
        nig = 2
    else:
        nig = 1
    OBS_AGE = age_detection(NAG,x[n],x[n+1],x[n+2],x[n+3],min_obs=0.025,n_infant_groups=nig)
else:
    OBS_AGE = age_detection(NAG,x[n],x[n+1],x[n+2])

print(params)
print("OBS_AGE",OBS_AGE)
# if lockdown == 'FlexStepwise' or re.match(r'\d{6}',lockdown):
#     print("Ts",[t_to_date(t) for t in Ts])
#     print("Fs",Fs)

# # sys.argv = ["fit_opt.py", pathogen, seed, lockdown, option1, option2, 0.01, 20, 1, 0.7]
# # from fit_opt import likelihood
# # # minimize the likelihood function from x using neldermead
# # opt = sp.optimize.minimize(likelihood, x,method='Nelder-Mead', options={'maxiter': 10000})
# # print(opt.x)

# result = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params.values(),t_eval=POINTS,method='RK45')
# obs = observations(result,params,OBS_AGE,incidence=False,time_conversion=30.44)

# # for each season from the 2015/16 season onwards, sum the total number of infections
# seasons = np.array([date_to_t(date) for date in ['2015-10-01','2016-10-01','2017-10-01','2018-10-01','2019-10-01','2020-10-01','2021-10-01','2022-10-01','2023-10-01']])
# season_infection_array = np.zeros((len(seasons)-1,3))
# season_infection_by_age = np.zeros((len(seasons)-1,NAG,3))
# for i in range(len(seasons)-1):
#     # get the number of infections in each season
#     season_start = np.argmax(result.t>=seasons[i])
#     season_end = np.argmax(result.t>=seasons[i+1])
#     pop_size = np.sum(result.y[:,season_start],dtype=np.float64)
#     age_pops = np.array([np.sum(result.y[range(i_age,(2*N_S+1)*NAG,NAG),season_start],axis=0) for i_age in range(NAG)])
#     season_infection_array[i,0] = np.sum(result.y[2*NAG:3*NAG,season_start:season_end])*params["REC_UP"][0]/pop_size
#     season_infection_array[i,1] = np.sum(result.y[4*NAG:5*NAG,season_start:season_end])*params["REC_UP"][1]/pop_size
#     season_infection_array[i,2] = np.sum(result.y[6*NAG:7*NAG,season_start:season_end])*params["REC_SAME"][2]/pop_size
#     season_infection_by_age[i,:,0] = np.sum(result.y[2*NAG:3*NAG,season_start:season_end],axis=1)*params["REC_UP"][0]/age_pops
#     season_infection_by_age[i,:,1] = np.sum(result.y[4*NAG:5*NAG,season_start:season_end],axis=1)*params["REC_UP"][1]/age_pops
#     season_infection_by_age[i,:,2] = np.sum(result.y[6*NAG:7*NAG,season_start:season_end],axis=1)*params["REC_SAME"][2]/age_pops 
# season_infections = np.sum(season_infection_array,axis=1)
# season_infection_by_age = np.sum(season_infection_by_age,axis=2)
# print("Proportion infected per season (including reinfections):",season_infections)
# print("Proportion infected per season (by age):",season_infection_by_age)



# WANE = params["WANE"]
# SEASONALITY = params["SEASONALITY"]
# OFFSET = params["OFFSET"]
# BETA = params["BETA"]
# REC_UP = params["REC_UP"]
# REC_SAME = params["REC_SAME"]
# S_REL = params["S_REL"]
# S_AGE = params["S_AGE"]
# I_REL = params["I_REL"]
# P_OBS = params["P_OBS"]
# IMPORT_RATE = params["IMPORT_RATE"]
# contact = params["contact"]
# regional_positivity = params["regional_positivity"]

# # get R(t)
# R0s = np.zeros(len(result.t))
# Rts = np.zeros(len(result.t))
# contact_ratios = np.zeros(len(result.t))
# for idx in range(len(result.t)):
#     pop_size = np.sum(result.y[:,idx],dtype=np.float64)
#     age_pops = np.array([np.sum(result.y[range(i_age,(2*N_S+1)*NAG,NAG),idx],axis=0) for i_age in range(NAG)])
#     contact_t = contact(result.t[idx],SEASONALITY,OFFSET)
#     infectious_contact_equal = np.dot(contact_t,np.sum(np.array([result.y[j,idx] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=np.float64).reshape((N_S,NAG))*I_REL,axis=0))/np.sum(np.array([result.y[j,idx] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=np.float64))
#     infectious_contact = np.dot(contact_t,np.sum(np.array([result.y[j,idx] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=np.float64).reshape((N_S,NAG))*I_REL,axis=0))/pop_size
#     import_contact = IMPORT_RATE*regional_positivity(result.t[idx])*arrivals(result.t[idx])*np.dot(contact_t,age_pops)/pop_size
#     contact_ratios[idx] = np.sum(np.repeat(S_REL,NAG*N_C)*np.tile(S_AGE,N_S*N_C)*BETA*np.tile(import_contact,N_S*N_C)*np.repeat(np.tile(np.array([0,1]+[0]*(N_C-2)),N_S),NAG)*np.array([np.tile(result.y[(2*i+1)*NAG:(2*i+2)*NAG,idx],N_C) for i in range(N_S)]).flatten())/np.sum(np.repeat(S_REL,NAG*N_C)*np.tile(S_AGE,N_S*N_C)*BETA*np.tile(infectious_contact,N_S*N_C)*np.repeat(np.tile(np.array([0,1]+[0]*(N_C-2)),N_S),NAG)*np.array([np.tile(result.y[(2*i+1)*NAG:(2*i+2)*NAG,idx],N_C) for i in range(N_S)]).flatten())
#     R0s[idx] = BETA*np.sum(infectious_contact_equal)/REC_UP[0]
#     Rts[idx] = (1/pop_size)*(1/REC_UP[0])*np.sum(np.repeat(S_REL,NAG*N_C)*np.tile(S_AGE,N_S*N_C)*BETA*np.tile(infectious_contact_equal,N_S*N_C)*np.repeat(np.tile(np.array([1,0]+[0]*(N_C-2)),N_S),NAG)*np.array([np.tile(result.y[(2*i+1)*NAG:(2*i+2)*NAG,idx],N_C) for i in range(N_S)]).flatten())
# print("R0:",np.median(R0s),"("+str(np.min(R0s))+"–"+str(np.max(R0s))+")")
# print("Rt:",np.median(Rts),"("+str(np.min(Rts))+"–"+str(np.max(Rts))+")")
# print("Ratio of import-caused cases to internal transmission:",np.median(contact_ratios),"("+str(np.min(contact_ratios))+"–"+str(np.max(contact_ratios))+")")

# fig = plt.figure(figsize=(13.3,7.5))
# ax1 = fig.add_subplot(3,1,1)
# ax2 = fig.add_subplot(3,1,2, sharex=ax1, sharey=ax1)
# ax3 = fig.add_subplot(3,1,3, sharex=ax1)
# ax = [ax1,ax2,ax3]
# kpsc_positive_test_plot(ax[0],pathogen=pathogen,AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, incidence=True, legend=False,aggregation="Month")
# ax[0].set_xlabel("")
# pnamedict = {"RSV":"RSV","InfluenzaA":"Influenza A","InfluenzaB":"Influenza B","Parainfluenza3":"Parainfluenza 3","Adenovirus":"Adenovirus","Metapneumovirus":"Metapneumovirus"}
# ax[0].set_title("Observed incidence of "+pnamedict[pathogen])
# ax[0].set_ylabel("Incidence per 10k")
# # legend
# ax[0].legend(frameon=False)

# # plt.rcParams.update({'font.size':20})
# # # text type is palatino
# # plt.rcParams['font.family'] = 'serif'
# # plt.rcParams['font.serif'] = ['Palatino']
# # fig, ax = plt.subplots(1,2,figsize=(14.5,2.8))
# mx = lockdown_incidence_plot(ax[1],STATE0,params,OBS_AGE,PERIOD,POINTS,date_to_t('2020-03-19'),365,result=result,obs=obs,label="Simulation",by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs)
# lockdown_incidence_format(ax[1],date_to_t('2020-03-19'),365,mx,year_window=2)
# ax[1].set_title("Simulated incidence of "+pnamedict[pathogen])
# # ax[1].set_xlabel("")
# # ax[1].set_ylabel("")
# # ax[1].set_xlabel("")
# # ax[1].set_xticklabels(["","2016","","2018","","2020","","2022","","2024"])
# # ax[0].set_yticks([])
# lockdown_susceptibility_plot(ax[2],STATE0,params,PERIOD,POINTS,date_to_t('2020-03-19'),result=result,relative=False,proportion=False, by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES)
# lockdown_susceptibility_format(ax[2],date_to_t('2020-03-19'),365,year_window=2,ymax=None,ymin=None)
# ax[1].set_title("Effective susceptibles")
# # ax[1].set_xlabel("")
# # ax[1].set_ylabel("")
# # ax[1].set_xlabel("")
# # ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
# # ax[1].set_xticklabels(["","2016","","2018","","2020","","2022","","2024"])
# plt.tight_layout()
# plt.savefig("Figures/DE_"+pathogen+lockdown+option1+option2+str(seed)+"_importobs.png",dpi=300)