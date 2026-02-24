## BJS March 2025
## Plotting results of fitting

import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import time
import pickle
import sys
import os

import contact_model as cm
from JAX_ODEs import deltas
from Parameters.census_population import *
from Parameters.times_and_contacts import *

from utils import *
from demography import *
from mobility_and_import import *
from sim_grid import *
from plotting import *
from fit_MCMC import *

pathogen, seed, lockdown, option1, option2, import_multiplier = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6])

# set seed
np.random.seed(seed)

if re.match(r'\d{4}-\d{2}-\d{2}',option1):
    start_date = option1
if re.match(r'\d{4}-\d{2}-\d{2}',option2):
    end_date = option2
    option2 = "flexage" #this is super hacky sorry

print(pathogen, seed)
if re.search(r'\d{6}',lockdown):
    with open("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+"FlexStepwise"+option1+option2+str(seed)+".pickle","rb") as f:
        opt = pickle.load(f)
else:
    try:
        with open("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","rb") as f:
            opt = pickle.load(f)
    except FileNotFoundError:
        print('File not found:',"Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle")
        sys.exit()
if opt.success:
    print("Optimization converged")
else:
    print("Optimization did not converge")
    print(opt.message)
    print(opt.x)
    sys.exit()
x = opt.x
# print(x)
# print likelihood
print("Log-Likelihood:",-1*opt.fun)

REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, tests_full = pathogen_parameters(pathogen, import_multiplier=import_multiplier)
daily_hospitalization_rates_pd = pd.read_csv('Data/Processed/KPSC_ARI_hospitalization_rates_by_day_age_group.csv',index_col=0,parse_dates=True)
daily_hospitalization_rates_pd = daily_hospitalization_rates_pd.fillna(0)
daily_hospitalization_rates = jnp.asarray(daily_hospitalization_rates_pd.values)
N_S, NAG = 3, 7
CONTACT_MATRIX = np.asarray(pd.read_csv('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', header=None).values)
BIRTH_RATE = np.genfromtxt('Data/Processed/birth_rate_daily.csv', delimiter=',')
age_pops = np.genfromtxt('Data/Processed/age_pops_daily.csv', delimiter=',')

start_date = '2015-07-04'
end_date = '2025-05-01'
EPOCH = pd.to_datetime('1970-01-01')
START = pd.to_datetime(start_date) 
END = pd.to_datetime(end_date)
FULL_PERIOD = pd.date_range(start=EPOCH, end=END, freq='D')
FULL_POINTS = np.array(date_to_t(FULL_PERIOD))
PERIOD = pd.date_range(start=START, end=END, freq='D')
POINTS = np.array(date_to_t(PERIOD))

start_idx = int(date_to_t(start_date) + 90 - date_to_t('2015-10-01'))
end_idx = int(date_to_t(end_date) - date_to_t('2015-10-01'))
tests = tests_full[start_idx:end_idx, :, :]

daily_hospitalization_rates_pd = pd.read_csv('Data/Processed/KPSC_ARI_hospitalization_rates_by_day_age_group.csv',index_col=0,parse_dates=True)
daily_hospitalization_rates_pd = daily_hospitalization_rates_pd.fillna(0)
daily_hospitalization_rates_full = jnp.asarray(daily_hospitalization_rates_pd.values)
daily_hospitalization_rates = daily_hospitalization_rates_full[start_idx:end_idx,]

## Initial conditions
STATE0 = jnp.zeros((2*N_S+1,NAG))
STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
STATE0 = STATE0.at[1,:].set(1)
# # flatten initial state and add maternal immunity compartment
STATE0 = STATE0.flatten()
STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

params = x_to_params(x, pathogen, lockdown, option1, option2, print_params=True)

solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS)
values = solution.ys.T
times = solution.ts

print(SIS_likelihood(tests, daily_hospitalization_rates, params, POINTS, STATE0, p_time_to_obs, solution=solution))

# # for each season from the 2015/16 season onwards, sum the total number of infections
seasons = np.array([date_to_t(date) for date in ['2015-10-01','2016-10-01','2017-10-01','2018-10-01','2019-10-01','2020-10-01','2021-10-01','2022-10-01','2023-10-01','2024-10-01','2025-05-01']])
season_infection_array = np.zeros((len(seasons)-1,3))
season_infection_by_age = np.zeros((len(seasons)-1,NAG,3))
first_infections = np.zeros((len(seasons)-1,NAG))
for i in range(len(seasons)-1):
    # get the number of infections in each season
    season_start = np.argmax(times>=seasons[i])
    season_end = np.argmax(times>=seasons[i+1])
    pop_size = np.sum(values[:-NAG,season_start],dtype=np.float64)
    age_pops = np.array([np.sum(values[range(1+i_age,2*N_S*NAG,NAG),season_start],axis=0) for i_age in range(NAG)])
    age_pops[0] += values[0,season_start]  # add maternal immunity compartment to first age group
    first_infections[i,:] = np.sum(values[1:1+NAG,season_start:season_end],axis=1)
    season_infection_array[i,0] = np.sum(values[1+NAG:1+2*NAG,season_start:season_end])*REC_UP[0]/pop_size
    season_infection_array[i,1] = np.sum(values[1+3*NAG:1+4*NAG,season_start:season_end])*REC_UP[1]/pop_size
    season_infection_array[i,2] = np.sum(values[1+5*NAG:1+6*NAG,season_start:season_end])*REC_SAME[2]/pop_size
    season_infection_by_age[i,:,0] = np.sum(values[1+NAG:1+2*NAG,season_start:season_end],axis=1)*REC_UP[0]/age_pops
    season_infection_by_age[i,:,1] = np.sum(values[1+3*NAG:1+4*NAG,season_start:season_end],axis=1)*REC_UP[1]/age_pops
    season_infection_by_age[i,:,2] = np.sum(values[1+5*NAG:1+6*NAG,season_start:season_end],axis=1)*REC_SAME[2]/age_pops 
average_age_of_first_infection = np.sum(first_infections*jnp.array(MEDIAN_AGE).reshape((1,NAG)),axis=1)/jnp.sum(first_infections,axis=1)
season_infections = np.sum(season_infection_array,axis=1)
season_infection_by_age = np.sum(season_infection_by_age,axis=2)
print("Average age of first infection per season:",average_age_of_first_infection/12)
print("Proportion infected per season (including reinfections):",season_infections)
# print("Proportion infected per season (by age):",season_infection_by_age)


# # get R(t)
# R0s = jnp.zeros(len(times))
# Rts = jnp.zeros(len(times))
# contact_ratios = jnp.zeros(len(times))
# SEASONALITY = x[-1]
# OFFSET = x[-2]

# for idx in range(len(times)):
#     pop_size = jnp.sum(values[:,idx],dtype=jnp.float64)
#     age_pops = jnp.array([jnp.sum(values[range(i_age,(2*N_S+1)*NAG,NAG),idx],axis=0) for i_age in range(NAG)])
#     contact_t = contact(times[idx],SEASONALITY,OFFSET)
#     infectious_contact_equal = jnp.dot(contact_t,jnp.sum(jnp.array([values[j,idx] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=jnp.float64).reshape((N_S,NAG))*I_REL,axis=0))/jnp.sum(jnp.array([values[j,idx] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=jnp.float64))
#     infectious_contact = jnp.dot(contact_t,jnp.sum(jnp.array([values[j,idx] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=jnp.float64).reshape((N_S,NAG))*I_REL,axis=0))/pop_size
#     import_contact = IMPORT_RATE*regional_positivity(times[idx])*arrivals(times[idx])*jnp.dot(contact_t,age_pops)/pop_size
#     contact_ratios[idx] = jnp.sum(jnp.repeat(S_REL,NAG*N_C)*jnp.tile(S_AGE,N_S*N_C)*BETA*jnp.tile(import_contact,N_S*N_C)*jnp.repeat(jnp.tile(jnp.array([0,1]+[0]*(N_C-2)),N_S),NAG)*jnp.array([jnp.tile(values[(2*i+1)*NAG:(2*i+2)*NAG,idx],N_C) for i in range(N_S)]).flatten())/jnp.sum(jnp.repeat(S_REL,NAG*N_C)*jnp.tile(S_AGE,N_S*N_C)*BETA*jnp.tile(infectious_contact,N_S*N_C)*jnp.repeat(jnp.tile(jnp.array([0,1]+[0]*(N_C-2)),N_S),NAG)*jnp.array([jnp.tile(values[(2*i+1)*NAG:(2*i+2)*NAG,idx],N_C) for i in range(N_S)]).flatten())
#     R0s[idx] = BETA*jnp.sum(infectious_contact_equal)/REC_UP[0]
#     Rts[idx] = (1/pop_size)*(1/REC_UP[0])*jnp.sum(jnp.repeat(S_REL,NAG*N_C)*jnp.tile(S_AGE,N_S*N_C)*BETA*jnp.tile(infectious_contact_equal,N_S*N_C)*jnp.repeat(jnp.tile(jnp.array([1,0]+[0]*(N_C-2)),N_S),NAG)*jnp.array([jnp.tile(values[(2*i+1)*NAG:(2*i+2)*NAG,idx],N_C) for i in range(N_S)]).flatten())
# print("R0:",jnp.median(R0s),"("+str(jnp.min(R0s))+"–"+str(jnp.max(R0s))+")")
# print("Rt:",jnp.median(Rts),"("+str(jnp.min(Rts))+"–"+str(jnp.max(Rts))+")")
# print("Ratio of import-caused cases to internal transmission:",jnp.median(contact_ratios),"("+str(jnp.min(contact_ratios))+"–"+str(jnp.max(contact_ratios))+")")

# remove "free" from pathogen name for plotting
if "free" in pathogen:
    pathogen_name = pathogen.replace("free","")
else:
    pathogen_name = pathogen

plt.rcParams.update({'font.size':14})
# text type is palatino
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino']
fig = plt.figure(figsize=(13.3,7.5))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2, sharex=ax1
, sharey=ax1
)
ax3 = fig.add_subplot(3,1,3, sharex=ax1)
ax = [ax1,ax2,ax3]
aggregation = "Month"
kpsc_proportion_positive_incidence_plot(ax[0], pathogen, AGE_GROUPS, AGE_GROUP_NAMES, aggregation="ME", factor=10000)
ax[0].set_xlabel("")
pnamedict = {"RSV":"RSV","InfluenzaA":"Influenza A","InfluenzaB":"Influenza B","Parainfluenza3":"Parainfluenza 3","Adenovirus":"Adenovirus","Metapneumovirus":"Metapneumovirus", "test":"test"}
# ax.set_title("Observed incidence of "+pnamedict[pathogen_name])
ax[0].set_ylabel("Monthly incidence per 10k")
# legend
ax[0].legend(frameon=False)

# fig, ax = plt.subplots(1,2,figsize=(14.5,2.8))
mx = lockdown_incidence_plot(ax[1],STATE0,params,POINTS,date_to_t('2020-03-19'),solution=solution,label="Simulation",by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=[1,30.44][[None,"Month"].index(aggregation)]*10000,p_time_to_obs=p_time_to_obs)
lockdown_incidence_format(ax[1],date_to_t('2020-03-19'),365,mx,year_window=2)

lockdown_susceptibility_plot(ax[2],STATE0,params,PERIOD,POINTS,date_to_t('2020-03-19'),solution=solution,relative=False,proportion=True, by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES)
lockdown_susceptibility_format(ax[2],date_to_t('2020-03-19'),365,year_window=2,ymax=None,ymin=None)
ax[2].set_title("Effective susceptibles")

plt.savefig("Figures/DE_"+pathogen+lockdown+option1+option2+str(seed)+"_test2.png",dpi=300)

# ax[1].set_title("Simulated incidence of "+pnamedict[pathogen])
# ax[1].set_xlabel("")
# ax[1].set_ylabel("")
# ax[1].set_xlabel("")
# ax[1].set_xticklabels(["","2016","","2018","","2020","","2022","","2024"])
# ax[0].set_yticks([])
# lockdown_susceptibility_plot(ax[2],STATE0,params,PERIOD,POINTS,date_to_t('2020-03-19'),solution=solution,relative=False,proportion=True, by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES)
# lockdown_susceptibility_format(ax[2],date_to_t('2020-03-19'),365,year_window=2,ymax=None,ymin=None)
# ax[2].set_title("Effective susceptibles")
# ax[1].set_xlabel("")
# ax[1].set_ylabel("")
# ax[1].set_xlabel("")
# ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
# ax.set_xticklabels(["","2016","","2018","","2020","","2022","","2024",""])
# # ax.set_yscale('log')
# # ax.set_ylim(1e-3,)
# # multiply y lables by 100
# ylabls = ax.get_yticks()
# ax.set_yticklabels([str(int(np.round(yl*100))) for yl in ylabls])
# plt.tight_layout()
# plt.savefig("Figures/DE_"+pathogen+lockdown+option1+option2+str(seed)+"_mini_noage_weekly.png",dpi=300)