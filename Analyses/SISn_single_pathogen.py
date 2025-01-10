## SIS model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import pandas as pd
import time
import itertools as it
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm as pltcm
import corner
import pickle
import colorsys
import datetime
from numba import jit
from sklearn import decomposition

from vaccination import birth_vax, all_vax
import contact_model as cm
from SISn_ODEs import single_pathogen_deltas as sis_deltas
from Parameters.census_population import *
from Parameters.InfluenzaA import *

from utils import *
from demography import *
from mobility_and_import import *
from clustering import *
from sim_grid import *
from plotting import *
from fit_MCMC import *

# print all numpy array elements
np.set_printoptions(threshold=np.inf)

fig, ax = plt.subplots(2,1,figsize=(13.3,7.5),sharey=True)
kpsc_positive_test_plot(ax[0],pathogen="Influenza A",AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, incidence=True, legend=False,aggregation="Month")
ax[0].set_xlabel('Time (years)')
# kpsc_positive_test_plot(ax[1],pathogen="RSV",AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, incidence=True, legend=False,aggregation="Month")
# plt.show()
# plt.savefig('Figures/KPSC_RSV_daily.png',dpi=300)


# ## Period of simulation
# EPOCH = pd.to_datetime('1970-01-01')
# START = pd.to_datetime('2015-08-01')
# END = pd.to_datetime('2023-10-01')
# PERIOD = pd.date_range(start=START, end=END, freq='D')

# ## Contacts and force of infection
# IMPORT_RATE = 1e-12
# # Contact matrix for all contact types
# CONTACT = np.genfromtxt('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', dtype=np.float64)
# print(CONTACT.shape)
# print(NAG)
# print(OBS_AGE.shape)
# print(CENSUS_AGE_POP.shape)

# # Lockdown and other mobility changes
# T_LOCKDOWN = date_to_t('2020-03-01')
# LOCKDOWN_DURATION = 365
# LOCKDOWN_REDUCTION = 0.4
# Ts = np.array([date_to_t(EPOCH),T_LOCKDOWN,date_to_t('2021-05-01'),date_to_t('2021-12-01'),date_to_t('2022-03-01')])
# Fs = np.array([1,0.4,1,0.4,1])
# @jit
# def contact(t,seasonality,offset):
#     return cm.piecewise(t,Ts,Fs)*(1+seasonality*np.cos(2*np.pi*(t/365-offset)))*CONTACT

# # ## Integrate the system
# # # POINTS = np.concat((np.zeros(1),np.arange(T_LOCKDOWN-12*12,T_LOCKDOWN+LOCKDOWN_DURATION+12*12,0.2),np.ones(1)*PERIOD))
# # # POINTS = np.array([date_to_t('1960-01-01') + pd.DateOffset(months=x) for x in range(PERIOD)])
# T_VAX = date_to_t('2035-01-01')

# POINTS = np.array(date_to_t(PERIOD))


from Parameters.times_and_contacts import *
T_LOCKDOWN = date_to_t('2019-03-19')
LOCKDOWN_DURATION = 365
p_time_to_obs = np.genfromtxt("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)

## Initial conditions
STATE0 = np.zeros((2*N_S+2)*NAG)
STATE0[NAG:2*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

# Parameters for the ODE
params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV, 'T_VAX': T_VAX,
'arrivals': arrivals, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
'contact': contact}

incidence = pd.read_csv("Data/Processed/KPSC_Influenza_A_incidence_age_daily.csv",index_col=0)

def likelihood(x):
    sim_params = params.copy()
    sim_params["WANE"] = np.array([0.0,x[0],0.0])
    sim_params["SEASONALITY"] = x[1]
    sim_params["OFFSET"] = x[2]
    sim_params["BETA"] = x[3]
    sim_params["IMPORT_RATE"] = x[4]
    sim_params["P_OBS"] =  x[5]*np.ones(3)
    sim_params["S_REL"] = np.array([1,x[6],x[7]])
    obs_age = age_detection(NAG,x[8],x[9],x[10])
    return -SIS_likelihood(incidence,sim_params,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)
opt = sp.optimize.minimize(likelihood,[1/270,0.1,0.8,0.1,1e-12,0.03,0.8,0.32,0.05,0.8,0.1],method='Nelder-Mead')
print(opt)

# print(likelihood([1/30,0.25,0.85,0.5,0.1,0.85,0.14,1e-12,0.03]))

# with open("Data/Processed/SIS_noisy_obs_daily_BETAp08_SEASONALITYp1_OFFSETp85_WANE10y_POBSp01_OBSAGEp2p85p1.pickle","rb") as f:
#     noisy_incidence = pickle.load(f)
# params["BETA"] = 0.08
# params["SEASONALITY"] = 0.1
# params["OFFSET"] = 0.85
# params["WANE"] = 1/10*np.array([0.0,1.0,0.0])/365
# params["P_OBS"] =  0.01*np.ones(N_S)
# OBS_AGE = age_detection(NAG,0.2,0.85,0.1)

pms = opt.x
params["WANE"] = np.array([0.0,pms[0],0.0])
params["SEASONALITY"] = pms[1]
params["OFFSET"] = pms[2]
params["BETA"] = pms[3]
params["IMPORT_RATE"] = pms[4]
params["P_OBS"] =  pms[5]*np.ones(3)
params["S_REL"] = np.array([1,pms[6],pms[6]])
OBS_AGE = age_detection(NAG,pms[8],pms[9],pms[10])

# # # # # # # # # #### One-shot line plot #####
# fig, ax = plt.subplots(figsize=(6.5,8.5))
result = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params.values(),t_eval=POINTS,method='RK45')
obs = observations(result,params,OBS_AGE,incidence=False,time_conversion=30.44)
mx = lockdown_incidence_plot(ax[1],STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,obs=obs,label="Simulation",by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs)
lockdown_incidence_format(ax[1],T_LOCKDOWN,LOCKDOWN_DURATION,mx,year_window=2)
plt.savefig('Figures/InfluenzA_fit_test.png',dpi=300)
# # params["WANE"] = np.array([0,1.195e-01,0])/365
# # params["P_OBS"] = 3.396e-02*np.array([1,0.46,0.31])
# # OBS_AGE = np.array([1,0.8229,0.6458,0.4687,0.2916,0.1375,0.8626])
# # params["SEASONALITY"] = 1.727e-01
# # params["OFFSET"] = 8.584e-01
# # params["BETA"] = 6.983e-02

# pms = [2.547e-02,1.656e-01
# ,8.209e-01,1.086e-01,1.238e-12,2.627e-02,3.167e-01,7.424e-01,5.960e-01]
# # mcmc value
# pms = [8.17814276e-05,1.51485052e-01,8.12003008e-01,1.14221137e-01
# ,1.53783669e-12,3.67742977e-02
# ,3.17488852e-01,7.53475678e-01,5.81731678e-01]
# # # print(likelihood(pms))


# result2 = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params.values(),t_eval=POINTS,method='RK45')
# obs2 = observations(result2,params,OBS_AGE,incidence=False,time_conversion=30.44)
# mx2 = lockdown_incidence_plot(ax[1],STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result2,obs=obs2,label="Fit",by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,color='#DC267F',p_time_to_obs=p_time_to_obs)
# lockdown_incidence_format(ax[1],T_LOCKDOWN,LOCKDOWN_DURATION,mx2,year_window=2)
# plt.tight_layout()
# plt.savefig('Figures/RSV_fit_test.png',dpi=300)

# ax[1].set_ylabel('Simulated incidence per 10k')
# ax[2].set_ylabel('Simulated incidence per 10k')
# ax[1].set_xlim(POINTS[89],POINTS[-1])
# ax[2].set_xlim(POINTS[89],POINTS[-1])
# # # # # # copy y limit from ax[1]
# # # ax[2].set_ylim(ax[1].get_ylim())

# ax[1].set_title('Hand-tuned simulation')
# ax[2].set_title('Maximum likelihood simulation')

# plt.tight_layout()

# plt.savefig('Figures/RSV_compare_fits_mcmc_trajectory_from_NM_test.png',dpi=300)

# params["BETA"] = 8.710e-02
# params["P_OBS"] = 1.535e-02*np.ones(N_S)
# params["SEASONALITY"] = 9.969e-02
# params["OFFSET"] = -1.369e-01
# params["WANE"] = 1/30*np.array([0.0,3.918e-01,0.0])/365
# OBS_AGE = age_detection(NAG,2.418,8.514e-01,4.772e-01)
# result2 = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params.values(),t_eval=POINTS,method='RK45')
# obs2 = observations(result2,params,OBS_AGE,incidence=False,time_conversion=1)

# # # # obs = np.sum(obs,axis=1)
# # # noisy_obs = np.random.poisson(obs)
# # # pop_size_by_age = np.array([np.sum(result.y[range(i_age,(2*N_S+1)*NAG,NAG),:],axis=0) for i_age in range(NAG)]).T
# # # noisy_incidence = noisy_obs/pop_size_by_age
# # # with open("Data/Processed/SIS_noisy_obs_daily_BETAp08_SEASONALITYp1_OFFSETp85_WANE10y_POBSp01_OBSAGEp2p85p1.pickle","wb") as f:
# # #     pickle.dump(noisy_incidence,f)
# # fig, ax = plt.subplots(2,1,figsize=(6.5,8.5))
# # mx = lockdown_incidence_plot(ax[0],STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,obs=obs,label="Simulation",by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000)
# mx = lockdown_incidence_plot(axes[1],STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,obs=obs,by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,color='#DC267F')
# # # plot monthly moving average of noisy incidence
# # moving_average = np.array([np.mean(noisy_incidence[i-30:i],axis=0)*10000 for i in range(30,len(noisy_incidence))])
# # # plot the age stratified incidence in shades of grey
# # # define grayscale color map
# hsv_colors = colormaps.hsv(-0.02+np.arange(7)/7)
# hsv_colors[3] = colormaps.hsv((3/7)+0.04)
# # for i in range(NAG):
# #     ax[0].plot(POINTS[30:],moving_average[:,i],color=hsv_colors[i],linestyle='--')
# #     ax[1].plot(POINTS[30:],moving_average[:,i],color=hsv_colors[i],linestyle='--')
# # # mx2 = lockdown_incidence_plot(ax,STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,obs=noisy_incidence,label="Simulation",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,color='#DC267F')
# # lockdown_incidence_format(ax[0],T_LOCKDOWN,LOCKDOWN_DURATION,mx,year_window=2)
# lockdown_incidence_format(axes[1],T_LOCKDOWN,LOCKDOWN_DURATION,mx,year_window=2)
# # # ax.legend()
# # plt.show()
# # ax[0].set_title('Simulation with noisy observations')
# # ax[1].set_title('Fit to noisy observations')
# # ax[0].set_ylabel('Simulated incidence per 10k')
# # ax[1].set_ylabel('Simulated incidence per 10k')
# # plt.savefig('Figures/test.png',dpi=300)
# # # axes[1].legend()
# axes[1].set_ylabel('Simulated incidence per 10k')
# # plot google_prestige_work
# axes2 = axes[1].twinx()
# # print([cm.google_prestige_work(point) for point in POINTS])
# axes2.plot(POINTS,[cm.piecewise(point,Ts,Fs) for point in POINTS],color='black',linestyle='--',label='Google workplace mobility')
# axes[1].legend(axes[1].lines,['<1y','1-4y','5-17y','18-39y','40-64y','>=65y'],loc='upper left',title='Age group')
# # axes[1].vlines(18952,0,mx,linestyle=':',color='black')
# axes[1].set_xlim(POINTS[0],POINTS[-1])
# @jit
# def contact(t,seasonality,offset):
#     return cm.google_prestige_work(t)*(1+seasonality*np.cos(2*np.pi*(t/365-offset)))*CONTACT
# params['contact'] = contact
# result = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params.values(),t_eval=POINTS,method='RK45')
# obs = observations(result,params,OBS_AGE,incidence=False,time_conversion=30.44)
# mx = lockdown_incidence_plot(axes[2],STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,obs=obs,label="Simulation",by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000)
# lockdown_incidence_format(axes[2],T_LOCKDOWN,LOCKDOWN_DURATION,mx,year_window=2,title="Simulation with workplace mobility contact model")
# axes[2].set_ylabel('Simulated incidence per 10k')
# axes3 = axes[2].twinx()
# axes3.plot(POINTS,[cm.google_prestige_work(point) for point in POINTS],color='black',linestyle='--',label='Google workplace mobility')
# # axes[2].legend(axes[2].lines,['<1y','1-4y','5-17y','18-39y','40-64y','>=65y'],loc='upper left',title='Age group')   
# axes[2].set_xlim(POINTS[0],POINTS[-1])
# plt.tight_layout()
# # plt.savefig('Figures/8param_fit_test.png',dpi=300)
# plt.savefig('Figures/RSV_mobility_tests.png',dpi=300)

# incidence = np.array(pd.read_csv("Data/Processed/KPSC_RSV_incidence_age.csv",index_col=0))
# with open("Data/Processed/SIS_noisy_obs_BETAp15_SEASONALITYp06_OFFSETp1_WANE10y_ADRp1.pickle","rb") as f:
#     incidence = pickle.load(f)

# # # # # case_data = pd.read_csv('Data/Processed/KPSC_rsv_hosp_incidence_by_age.csv')
# # # # # # case_data = pd.read_csv('Data/Processed/KPSC_flu_hosp.csv')['Count']
# # # # # case_data = np.array(case_data)
# # # # # # print(case_data)

# # # # points_trimmed = np.array(date_to_t(PERIOD[PERIOD < '2020-01-01']))

# params["WANE"] = np.array([0.0,pms[0],0.0])/365
# OBS_AGE = age_detection(NAG,pms[1],pms[2],pms[3])
# params["SEASONALITY"] = pms[4]
# params["OFFSET"] = pms[5]
# params["BETA"] = pms[6]
# params["IMPORT_RATE"] = pms[7]
# params["P_OBS"] =  pms[8]*np.array([1,0.46,0.31])


# np.random.seed(241108)
# variables = ["WANE","SEASONALITY","OFFSET","BETA","IMPORT_RATE","P_OBS","S_REL1","S_REL2","OBS_AGE_YOUNG","OBS_AGE_OLD","OBS_AGE_YOUNG_OLD"]
# initial_scalars = np.array([opt.x])
# log_priors_distribution = sp.stats.multivariate_normal([-1]*11,np.diag([1.5]*11))
# log_priors = lambda x : log_priors_distribution.logpdf(x)
# proposal_cov = np.diag([1e-4]*11)
# mcmc_trajectory, acceptance_rate, likelihoods = mcmc(incidence, params, POINTS, STATE0, OBS_AGE, SIS_likelihood, p_time_to_obs, variables, initial_scalars, log_priors, proposal_cov, 200, True, True)
# print(acceptance_rate)
# print(np.mean(mcmc_trajectory,axis=0))
# plt.plot(mcmc_trajectory)
# plt.savefig('Figures/mcmc_trajectory_from_NM_test.png',dpi=300)
# with open('Data/Processed/mcmc_trajectory_from_NM_test.pickle','wb') as f:
#     pickle.dump(mcmc_trajectory,f)

# principal_componets = decomposition.PCA(n_components=2)
# X = principal_componets.fit_transform(mcmc_trajectory)
# fig, ax = plt.subplots(1,1,figsize=(6.5,6.5))
# ax.plot(X[:,0],X[:,1],color='silver')
# ax.scatter(X[:,0],X[:,1],c=likelihoods,cmap='viridis')
# plt.savefig('Figures/PCA_mcmc_trajectory_from_NM_test.png',dpi=300)


# # plt.show()
# with open('Data/Processed/mcmc_trajectory_multivariate.pickle','rb') as f:
#     mcmc_trajectory = pickle.load(f)
# # with open('Data/Processed/mcmc_trajectory_pre_lockdown.pickle','rb') as f:
# #     mcmc_trajectory_pre_lockdown = pickle.load(f)
# plt.plot(mcmc_trajectory)
# plt.show()

# print(np.mean(mcmc_trajectory,axis=0))

# burn_in = 1000

# # plt.plot(mcmc_trajectory)
# # plt.show()

# # figure = corner.corner(mcmc_trajectory[burn_in:],labels=['Transmissibility','Seasonality','Immunity'])
# # plt.savefig('Figures/SIS_MCMC_corner.png',dpi=300)

# fig = plt.figure(figsize=(13.3,7.5),layout='constrained')
# gs = GridSpec(2,3,figure=fig)
# trajectory_ax = fig.add_subplot(gs[0,:])
# heat_axes = np.array([fig.add_subplot(gs[1,i]) for i in range(3)])
# heat_axes[0].hist2d(mcmc_trajectory[burn_in:,0],mcmc_trajectory[burn_in:,1],bins=20)
# # outline the cell corresponding to the true values - 0.06 and 0.15 - in red
# heat_axes[0].plot([0.15,0.15,0.1501,0.1501,0.15],[0.05995,0.06005,0.06005,0.05995,0.05995],color='red')
# heat_axes[0].set_xlabel('Transmissibility')
# heat_axes[0].set_ylabel('Seasonality')
# heat_axes[1].hist2d(mcmc_trajectory[burn_in:,0],mcmc_trajectory[burn_in:,2],bins=20)
# heat_axes[1].plot([0.15,0.15,0.1501,0.1501,0.15],[0.39995,0.40005,0.40005,0.39995,0.39995],color='red')
# heat_axes[1].set_xlabel('Transmissibility')
# heat_axes[1].set_ylabel('Immunity')
# heat_axes[2].hist2d(mcmc_trajectory[burn_in:,1],mcmc_trajectory[burn_in:,2],bins=20)
# heat_axes[2].plot([0.05995,0.05995,0.06005,0.06005,0.05995],[0.39995,0.40005,0.40005,0.39995,0.39995],color='red')
# heat_axes[2].set_xlabel('Seasonality')
# heat_axes[2].set_ylabel('Immunity')
# trajectory_ax.plot(mcmc_trajectory[:,0],label='Transmissibility')
# trajectory_ax.plot(mcmc_trajectory[:,1],label='Seasonality')
# trajectory_ax.plot(mcmc_trajectory[:,2],label='Immunity')
# trajectory_ax.legend()
# trajectory_ax.set_xlabel('Iteration')
# trajectory_ax.set_ylabel('Parameter value')
# # plt.tight_layout()
# plt.savefig('Figures/example_MCMC_trajectory_multivariate.png',dpi=300)

# # ####### Computing observations ########
# with open('Data/Processed/SIS_3D_little.pickle','rb') as f:
#     results = pickle.load(f)
# with open('Data/Processed/SIS_3D_little_params.pickle','rb') as f:
#     param_dict = pickle.load(f)
# obses = {}
# for key,result in results.items():
#     if np.all(np.array(key[1:]) == 0):
#         print(key)
#     params = param_dict[key]
#     # params['contact'] = lambda t, seasonality, offset : contact(t,shape_static,seasonality,offset)
#     obs = observations(result,params,OBS_AGE,incidence=True)
#     obses[key] = obs
# with open('Data/Processed/SIS_3D_little_obs.pickle','wb') as f:
#     pickle.dump(obses,f)

# # ######## Plotting clusters ########
# with open('Data/Processed/SIS_3D_little.pickle','rb') as f:
#     results = pickle.load(f)
# with open('Data/Processed/SIS_3D_little_obs.pickle','rb') as f:
#     obses = pickle.load(f)

# # #### plot age-based clusters
# ages = np.array([np.mean(age_of_first_infection(results[key],MEDIAN_AGE)[np.argmax(results[key].t>=T_LOCKDOWN-5*12):np.argmax(results[key].t>T_LOCKDOWN)]) for key in results.keys()])

# print(np.sum(ages<=3))
# ages_label = np.array([int(age>3) + int(age>12) + int(age>5*12) + int(age>18*12) + int(age>40*12) + int(age>65*12) for age in ages])
# n_age_clusters = max(ages_label)+1
# print(n_age_clusters)

# fig,axes = plt.subplots(n_age_clusters,4,figsize=(6.5,1.7*n_age_clusters),sharex='col',layout='constrained',squeeze=False)
# for row in range(n_age_clusters):
#     axes[row,1].sharey(axes[row,2])
#     axes[row,2].sharey(axes[row,3])
# cluster_plot(axes,results,obses,n_age_clusters,ages_label,None,color=False,line=True,N=15)
# fig.align_ylabels()
# plt.savefig('Figures/SIS_3D_2_age_clusters.png',dpi=500)

# # # # ## plot n clusters
# n_clusters = 3
# model = cluster_sims(results,obses,T_LOCKDOWN,n_clusters,width=5)
# fig, axes = plt.subplots(n_clusters,4,figsize=(6.5,1.2*n_clusters),sharex='col',layout='constrained',squeeze=False)
# for row in range(n_clusters):
#     axes[row,1].sharey(axes[row,2])
#     axes[row,2].sharey(axes[row,3])
# cluster_plot(axes,results,obses,model.n_clusters,model.labels_,model.cluster_centers_,color=True,line=False,
# base_values=[1/10,1/913,1/4],factors=[0.7,3,1],grid_mode=("scale","scale","based_vec"),
# N=20)
# fig.align_ylabels()
# plt.savefig('Figures/SIS_3D_little_'+str(n_clusters)+'clusters.png',dpi=500)

# ### plot all sims and select clusters
# model = cluster_sims(results,obses,T_LOCKDOWN,6)
# model1 = cluster_sims(results,obses,T_LOCKDOWN,1)
# ages = np.array([np.mean(age_of_first_infection(results[key],MEDIAN_AGE)[np.argmax(results[key].t>=T_LOCKDOWN-5*12):np.argmax(results[key].t>T_LOCKDOWN)]) for key in results.keys()])
# ages_sd = np.array([np.mean(age_of_first_infection(results[key],MEDIAN_AGE,sd=True)[1][np.argmax(results[key].t>=T_LOCKDOWN-5*12):np.argmax(results[key].t>T_LOCKDOWN)]) for key in results.keys()])

# def hsv_to_hex(h, s, v):
#     r, g, b = colorsys.hsv_to_rgb(h, s, v)
#     return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))

# age_colors = np.array([hsv_to_hex(mu/np.max(ages),1-sd/np.max(ages_sd), 0.9) for mu,sd in zip(ages,ages_sd)])

# fig, axes = plt.subplots(4,4,figsize=(6.5,1.7*4),sharex='col',layout='constrained',squeeze=False)
# for row in range(4):
#     axes[row,1].sharey(axes[row,2])
#     axes[row,2].sharey(axes[row,3])
# first_row = axes[0,:]
# first_row.shape = (1,4)
# # cluster_colors = np.array([["#648FFF", "#DC267F", "#785EF0", "#FFB000", "#FF832B", "#000000", "#FFD662", "#FF34FF", "#8B0000", "#00FF00"][i] for i in model.labels_])
# cluster_plot(first_row,results,obses,model1.n_clusters,model1.labels_,None,color=True,line=False,color_values_all=age_colors)
# cluster_plot(axes[1:,:],results,obses,model.n_clusters,model.labels_,model.cluster_centers_,color=True,clusters=[0,2,4],line=False,color_values_all=age_colors)
# fig.align_ylabels()
# plt.savefig('Figures/SIS_3D_clusters_1plus3of6_age_color.png',dpi=900)

# # ## Computing observations for a given cluster
# with open('Data/Processed/SIS_3D_based.pickle','rb') as f:
#     results = pickle.load(f)
# pick_cluster = 0
# # separate out results and observations from cluster pick and save
# cluster_pick_results = {}
# cluster_pick_obses = {}
# for key,result in results.items():
#     if model.labels_[list(results.keys()).index(key)] == pick_cluster:
#         cluster_pick_results[key] = result
#         cluster_pick_obses[key] = obses[key]
# with open('Data/Processed/SIS_3D_based_cluster'+str(pick_cluster+1)+'_results.pickle','wb') as f:
#     pickle.dump(cluster_pick_results,f)
# with open('Data/Processed/SIS_3D_based_cluster'+str(pick_cluster+1)+'_obses.pickle','wb') as f:
#     pickle.dump(cluster_pick_obses,f)

# ## Plot sub-clusters
# pick_cluster = 0
# # # ## load cluster and divide into more clusters
# with open('Data/Processed/SIS_3D_based_cluster'+str(pick_cluster+1)+'_results.pickle','rb') as f:
#     results = pickle.load(f)
# with open('Data/Processed/SIS_3D_based_cluster'+str(pick_cluster+1)+'_obses.pickle','rb') as f:
#     obses = pickle.load(f)

# n_clusters = 5
# model = cluster_sims(results,obses,T_LOCKDOWN,n_clusters)
# with open('Data/Processed/SIS_3D_based_'+str(n_clusters)+'clusters_of_cluster'+str(pick_cluster+1)+'of3.pickle','wb') as f:
#     pickle.dump(model,f)
# fig, axes = plt.subplots(n_clusters,4,figsize=(6.5,1.1*n_clusters),sharex='col',layout='constrained',squeeze=False)
# for row in range(n_clusters):
#     axes[row,1].sharey(axes[row,2])
#     axes[row,2].sharey(axes[row,3])
# cluster_plot(axes,results,obses,model.n_clusters,model.labels_,model.cluster_centers_,color=False,line=True,y_value="rebound peak incidence")
# fig.align_ylabels()
# plt.savefig('Figures/SIS_3D_based_'+str(n_clusters)+'clusters_of_cluster'+str(pick_cluster+1)+'of3_rebound_size.png',dpi=300)

#### line plots with different parameter values, showing incidence and susceptibility #####
# params['BETA'] = 70
# # params['SEASONALITY'] = 0.061
# params['S_REL'] = np.array([1,0.8,0.6])
# fig, ax = plt.subplots(2,1,figsize=(6.5,6.5))
# mx = np.zeros(4)
# colors = ['#648FFF', '#DC267F', '#785EF0', '#FFB000']
# # Plot the incidence
# for i in range(2):
#     params['BETA'] = [30,50][i]
#     result = sp.integrate.solve_ivp(sis_deltas,(0,PERIOD),STATE0,args=(params,),t_eval=POINTS,method='RK45')
#     # plot each susceptible compartment
# #     ax[i//2,i%2].plot(result.t[1:],np.sum(result.y[NAG:2*NAG,:],axis=0)[1:],label='S1',color=colors[0])
# #     ax[i//2,i%2].plot(result.t[1:],np.sum(result.y[3*NAG:4*NAG,:],axis=0)[1:],label='S2',color=colors[1])
# #     ax[i//2,i%2].plot(result.t[1:],np.sum(result.y[5*NAG:6*NAG,:],axis=0)[1:],label='S3',color=colors[2])
# #     ax[i//2,i%2].set_title(f"Aquired immunity: {0.1*(i+2):.2f}")
# # ax[0,0].legend()
# # plt.show()
#     obs = observations(result,params,OBS_AGE,incidence=True)
#     print(params['P_OBS'])
#     print(OBS_AGE)
#     print(np.max(obs))
#     pre_obs = obs[(result.t>T_LOCKDOWN-12*12) & (result.t<T_LOCKDOWN)]
#     corr = np.correlate(pre_obs, pre_obs, mode='same')
#     acorr = corr[len(pre_obs)//2 + 1:] / (pre_obs.var() * np.arange(len(pre_obs)-1, len(pre_obs)//2, -1))
#     acorr = acorr + np.linspace(0.1, 0, len(acorr))
#     lag = np.abs(acorr).argmax() + 1
#     print(lag)
#     mx[i] = lockdown_incidence_plot(ax[0],STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,label=f'{[50,70][i]:.2f}',color=colors[i],relative=False,obs=obs)
#     lockdown_susceptibility_plot(ax[1],STATE0,params,PERIOD,POINTS,T_LOCKDOWN,result=result,label=f'{[50,70][i]:.2f}',color=colors[i],relative=False)

# lockdown_incidence_format(ax[0],T_LOCKDOWN,LOCKDOWN_DURATION,max(mx),year_window=10)
# lockdown_susceptibility_format(ax[1],T_LOCKDOWN,LOCKDOWN_DURATION,ymax=None,year_window=10)
# ax[0].set_ylabel("Observed incidence")
# # ax[0].set_ylim(0,1.1)
# ax[1].set_ylabel("Effective population susceptibility")
# # ax[1].set_ylim(1.5e7,2.2e7)
# ax[1].legend(title="Transm.")
# plt.tight_layout()
# plt.savefig('Figures/test.png',dpi=300)

# # ##### run multi-dimensional GRID SIMS #####
# start = time.time()
# param_dict, results = sim_grid(STATE0,params,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,
# grid_params=(("BETA",),("WANE",),("S_REL",)),
# N=2,factors=(0.7,3,1),grid_mode=("scale","scale","based_vec"))
# print(f"Simulation took {time.time()-start:.2f} seconds")
# # Save the results
# with open('Data/Processed/SIS_3D_little.pickle','wb') as f:
#     pickle.dump(results,f)
# with open('Data/Processed/SIS_3D_little_params.pickle','wb') as f:
#     pickle.dump(param_dict,f)

# ##### Plotting multi-dimensional grid sims #####
# # Load the results
# with open('Data/Processed/SIS_4D.pickle','rb') as f:
#     results = pickle.load(f)

# fig, axes = plt.subplots(2,3,figsize=(6.5,4.5),layout='constrained')
# start = time.time()
# im = grid_plot(axes,results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE,
# grid_params=(("BETA",),("WANE",),("S_REL",),("SEASONALITY",)),
# factors=(1,1,1,2),
# grid_mode=("scale","scale","fade_vec","scale"),
# label_mode=("mean","nz_mean","fade_vec","mean"),
# x_labels=("Transmission","Waning","Acquired immunity","Seasonality"),
# z_value="time to rebound",z_label="",
# fix=True,vmin=0,vmax=5)
# print(f"Plotting took {time.time()-start:.2f} seconds")
# # colorbar
# fig.colorbar(im, ax=axes, orientation='horizontal', label="Time to rebound (years)")
# # plt.tight_layout()
# # plt.show()
# plt.savefig('Figures/SIS_4D_rebound_time_to_rebound_fix.png',dpi=300)

##### Grid plots of oscillation size and peak incidence #####
# fig, ax = plt.subplots(2,1,figsize=(6.5,6.5))
# grid_plot(ax[0],results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE,
# grid_params=(("BETA",),("REC_UP","REC_SAME")),label_mode=("mean","nz_mean"),
# z_value="oscillation size",z_label="Oscillation size")
# grid_plot(ax[1],results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE,
# grid_params=(("BETA",),("REC_UP","REC_SAME")),label_mode=("mean","nz_mean"),
# z_value="peak incidence",z_label="Peak observed incidence")
# plt.show()

##### Grid plots of child infections, incidence, perodicity, time to rebound, susceptibility #####
# # # measure time to get results
# # start = time.time()
# # results = sim_grid(STATE0,params,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,N=10
# # ,grid_params=(("BETA","REC_UP","REC_SAME"),("S_REL",)),grid_mode=("scale","fade_vec"),factors=(1,1))
# # # save results
# # with open('Data/Processed/test_adult_SIS.pkl','wb') as f:
# #     pickle.dump(results,f)
# # end = time.time()
# # print('Time to get results:',end-start)
# # # load results
# with open('Data/Processed/test_adult_SIS.pkl','rb') as f:
#     results = pickle.load(f)
# fig, axes = plt.subplots(4,2,figsize=(6.5,8.5))
# z_values = [["child infections","under-five infections"],
# ["peak incidence","rebound peak incidence"],
# ["periodicity","time to rebound"],
# ["pre-lockdown susceptibility","post-lockdown susceptibility"]]
# titles = [["Child-caused\ninfections","Under-five-caused\ninfections"],
# ["Peak incidence","Rebound peak"],
# ["Periodicity","Time to rebound"],
# ["Pre-lockdown\nsusceptibility","Post-lockdown\nsusceptibility"]]
# cbar_labels = [["","Infections"],
# ["","Observed infections"],
# ["","Years"],
# ["","Susceptibility"]]
# for i in range(4):
#     for j in range(2):
#         grid_plot(axes[i,j],results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE=OBS_AGE,
#         grid_params=(("BETA","REC_UP"),("S_REL",)),grid_mode=("scale","fade_vec"),factors=(1,1),
#         label_mode=("diff_mean","fade_vec"),
#         z_value=z_values[i][j],z_label=cbar_labels[i][j])
#         axes[i,j].set_title(titles[i][j])
#         if i == 3:
#             axes[i,j].set_xlabel("Immunity")
#         if j == 0:
#             axes[i,j].set_ylabel("Growth rate")
# plt.tight_layout()
# plt.show()