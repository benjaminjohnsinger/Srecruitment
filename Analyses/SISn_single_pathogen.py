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

from vaccination import birth_vax, all_vax
import contact_model as cm
from SISn_ODEs import single_pathogen_deltas_unjit as deltas_unjit
from SISn_ODEs import single_pathogen_deltas as sis_deltas
from Parameters.test_population import *
from Parameters.generic_disease import *

from utils import *
from demography import *
from mobility_and_import import *
from clustering import *
from sim_grid import *
from plotting import *
from fit_MCMC import *

## Period of simulation
EPOCH = pd.to_datetime('1970-01-01')
START = pd.to_datetime('2016-01-01')
END = pd.to_datetime('2024-01-01')
PERIOD = pd.date_range(start=START, end=END, freq='MS')

## Contacts and force of infection
IMPORT_RATE = 1e-5*np.ones(N_S)
# Contact matrix for all contact types
CONTACT = np.genfromtxt('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', dtype=np.float64)
# Lockdown and other mobility changes
T_LOCKDOWN = date_to_t('2020-03-01')
LOCKDOWN_DURATION = 365
LOCKDOWN_REDUCTION = 0.4
@jit
def contact(t,seasonality,offset):
    return cm.STEP(t,T_LOCKDOWN,LOCKDOWN_DURATION,LOCKDOWN_REDUCTION)*(1+seasonality*np.cos(2*np.pi*(t/365-offset)))*CONTACT

## Initial conditions
STATE0 = np.zeros((2*N_S+2)*NAG)
STATE0[NAG:2*NAG] = KP_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

STATE1 = np.copy(STATE0) 
STATE1[NAG:2*NAG] = KP_AGE_POP-1

## Integrate the system
# POINTS = np.concat((np.zeros(1),np.arange(T_LOCKDOWN-12*12,T_LOCKDOWN+LOCKDOWN_DURATION+12*12,0.2),np.ones(1)*PERIOD))
# POINTS = np.array([date_to_t('1960-01-01') + pd.DateOffset(months=x) for x in range(PERIOD)])
T_VAX = date_to_t('2035-01-01')

POINTS = np.array(date_to_t(PERIOD))

# Parameters for the ODE
params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV, 'T_VAX': T_VAX,
'arrivals': arrivals, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
'contact': contact}

# start = time.time()
# z=deltas_unjit(0,STATE0,params)
# end = time.time()
# print("unjitted: ",end-start)
# start = time.time()
# y=sis_deltas(0,STATE0,*params.values())
# end = time.time()
# print("jitted: ",end-start)
# STATE0[12] += 1
# start = time.time()
# z=deltas_unjit(1,STATE0,params)
# end = time.time()
# print("reunjitted: ",end-start)
# start = time.time()
# x=sis_deltas(1,STATE0,*params.values())
# end = time.time()
# print("rejitted: ",end-start)

params['SEASONALITY'] = 0.06
# params['BETA'] = 0.14
# params['S_REL'] = np.array([1,0.7,0.4])

with open('Data/Processed/SIS_noisy_obs_BETAp15_SEASp06_IMMp4.pickle','rb') as f:
    case_data = pickle.load(f)

np.random.seed(241108)
priors_tanh = {'BETA': sp.stats.norm(-1,1),'S_REL1': sp.stats.norm(-1,1),'S_REL2': sp.stats.norm(-1,1)}
proposal_widths = {'BETA': 0.01,'S_REL1': 0.005,'S_REL2': 0.005}
mcmc_trajectory, acceptance_rate = mcmc(case_data, params, POINTS, STATE0, OBS_AGE, SIS_likelihood, ['BETA','S_REL1','S_REL2'], priors_tanh, proposal_widths, 1000)
print(acceptance_rate)
with open('Data/Processed/mcmc_trajectory.pickle','wb') as f:
    pickle.dump(mcmc_trajectory,f)

# # burn_in = 5000

# plt.plot(mcmc_trajectory)
# plt.show()

# figure = corner.corner(mcmc_trajectory[burn_in:],labels=['Transmissibility','Seasonality','Immunity'])
# plt.savefig('Figures/SIS_MCMC_corner.png',dpi=300)

# fig, axes = plt.subplots(2,2,figsize=(6.5,6.5))
# axes[0,0].hist2d(mcmc_trajectory[burn_in:,0],mcmc_trajectory[burn_in:,1],bins=20)
# # outline the cell corresponding to the true values - 0.06 and 0.15 - in red
# axes[0,0].plot([0.15,0.15,0.1501,0.1501,0.15],[0.05995,0.06005,0.06005,0.05995,0.05995],color='red')
# axes[0,0].set_xlabel('Transmissibility')
# axes[0,0].set_ylabel('Seasonality')
# axes[0,1].hist2d(mcmc_trajectory[burn_in:,0],mcmc_trajectory[burn_in:,2],bins=20)
# axes[0,1].plot([0.15,0.15,0.1501,0.1501,0.15],[0.39995,0.40005,0.40005,0.39995,0.39995],color='red')
# axes[0,1].set_xlabel('Transmissibility')
# axes[0,1].set_ylabel('Immunity')
# axes[1,0].hist2d(mcmc_trajectory[burn_in:,1],mcmc_trajectory[burn_in:,2],bins=20)
# axes[1,0].plot([0.05995,0.05995,0.06005,0.06005,0.05995],[0.39995,0.40005,0.40005,0.39995,0.39995],color='red')
# axes[1,0].set_xlabel('Seasonality')
# axes[1,0].set_ylabel('Immunity')
# axes[1,1].plot(mcmc_trajectory[:,0],label='Transmissibility')
# axes[1,1].plot(mcmc_trajectory[:,1],label='Seasonality')
# axes[1,1].plot(mcmc_trajectory[:,2],label='Immunity')
# axes[1,1].legend()
# axes[1,1].set_xlabel('Iteration')
# axes[1,1].set_ylabel('Parameter value')
# plt.tight_layout()
# plt.show()

# # using optimizer, find initial age distribution that leads to age distribuiton matching KP_AGE_POP after simulating through to 2020 census (april 1)
# params['BETA'] = 0
# def age_diff(age_pop):
#     STATE0 = np.zeros((2*N_S+2)*NAG)
#     STATE0[NAG:2*NAG] = age_pop
#     result = sp.integrate.solve_ivp(sis_deltas,(POINTS[0],POINTS[-1]),STATE0,args=(params,),t_eval=POINTS,method='RK45')
#     final_age_pop = result.y[NAG:2*NAG,-1]
#     return np.sum(np.square(final_age_pop-KP_AGE_POP))

# res = sp.optimize.minimize(age_diff,KP_AGE_POP,bounds=[(0,4e7)]*NAG)
# # save results
# with open('Data/Processed/inital_age_optimize_result.pickle','wb') as f:
#     pickle.dump(res,f)
# print(res)
# print(res.x)

# #### One-shot line plot #####
# params['SEASONALITY'] = 0.06
# result0 = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params.values(),t_eval=POINTS,method='RK45')
# params['BETA'] = 0.06
# params['S_REL'] = np.array([1,0.74,0.5])
# result = sp.integrate.solve_ivp(deltas_unjit,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=(params,),t_eval=POINTS,method='RK45')
# params['BETA'] = 0.15
# params['S_REL'] = np.array([1,0.6,0.2])
# result2 = sp.integrate.solve_ivp(deltas_unjit,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=(params,),t_eval=POINTS,method='RK45')


# # print(result)
# # params['contact'] = lambda t, seasonality, offset : contact(t,shape_ramp,seasonality,offset)
# # result_ramp = sp.integrate.solve_ivp(sis_deltas,(0,PERIOD),STATE0,args=(params,),t_eval=POINTS,method='RK45')
# obs_full = observations(result,params,OBS_AGE,incidence=False)
# obs = np.sum(obs_full,axis=1)
# obs_noisy = np.random.poisson(obs)
# # incidence = obs_noisy/np.sum(result.y,axis=0)
# # save noisy incidence
# with open('Data/Processed/SIS_noisy_obs_BETAp15_SEASp06_IMMp4.pickle','wb') as f:
#     pickle.dump(obs_noisy,f)

# # # pre_obs = obs[(result.t>T_LOCKDOWN-12*12) & (result.t<T_LOCKDOWN)]
# # # corr = np.correlate(pre_obs, pre_obs, mode='same')
# # # acorr = corr[len(pre_obs)//2 + 1:] / (pre_obs.var() * np.arange(len(pre_obs)-1, len(pre_obs)//2, -1))
# # # acorr = acorr + np.linspace(0.1, 0, len(acorr))
# # # lag = np.abs(acorr).argmax() + 1
# # # print(lag/12)
# # # mx = np.max(obs[(result.t>T_LOCKDOWN-12*12) & (result.t<T_LOCKDOWN)])
# # # plt.plot(result.t,np.sum(result.y,axis=0))
# # # fig, ax = plt.subplots(2,2,figsize=(10,5.6))
# # # sum of each age group
# # # print(np.array([np.sum(result.y[range(i,7*NAG,NAG),:],axis=0) for i in range(NAG)]))
# # # print(np.sum(np.abs(KP_AGE_POP-np.array([np.sum(result.y[range(i,7*NAG,NAG),-1]) for i in range(NAG)]))))
# # # plt.show()
# # # plt.xlim(T_LOCKDOWN-12*12,T_LOCKDOWN)
# # # plt.xticks(np.arange(T_LOCKDOWN-12*12,T_LOCKDOWN+1,12),np.arange(0,13))
# # # plt.ylim(0,1.1*mx)
# fig, ax = plt.subplots(1,1,figsize=(10,5.6))
# mx0 = lockdown_incidence_plot(ax,STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result0,window=2*365,color='black')
# # mx = lockdown_incidence_plot(ax,STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,window=2*365)
# # mx2 = lockdown_incidence_plot(ax,STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result2,window=2*365,color='#FF832B')
# lockdown_incidence_format(ax,T_LOCKDOWN,LOCKDOWN_DURATION,mx0,year_window=2)
# # plt.savefig('Figures/noisy_trajectory_BETAp15_SEASp06_IMMp4.png',dpi=300)
# plt.show()
# params['WANE'] = 1/12*np.array([0.0,1.0,0.0])
# params['BETA'] = 30

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