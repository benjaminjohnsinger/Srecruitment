## SIS model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import time
import itertools as it
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm as pltcm
import pickle

from vaccination import birth_vax, all_vax
import contact_model as cm
from SISn_ODEs import single_pathogen_deltas as sis_deltas
from Parameters.test_population import *
from Parameters.generic_disease import *

from clustering import *
from plotting import *

## Period of simulation in months
PERIOD = 12*50

## Contacts and force of infection
IMPORT = 0.01*np.ones(N_S)
# Contact matrix for all contact types
CONTACT = np.genfromtxt('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',')
CONTACT /= 12.99 # transform to contact proportions
# Lockdown and other mobility changes
T_LOCKDOWN = 37*12
LOCKDOWN_DURATION = 12
LOCKDOWN_REDUCTION = 0.4
shape_step = lambda t : cm.STEP(t,T_LOCKDOWN,LOCKDOWN_DURATION,LOCKDOWN_REDUCTION)
shape_static = lambda t : 1
def contact(t,shape,seasonality=SEASONALITY,offset=OFFSET,c_rate=CONTACT):
    return shape(t)*(1+seasonality*np.cos(2*np.pi*(t/12-offset)))*c_rate

## Initial conditions
STATE0 = np.zeros((2*N_S+2)*NAG)
STATE0[NAG:2*NAG] = KP_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

## Integrate the system
# POINTS = np.concat((np.zeros(1),np.arange(T_LOCKDOWN-12*12,T_LOCKDOWN+LOCKDOWN_DURATION+12*12,0.2),np.ones(1)*PERIOD))
POINTS = np.arange(0,PERIOD+1,1)
T_VAX = PERIOD

# Parameters for the ODE
params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': BIRTH_RATE, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV, 'T_VAX': T_VAX,
'IMPORT': IMPORT, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
'contact':lambda t, seasonality, offset : contact(t,shape_step,seasonality,offset)}

params['WANE'] = 1/12*np.array([0.0,1.0,0.0])
params['BETA'] = 30

with open('Data/Processed/SIS_3D.pickle','rb') as f:
    results = pickle.load(f)
with open('Data/Processed/SIS_3D_obs.pickle','rb') as f:
    obses = pickle.load(f)

######## Plotting age-based clusters ########
# ages = np.array([np.mean(age_of_first_infection(results[key],MEDIAN_AGE)[np.argmax(results[key].t>=T_LOCKDOWN-5*12):np.argmax(results[key].t>T_LOCKDOWN)]) for key in results.keys()])
# print(np.sum(ages<=3))
# ages_label = np.array([int(age>3) + int(age>12) + int(age>5*12) + int(age>18*12) + int(age>40*12) + int(age>65*12) for age in ages])
# n_age_clusters = max(ages_label)+1

# fig,axes = plt.subplots(4,4,figsize=(6.5,1.7*4),sharex='col',layout='constrained',squeeze=False)
# for row in range(4):
#     axes[row,1].sharey(axes[row,2])
#     axes[row,2].sharey(axes[row,3])
# cluster_plot(axes,results,obses,n_age_clusters,ages_label,None,color=False)
# fig.align_ylabels()
# plt.savefig('Figures/SIS_3D_age_clusters.png',dpi=500)

######## Computing observations ########
# with open('Data/Processed/SIS_3D.pickle','rb') as f:
#     results = pickle.load(f)
# obses = {}
# for key,result in results.items():
#     print(key)
#     obs = observations(result,params,OBS_AGE,incidence=True)
#     obses[key] = obs
# with open('Data/Processed/SIS_3D_obs.pickle','wb') as f:
#     pickle.dump(obses,f)

######## Plotting clusters ########
# with open('Data/Processed/SIS_3D.pickle','rb') as f:
#     results = pickle.load(f)
# with open('Data/Processed/SIS_3D_obs.pickle','rb') as f:
#     obses = pickle.load(f)

# ## plot six clusters
# model = cluster_sims(results,obses,T_LOCKDOWN,6)
# fig, axes = plt.subplots(6,4,figsize=(6.5,1.7*4),sharex='col',layout='constrained',squeeze=False)
# for row in range(6):
#     axes[row,1].sharey(axes[row,2])
#     axes[row,2].sharey(axes[row,3])
# cluster_plot(axes,results,obses,model.n_clusters,model.labels_,model.cluster_centers_,color=True)
# fig.align_ylabels()
# plt.savefig('Figures/SIS_3D_6clusters_color.png',dpi=500)

### plot all sims and select clusters
model = cluster_sims(results,obses,T_LOCKDOWN,6)
model1 = cluster_sims(results,obses,T_LOCKDOWN,1)
fig, axes = plt.subplots(4,4,figsize=(6.5,1.7*4),sharex='col',layout='constrained',squeeze=False)
for row in range(4):
    axes[row,1].sharey(axes[row,2])
    axes[row,2].sharey(axes[row,3])
first_row = axes[0,:]
first_row.shape = (1,4)
cluster_colors = np.array([["#648FFF", "#DC267F", "#785EF0", "#FFB000", "#FF832B", "#000000", "#FFD662", "#FF34FF", "#8B0000", "#00FF00"][i] for i in model.labels_])
cluster_plot(first_row,results,obses,model1.n_clusters,model1.labels_,None,color=True,color_values_all=cluster_colors)
cluster_plot(axes[1:,:],results,obses,model.n_clusters,model.labels_,model.cluster_centers_,color=True,clusters=[0,2,4],color_values_all=cluster_colors)
fig.align_ylabels()
plt.savefig('Figures/SIS_3D_clusters_1plus3of6_cluster_color.png',dpi=900)

### Computing observations for a given cluster
# with open('Data/Processed/SIS_3D.pickle','rb') as f:
#     results = pickle.load(f)
# # separate out results and observations from cluster 4 and save
# cluster4_results = {}
# cluster4_obses = {}
# for key,result in results.items():
#     if model.labels_[list(results.keys()).index(key)] == 3:
#         cluster4_results[key] = result
#         cluster4_obses[key] = obses[key]
# with open('Data/Processed/SIS_3D_cluster4_results.pickle','wb') as f:
#     pickle.dump(cluster4_results,f)
# with open('Data/Processed/SIS_3D_cluster4_obses.pickle','wb') as f:
#     pickle.dump(cluster4_obses,f)

# ## plot sub-clusters of cluster 4
# ## load cluster 4 and divide into 4 more clusters
# with open('Data/Processed/SIS_3D_cluster4_results.pickle','rb') as f:
#     results = pickle.load(f)
# with open('Data/Processed/SIS_3D_cluster4_obses.pickle','rb') as f:
#     obses = pickle.load(f)

# n_clusters = 6
# model = cluster_sims(results,obses,T_LOCKDOWN,n_clusters)
# fig, axes = plt.subplots(n_clusters,4,figsize=(6.5,1.7*n_clusters),sharex='col',layout='constrained',squeeze=False)
# for row in range(n_clusters):
#     axes[row,1].sharey(axes[row,2])
#     axes[row,2].sharey(axes[row,3])
# cluster_plot(axes,results,obses,model.n_clusters,model.labels_,model.cluster_centers_,color=True)
# fig.align_ylabels()
# plt.savefig('Figures/SIS_3D_'+str(n_clusters)+'clusters_of_cluster4of6_color.png',dpi=300)

# ##### One-shot line plot #####
# params['BETA'] = 44
# params['SEASONALITY'] = 0.02
# params['S_REL'] = np.array([1,0.95,0.9])
# result = sp.integrate.solve_ivp(sis_deltas,(0,PERIOD),STATE0,args=(params,),t_eval=POINTS,method='RK45')
# # obs = observations(result,params,OBS_AGE,incidence=True)
# # pre_obs = obs[(result.t>T_LOCKDOWN-12*12) & (result.t<T_LOCKDOWN)]
# # corr = np.correlate(pre_obs, pre_obs, mode='same')
# # acorr = corr[len(pre_obs)//2 + 1:] / (pre_obs.var() * np.arange(len(pre_obs)-1, len(pre_obs)//2, -1))
# # acorr = acorr + np.linspace(0.1, 0, len(acorr))
# # lag = np.abs(acorr).argmax() + 1
# # print(lag/12)
# # mx = np.max(obs[(result.t>T_LOCKDOWN-12*12) & (result.t<T_LOCKDOWN)])
# # plt.plot(result.t,obs)
# # plt.xlim(T_LOCKDOWN-12*12,T_LOCKDOWN)
# # plt.xticks(np.arange(T_LOCKDOWN-12*12,T_LOCKDOWN+1,12),np.arange(0,13))
# # plt.ylim(0,1.1*mx)
# fig, ax = plt.subplots(1,1,figsize=(6.5,4.5))
# mx = lockdown_incidence_plot(ax,STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result)
# lockdown_incidence_format(ax,T_LOCKDOWN,LOCKDOWN_DURATION,mx,year_window=10)
# plt.show()

#### line plots with different parameter values, showing incidence and susceptibility #####
# params['BETA'] = 41
# params['SEASONALITY'] = 0.061
# params['S_REL'] = np.array([1,0.55,0.1])
# fig, ax = plt.subplots(2,1,figsize=(6.5,6.5))
# mx = np.zeros(4)
# colors = ['#648FFF', '#DC267F', '#785EF0', '#FFB000']
# # Plot the incidence
# for i in range(2):
#     p_value = 0.06+0.02*i
#     params['WANE'] = p_value*np.array([0.0,1.0,0.0])
#     result = sp.integrate.solve_ivp(sis_deltas,(0,PERIOD),STATE0,args=(params,),t_eval=POINTS,method='RK45')
#     # plot each susceptible compartment
# #     ax[i//2,i%2].plot(result.t[1:],np.sum(result.y[NAG:2*NAG,:],axis=0)[1:],label='S1',color=colors[0])
# #     ax[i//2,i%2].plot(result.t[1:],np.sum(result.y[3*NAG:4*NAG,:],axis=0)[1:],label='S2',color=colors[1])
# #     ax[i//2,i%2].plot(result.t[1:],np.sum(result.y[5*NAG:6*NAG,:],axis=0)[1:],label='S3',color=colors[2])
# #     ax[i//2,i%2].set_title(f"Aquired immunity: {0.1*(i+2):.2f}")
# # ax[0,0].legend()
# # plt.show()
#     obs = observations(result,params,OBS_AGE,incidence=True)
#     pre_obs = obs[(result.t>T_LOCKDOWN-12*12) & (result.t<T_LOCKDOWN)]
#     corr = np.correlate(pre_obs, pre_obs, mode='same')
#     acorr = corr[len(pre_obs)//2 + 1:] / (pre_obs.var() * np.arange(len(pre_obs)-1, len(pre_obs)//2, -1))
#     acorr = acorr + np.linspace(0.1, 0, len(acorr))
#     lag = np.abs(acorr).argmax() + 1
#     print(lag)
#     mx[i] = lockdown_incidence_plot(ax[0],STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,label=f'{(1/12)/p_value:.2f} y',color=colors[i],relative=False,obs=obs)
#     lockdown_susceptibility_plot(ax[1],STATE0,params,PERIOD,POINTS,T_LOCKDOWN,result=result,label=f'{(1/12)/p_value:.2f} y',color=colors[i],relative=False)

# lockdown_incidence_format(ax[0],T_LOCKDOWN,LOCKDOWN_DURATION,max(mx),year_window=10)
# lockdown_susceptibility_format(ax[1],T_LOCKDOWN,LOCKDOWN_DURATION,ymax=None,year_window=10)
# ax[0].set_ylabel("Observed incidence")
# # ax[0].set_ylim(0,1.1)
# ax[1].set_ylabel("Effective population susceptibility")
# ax[1].set_ylim(1.5e7,2.2e7)
# ax[1].legend(title=r"Waning time")
# plt.tight_layout()
# plt.savefig('Figures/SIS_beta41_seasonality0p061_acqimm0p45_vary_wane.png',dpi=300)

##### run multi-dimensional grid sims #####
# start = time.time()
# results = sim_grid(STATE0,params,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,
# grid_params=(("BETA",),("WANE",),("S_REL",),("SEASONALITY",)),
# N=10,factors=(1,1,1,2),grid_mode=("scale","scale","fade_vec","scale"))
# print(f"Simulation took {time.time()-start:.2f} seconds")
# # Save the results
# with open('Data/Processed/SIS_4D.pickle','wb') as f:
#     pickle.dump(results,f)

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