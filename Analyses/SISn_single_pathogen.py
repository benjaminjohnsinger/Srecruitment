## SIS model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import time
import itertools as it
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle

from vaccination import birth_vax, all_vax
import contact_model as cm
from SISn_ODEs import single_pathogen_deltas as deltass
from Parameters.test_population import *
from Parameters.generic_disease import *

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
POINTS = np.concat((np.zeros(1),np.arange(T_LOCKDOWN-5*12,T_LOCKDOWN+LOCKDOWN_DURATION+5*12,1),np.ones(1)*PERIOD))
# POINTS = np.arange(0,PERIOD+1,1)
T_VAX = PERIOD

# Parameters for the ODE
params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': BIRTH_RATE, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV, 'T_VAX': T_VAX,
'IMPORT': IMPORT, 'BETA': BETA, 'contact':lambda t : contact(t,shape_step)}

# with open('Data/Processed/SIS_6D.pickle','rb') as f:
#     results = pickle.load(f)
# obses = {}
# for key,result in results.items():
#     print(key)
#     obs = observations(result,params,OBS_AGE,incidence=True)
#     obses[key] = obs
# print(obses)
# with open('Data/Processed/SIS_6D_obs.pickle','wb') as f:
#     pickle.dump(obses,f)

# result = sp.integrate.solve_ivp(deltass,(0,PERIOD),STATE0,args=(params,),t_eval=POINTS,method='RK45')
# obs = observations(result,params,OBS_AGE,incidence=True)
# mx=lockdown_incidence_plot(plt.gca(),STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result)
# lockdown_incidence_format(plt.gca(),T_LOCKDOWN,LOCKDOWN_DURATION,mx)
# plt.show()

## four line plots with different values of aquired immunity, showing incidence and susceptibility
fig, ax = plt.subplots(2,1,figsize=(6.5,6.5))
mx = np.zeros(4)
colors = ['#648FFF', '#DC267F', '#785EF0', '#FFB000']
# Plot the incidence
params['BETA'] = 50
for i in range(4):
    params['S_REL'] = np.linspace(1,(1-(N_S-1)*0.1*(i+2)),N_S)
    result = sp.integrate.solve_ivp(deltass,(0,PERIOD),STATE0,args=(params,),t_eval=POINTS,method='RK45')
    # plot each susceptible compartment
#     ax[i//2,i%2].plot(result.t[1:],np.sum(result.y[NAG:2*NAG,:],axis=0)[1:],label='S1',color=colors[0])
#     ax[i//2,i%2].plot(result.t[1:],np.sum(result.y[3*NAG:4*NAG,:],axis=0)[1:],label='S2',color=colors[1])
#     ax[i//2,i%2].plot(result.t[1:],np.sum(result.y[5*NAG:6*NAG,:],axis=0)[1:],label='S3',color=colors[2])
#     ax[i//2,i%2].set_title(f"Aquired immunity: {0.1*(i+2):.2f}")
# ax[0,0].legend()
# plt.show()
    mx[i] = lockdown_incidence_plot(ax[0],STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,label=f'{0.1*(i+2):.2f}',color=colors[i],relative=True)
    lockdown_susceptibility_plot(ax[1],STATE0,params,PERIOD,POINTS,T_LOCKDOWN,result=result,label=f'{0.1*(i+2):.2f}',color=colors[i],relative=True)

lockdown_incidence_format(ax[0],T_LOCKDOWN,LOCKDOWN_DURATION,max(mx))
lockdown_susceptibility_format(ax[1],T_LOCKDOWN,LOCKDOWN_DURATION,ymax=None)
ax[0].set_ylabel("Incidence relative to\npre-lockdown peak")
ax[1].set_ylabel("Relative susceptibility")
ax[1].set_ylim(0.95,1.25)
ax[0].legend(title=r"Aquired immunity")
plt.tight_layout()
plt.savefig('Figures/SIS_acqimm_relative_beta_boost.png',dpi=300)

# start = time.time()
# results = sim_grid(STATE0,params,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,
# grid_params=(("BETA",),("REC_UP","REC_SAME"),("WANE",),("S_REL",),("S_AGE",),("IMPORT",)),
# N=5,factors=(1,1,1,1,1,1,1),grid_mode=("scale","scale","scale","fade_vec","fade_vec","scale"))
# print(f"Simulation took {time.time()-start:.2f} seconds")
# # Save the results
# with open('Data/Processed/SIS_6D.pickle','wb') as f:
#     pickle.dump(results,f)
# Load the results
# with open('Data/Processed/SIS_6D.pickle','rb') as f:
#     results = pickle.load(f)

# fig, axes = plt.subplots(5,3,figsize=(8.27,11.69),layout='constrained')
# start = time.time()
# im = grid_plot(axes,results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE,
# grid_params=(("BETA",),("REC_UP","REC_SAME"),("WANE",),("S_REL",),("S_AGE",),("IMPORT",)),
# factors=(1,1,1,1,1,1,1),grid_mode=("scale","scale","scale","fade_vec","fade_vec","scale"),
# label_mode=("mean","mean","mean","fade_vec","fade_vec","mean"),
# x_labels=("Transmission","Recovery","Waning","Acquired immunity","Age immunity","Imported cases"),
# z_value="rebound peak incidence",z_label="")
# print(f"Plotting took {time.time()-start:.2f} seconds")
# # colorbar
# fig.colorbar(im, ax=axes, orientation='horizontal', label="Proportional rebound in peak incidence")
# # plt.tight_layout()
# # plt.show()
# plt.savefig('Figures/SIS_6D_rebound_proporitonal_fixup.png',dpi=300)

# fig, ax = plt.subplots(2,1,figsize=(6.5,6.5))
# grid_plot(ax[0],results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE,
# grid_params=(("BETA",),("REC_UP","REC_SAME")),label_mode=("mean","nz_mean"),
# z_value="oscillation size",z_label="Oscillation size")
# grid_plot(ax[1],results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE,
# grid_params=(("BETA",),("REC_UP","REC_SAME")),label_mode=("mean","nz_mean"),
# z_value="peak incidence",z_label="Peak observed incidence")
# plt.show()

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