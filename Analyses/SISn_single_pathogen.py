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
POINTS = np.concat((np.zeros(1),np.arange(T_LOCKDOWN-5*12,T_LOCKDOWN+LOCKDOWN_DURATION+5*12,0.1),np.ones(1)*PERIOD))
# POINTS = np.arange(0,PERIOD+1,1)
T_VAX = PERIOD

# Parameters for the ODE
params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': BIRTH_RATE, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV, 'T_VAX': T_VAX,
'IMPORT': IMPORT, 'BETA': BETA, 'contact':lambda t : contact(t,shape_step)}
result = sp.integrate.solve_ivp(deltass,(0,PERIOD),STATE0,args=(params,),t_eval=POINTS,method='RK45')

## four line plots with different values of aquired immunity, showing incidence and susceptibility
fig, ax = plt.subplots(2,1,figsize=(6.5,6.5))
mx = np.zeros(4)
colors = ['#648FFF', '#DC267F', '#785EF0', '#FFB000']
# Plot the incidence
for i in range(4):
    params['BETA'] = 30+i*20
    result = sp.integrate.solve_ivp(deltass,(0,PERIOD),STATE0,args=(params,),t_eval=POINTS,method='RK45')
    mx[i] = lockdown_incidence_plot(ax[0],STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,label="Infectiousness "+f'{30+i*20:.2f}'+r" $m^{-1}$",color=colors[i])
    lockdown_susceptibility_plot(ax[1],STATE0,params,PERIOD,POINTS,T_LOCKDOWN,result=result,label="Infectiousness "+f'{30+i*20:.2f}'+r" $m^{-1}$",color=colors[i],relative=False)

lockdown_incidence_format(ax[0],POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,max(mx))
lockdown_susceptibility_format(ax[1],POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,ymax=None)
ax[0].set_ylabel("Observed incidence")
ax[0].legend()
plt.tight_layout()
plt.savefig('Figures/SISn_beta.png',dpi=300)

# # results = sim_grid(STATE0,params,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,grid_params=(("BETA",),("REC_UP","REC_SAME")),N=10,factors=(2,2))
# # # Save the results
# # with open('Data/Processed/SISn_beta_rec2.pickle','wb') as f:
# #     pickle.dump(results,f)
# # Load the results
# with open('Data/Processed/SISn_beta_rec2.pickle','rb') as f:
#     results = pickle.load(f)

# fig, ax = plt.subplots(2,1,figsize=(6.5,6.5))
# grid_plot(ax[0],results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE,
# grid_params=(("BETA",),("REC_UP","REC_SAME")),label_mode=("mean","nz_mean"),
# z_value="oscillation size",z_label="Oscillation size")
# grid_plot(ax[1],results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE,
# grid_params=(("BETA",),("REC_UP","REC_SAME")),label_mode=("mean","nz_mean"),
# z_value="peak incidence",z_label="Peak observed incidence")
# plt.show()