## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import itertools as it
import matplotlib.pyplot as plt

from vaccination import birth_vax, all_vax
import contact_model as cm
from SIRn_ODEs import single_pathogen_deltas as deltas
from Parameters.test_population import *
from Parameters.child_disease import *

from plotting import *

## Period of simulation in months
T_FACTOR = 1
PERIOD = int(T_FACTOR*12*50)

# Correct parameters with T_FACTOR
WANE_UP /= T_FACTOR
WANE_SAME /= T_FACTOR
REC /= T_FACTOR
AGING_RATE /= T_FACTOR
BIRTH_RATE /= T_FACTOR
ACOV_SCALED = lambda t,T_VAX : ACOV(t,T_VAX)/T_FACTOR

## Contacts and force of infection
IMPORT = 0.01
# Contact matrix for all contact types
CONTACT = np.genfromtxt('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',')
CONTACT /= 12.99 # transform to contact proportions
# Lockdown and other mobility changes
T_LOCKDOWN = 37*12*T_FACTOR
LOCKDOWN_DURATION = 12*T_FACTOR
LOCKDOWN_REDUCTION = 0.4
shape_step = lambda t : cm.STEP(t,T_LOCKDOWN,LOCKDOWN_DURATION,LOCKDOWN_REDUCTION)
shape_static = lambda t : 1
def contact(t,shape,seasonality=SEASONALITY,offset=OFFSET,t_factor=T_FACTOR,contact=CONTACT):
    return shape(t)*(1+seasonality*np.cos(2*np.pi*(t/(12*t_factor)-offset)))*contact

## Initial conditions
STATE0 = np.zeros((1+N_S*3)*(NAG))
STATE0[NAG:2*NAG] = KP_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

## Integrate the system
POINTS = PERIOD
T_VAX = PERIOD

# I_REL = np.array([[1],[0.9],[0.8]])
# BETA = 70


fig,axes = plt.subplots(2,1,figsize=(6.5,6.5))

from Parameters.child_disease import *
args1=((NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV_SCALED, BCOV, T_VAX,
    IMPORT, BETA,lambda t : contact(t, shape_step,SEASONALITY,OFFSET,T_FACTOR,CONTACT)),)
result1 = sp.integrate.solve_ivp(deltas, [0,PERIOD], STATE0, method='RK45', t_eval=np.linspace(0,PERIOD,POINTS),args=args1)
mx1 = lockdown_incidence_plot(axes[0],STATE0,args1,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result1,label="Child disease",color='#648FFF')
lockdown_susceptibility_plot(axes[1],STATE0,args1,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result1,label="Child disease",color='#648FFF')

from Parameters.adult_disease import *
args2=((NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV_SCALED, BCOV, T_VAX,
    IMPORT, BETA,lambda t : contact(t, shape_step,SEASONALITY,OFFSET,T_FACTOR,CONTACT)),)
result2 = sp.integrate.solve_ivp(deltas, [0,PERIOD], STATE0, method='RK45', t_eval=np.linspace(0,PERIOD,POINTS),args=args2)
mx2 = lockdown_incidence_plot(axes[0],STATE0,args2,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result2,label="Adult disease",color='#DC267F')
lockdown_susceptibility_plot(axes[1],STATE0,args2,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result2,label="Adult disease",color='#DC267F')

lockdown_incidence_format(axes[0],POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,max(mx1,mx2))
lockdown_susceptibility_format(axes[1],POINTS,T_LOCKDOWN,LOCKDOWN_DURATION)

axes[0].legend()

plt.tight_layout()
plt.show()

# viridis = plt.cm.get_cmap('viridis', NAG)

# # plt.plot(result_static.t,np.sum(obs_static,axis=1)/pop_size, label='Observed cases',color='blue')
# # plt.plot(result.t,np.sum(obs,axis=1)/pop_size, label='Observed cases')
# plt.xlim(25*12,43*12)
# # plt.ylim(0,7e-4)
# mx = 1.1*np.max((np.sum(obs,axis=1)/pop_size)[int(2*POINTS/3):POINTS])
# # plt.ylim(mn,mx)
# plt.ylabel('Observed incidence')
# plt.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],0,mx,color='gray',alpha=0.2)
# plt.xticks(np.arange(25*12,44*12,12),[str(int(x)-13) for x in np.arange(0,19,1)])
# plt.xlabel('Time (years)')
# plt.title('Incidence of non-pediatric disease with 1-year lockdown')
# plt.tight_layout()
# # plt.savefig('Figures/rec_wane_timing240930/child_growth174_wane0p07.png',dpi=300)
# plt.show()

# fig, axes = plt.subplots(2,2,figsize=(6.5,6.5))
# axes[0,0].plot(result.t,np.sum(obs,axis=1)/pop_size, label='Observed cases')
# axes[0,0].set_title('Simulation (including burn-in)')
# axes[0,0].set_ylabel('Observed incidence')
# axes[0,1].plot(result.t,np.sum(obs,axis=1)/pop_size, label='Observed cases')
# axes[0,1].set_xlim(2*PERIOD/3,PERIOD)
# axes[0,1].set_ylim(0,1e-4)
# # axes[0,1].set_ylim(0,4e-4)
# axes[0,1].set_title('Detail after burn-in')
# axes[0,1].set_ylabel('Observed incidence')
# # axes[1,0].plot(result.t,pop_size, label='Population size')
# # axes[1,0].set_title('Total population')
# # axes[1,0].set_ylabel('Persons')

# # Infected persons in each age group
# for i in range(NAG):
#     axes[1,0].plot(result.t,obs[:,i]/np.sum(result.y[range(i,10*NAG,NAG),:],axis=0), label=AGE_GROUP_NAMES[i], color=viridis(i), alpha=0.5)
# axes[1,0].set_xlim(2*PERIOD/3,PERIOD)
# axes[1,0].set_ylim(0,5e-4)
# # axes[1,0].set_ylim(0,1e-3)
# axes[1,0].set_title('Incidence by age group')
# axes[1,0].set_ylabel('Observed incidence')

# # Population in each age group
# for i in range(NAG):
#     axes[1,1].plot(result.t,np.sum(result.y[range(i,10*NAG,NAG),:],axis=0), label=AGE_GROUP_NAMES[i], color=viridis(i))
# axes[1,1].set_title('Age groups')
# axes[1,1].set_ylabel('Age group population')


# plt.tight_layout()
# # plt.savefig('Figures/SIR3_rotalike_demo_yearly.png')
# plt.show()