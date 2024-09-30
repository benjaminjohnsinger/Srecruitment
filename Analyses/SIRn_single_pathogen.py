## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import itertools as it

from vaccination import birth_vax, all_vax
import contact_model as cm
from SIRn_ODEs import single_pathogen_deltas as deltas
from Parameters.test_population import *
from Parameters.child_disease import *

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

N=50
pre_peaks = np.zeros((N,N))
post_peaks = np.zeros((N,N))

for beta_n in range(N):
    for rec_n in range(N):
        sim_beta = BETA*(1+(1/N)*(beta_n-N/2))
        sim_rec = REC*(1+(1/N)*(rec_n-N/2))
        print(sim_beta,sim_rec)
        result = sp.integrate.solve_ivp(deltas, [0,PERIOD], STATE0, method='RK45', t_eval=np.linspace(0,PERIOD,POINTS),
            args=((NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, sim_rec, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV_SCALED, BCOV, T_VAX, IMPORT, sim_beta,
            lambda t : contact(t, shape_step,SEASONALITY,OFFSET,T_FACTOR,CONTACT)),))
        ## Calculate observations
        obs = np.zeros((len(result.t),NAG))
        pop_size = np.sum(result.y,axis=0)
        for i_t,t in enumerate(result.t):
            foi = sim_beta*np.dot(contact(t,shape_step),np.sum((np.array([result.y[(3*j+2)*NAG:(3*j+3)*NAG,i_t] for j in range(N_S)])*I_REL),axis=0))/pop_size[i_t]
            for i in range(N_S):
                obs[i_t,:] += OBS_AGE*P_OBS[i]*S_REL[i]*foi*result.y[(3*i+1)*NAG:(3*i+2)*NAG,i_t]*T_FACTOR
        if len(result.t) != POINTS:
            pre_peaks[beta_n,rec_n] = np.nan
            post_peaks[beta_n,rec_n] = np.nan
            continue
        else:
            pre_peaks[beta_n,rec_n] = np.max(np.sum(obs[(result.t>30*12*T_FACTOR) & (result.t<T_LOCKDOWN)],axis=1)/pop_size[(result.t>30*12*T_FACTOR) & (result.t<T_LOCKDOWN)])
            post_peaks[beta_n,rec_n] = np.max(np.sum(obs[result.t>=T_LOCKDOWN],axis=1)/pop_size[result.t>=T_LOCKDOWN])

## Plot the results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1,2,figsize=(10,5))

plt.sca(axes[0])
plt.imshow(pre_peaks)
plt.colorbar()
plt.xlabel('Recovery rate')
plt.ylabel('Infectiousness')
plt.title('Pre-lockdown peak incidence')
plt.xticks(range(0,N,5),[f'{np.mean(REC)*(1+(1/N)*(rec_n-N/2)):.1f}' for rec_n in range(0,N,5)])
plt.yticks(range(0,N,5),[f'{BETA*(1+(1/N)*(beta_n-N/2)):.1f}' for beta_n in range(0,N,5)])
for tick in axes[0].get_xticklabels():
    tick.set_rotation(45)

plt.sca(axes[1])
plt.imshow(post_peaks)
plt.colorbar()
plt.xlabel('Recovery rate')
plt.ylabel('Infectiousness')
plt.title('Post-lockdown peak incidence')
plt.xticks(range(0,N,5),[f'{np.mean(REC)*(1+(1/N)*(rec_n-N/2)):.1f}' for rec_n in range(0,N,5)])
plt.yticks(range(0,N,5),[f'{BETA*(1+(1/N)*(beta_n-N/2)):.1f}' for beta_n in range(0,N,5)])
for tick in axes[0].get_xticklabels():
    tick.set_rotation(45)

plt.tight_layout()
plt.savefig('Figures/child_peaks2.png')

# viridis = plt.cm.get_cmap('viridis', NAG)

# # plt.plot(result_static.t,np.sum(obs_static,axis=1)/pop_size, label='Observed cases',color='blue')
# plt.plot(result.t,np.sum(obs,axis=1)/pop_size, label='Observed cases',color='red')
# plt.xlim(33*12,33*12+10*12)
# # plt.ylim(0,7e-4)
# mx = 1.1*np.max((np.sum(obs,axis=1)/pop_size)[int(2*POINTS/3):POINTS])
# plt.ylim(0,mx)
# plt.ylabel('Observed incidence')
# plt.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],0,mx,color='gray',alpha=0.2)
# plt.xticks(np.arange(33*12,33*12+11*12,12),[str(int(x)-5) for x in np.arange(0,11,1)])
# plt.xlabel('Time (years)')
# plt.title('Incidence of non-pediatric disease with 1-year lockdown')
# plt.tight_layout()
# # plt.savefig('Figures/flu_lockdown_w_imports_compare.svg')
# plt.show()

# # fig, axes = plt.subplots(2,2,figsize=(6.5,6.5))
# # axes[0,0].plot(result.t,np.sum(obs,axis=1)/pop_size, label='Observed cases')
# # axes[0,0].set_title('Simulation (including burn-in)')
# # axes[0,0].set_ylabel('Observed incidence')
# # axes[0,1].plot(result.t,np.sum(obs,axis=1)/pop_size, label='Observed cases')
# # axes[0,1].set_xlim(2*PERIOD/3,PERIOD)
# # axes[0,1].set_ylim(0,1e-4)
# # # axes[0,1].set_ylim(0,4e-4)
# # axes[0,1].set_title('Detail after burn-in')
# # axes[0,1].set_ylabel('Observed incidence')
# # # axes[1,0].plot(result.t,pop_size, label='Population size')
# # # axes[1,0].set_title('Total population')
# # # axes[1,0].set_ylabel('Persons')

# # # Infected persons in each age group
# # for i in range(NAG):
# #     axes[1,0].plot(result.t,obs[:,i]/np.sum(result.y[range(i,10*NAG,NAG),:],axis=0), label=AGE_GROUP_NAMES[i], color=viridis(i), alpha=0.5)
# # axes[1,0].set_xlim(2*PERIOD/3,PERIOD)
# # axes[1,0].set_ylim(0,5e-4)
# # # axes[1,0].set_ylim(0,1e-3)
# # axes[1,0].set_title('Incidence by age group')
# # axes[1,0].set_ylabel('Observed incidence')

# # # Population in each age group
# # for i in range(NAG):
# #     axes[1,1].plot(result.t,np.sum(result.y[range(i,10*NAG,NAG),:],axis=0), label=AGE_GROUP_NAMES[i], color=viridis(i))
# # axes[1,1].set_title('Age groups')
# # axes[1,1].set_ylabel('Age group population')


# # plt.tight_layout()
# # # plt.savefig('Figures/SIR3_rotalike_demo_yearly.png')
# # plt.show()