## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import itertools as it
from vaccination import birth_vax, all_vax
from Parameters.influenza import *

## Period of simulation in months
T_FACTOR = 1
PERIOD = int(T_FACTOR*12*50)

## Population size and age distribution
POP_SIZE = 3.9e7 # California population in 2022
AGE_POP = np.genfromtxt('Data/Processed/US_Census_population_by_age.csv', delimiter=',') # US in 2022
KP_AGE_GROUPS = [
    range(3),
    range(3,12),
    range(12,5*12),
    range(5*12,18*12),
    range(18*12,40*12),
    range(40*12,65*12),
    range(65*12,100*12)]
AGE_GROUP_NAMES = ['Newborns','Infants','Young children','Older children','Young adults','Middle-aged adults','Older adults']
AGE_PROPORTION = np.array([np.sum(AGE_POP[group]) for group in KP_AGE_GROUPS])/np.sum(AGE_POP)
KP_AGE_POP = AGE_PROPORTION*POP_SIZE
NAG = len(KP_AGE_GROUPS)

# Rate of aging out of each age group. Last rate informed by US life expectancy at age 65.
AGING_RATE = 1/(T_FACTOR*np.array([3,9,4*12,13*12,22*12,25*12,18.35*12]))

# Birth rate (for California in 2022)
BIRTH_RATE = 3.99e5/(POP_SIZE*12*T_FACTOR)
# AGING_RATE = BIRTH_RATE/AGE_PROPORTION # Stable population distribution

# Correct parameters with T_FACTOR
WANE_UP /= T_FACTOR
WANE_SAME /= T_FACTOR
REC /= T_FACTOR
ACOV /= T_FACTOR

# Force of infection per contact
SEASONALITY = 0.05
OFFSET = 0.636*T_FACTOR
# Contact matrix for all contact types
CONTACT = np.genfromtxt('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',')
def BETA_PC(t):
    return (BETA_FUDGE_FACTOR*23.25*4.3/(12.99*T_FACTOR))*CONTACT
def INFECTIOUS_CONTACT(t):
    return BETA_PC(t)*(1+SEASONALITY*np.cos(2*np.pi*(t/(12*T_FACTOR)-OFFSET)))

## Initial conditions
STATE0 = np.zeros((1+N_S*3)*(NAG))
STATE0[NAG:2*NAG] = KP_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

## Differential equations
def deltas(t,state,params):
    delta = np.zeros(state.shape)
    pop_size = np.sum(state)
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, INFECTIOUS_CONTACT = params
    # Susceptible, infected, recovered - waning, aging, infection, recovery for all susceptibility classes
    for i in range(N_S):
        delta[(3*i+1)*NAG:(3*i+2)*NAG] = birth_vax(t,i,S_VAX,BCOV,T_VAX)*BIRTH_RATE*pop_size*np.concatenate((np.ones(1),np.zeros(NAG-1)))\
            -S_REL[i]*S_AGE*(np.dot(INFECTIOUS_CONTACT(t),np.sum(np.array(([state[(3*j+2)*NAG:(3*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(3*i+1)*NAG:(3*i+2)*NAG]\
            + WANE_UP[i-1]*state[(3*i)*NAG:(3*i+1)*NAG] + WANE_SAME[i]*state[(3*i+3)*NAG:(3*i+4)*NAG]\
            - AGING_RATE*state[(3*i+1)*NAG:(3*i+2)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(3*i+1)*NAG:(3*i+2)*NAG-1]))\
            + (all_vax(t,3*i+1,S_VAX,ACOV,T_VAX,NAG,N_S)*state).reshape((3*N_S+1,NAG)).sum(axis=0)
        delta[(3*i+2)*NAG:(3*i+3)*NAG] = S_REL[i]*S_AGE*(np.dot(INFECTIOUS_CONTACT(t),np.sum(np.array(([state[(3*j+2)*NAG:(3*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(3*i+1)*NAG:(3*i+2)*NAG]\
            - REC[i]*state[(3*i+2)*NAG:(3*i+3)*NAG]\
            - AGING_RATE*state[(3*i+2)*NAG:(3*i+3)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(3*i+2)*NAG:(3*i+3)*NAG-1]))\
            + (all_vax(t,3*i+2,S_VAX,ACOV,T_VAX,NAG,N_S)*state).reshape((3*N_S+1,NAG)).sum(axis=0)
        delta[(3*i+3)*NAG:(3*i+4)*NAG] = REC[i]*state[(3*i+2)*NAG:(3*i+3)*NAG]\
            - (WANE_UP[i]+WANE_SAME[i])*state[(3*i+3)*NAG:(3*i+4)*NAG]\
            - AGING_RATE*state[(3*i+3)*NAG:(3*i+4)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(3*i+3)*NAG:(3*i+4)*NAG-1]))\
            + (all_vax(t,3*i+3,S_VAX,ACOV,T_VAX,NAG,N_S)*state).reshape((3*N_S+1,NAG)).sum(axis=0)
    # # Enforce stable population size
    # if np.sum(delta) > 0:
    #     delta -= np.sum(delta)*state/np.sum(state)
    return delta

## Integrate the system
POINTS = 1000
T_VAX = 500*T_FACTOR
result = sp.integrate.solve_ivp(deltas, [0,PERIOD], STATE0, method='RK45', t_eval=np.linspace(0,PERIOD,POINTS),
 args=((NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, INFECTIOUS_CONTACT),))

## Calculate observations
obs = np.zeros((len(result.t),NAG))
pop_size = np.sum(result.y,axis=0)
for i_t,t in enumerate(result.t):
    foi = np.dot(INFECTIOUS_CONTACT(t),np.sum((np.array([result.y[(3*j+2)*NAG:(3*j+3)*NAG,i_t] for j in range(N_S)])*I_REL),axis=0))/pop_size[i_t]
    for i in range(N_S):
        obs[i_t,:] += OBS_AGE*P_OBS[i]*S_REL[i]*foi*result.y[(3*i+1)*NAG:(3*i+2)*NAG,i_t]

## Plot the results
import matplotlib.pyplot as plt

viridis = plt.cm.get_cmap('viridis', NAG)

fig, axes = plt.subplots(2,2,figsize=(6.5,6.5))
axes[0,0].plot(result.t,np.sum(obs,axis=1)/pop_size, label='Observed cases')
axes[0,0].set_title('Simulation (including burn-in)')
axes[0,0].set_ylabel('Observed incidence')
axes[0,1].plot(result.t,np.sum(obs,axis=1)/pop_size, label='Observed cases')
axes[0,1].set_xlim(2*PERIOD/3,PERIOD)
axes[0,1].set_ylim(0,4e-4)
# axes[0,1].set_ylim(0,2e-5)
axes[0,1].set_title('Detail after burn-in')
axes[0,1].set_ylabel('Observed incidence')
# axes[1,0].plot(result.t,pop_size, label='Population size')
# axes[1,0].set_title('Total population')
# axes[1,0].set_ylabel('Persons')

# Infected persons in each age group
for i in range(NAG):
    axes[1,0].plot(result.t,obs[:,i]/np.sum(result.y[range(i,10*NAG,NAG),:],axis=0), label=AGE_GROUP_NAMES[i], color=viridis(i), alpha=0.5)
axes[1,0].set_xlim(2*PERIOD/3,PERIOD)
axes[1,0].set_ylim(0,1e-3)
axes[1,0].set_title('Incidence by age group')
axes[1,0].set_ylabel('Observed incidence')

# Population in each age group
for i in range(NAG):
    axes[1,1].plot(result.t,np.sum(result.y[range(i,10*NAG,NAG),:],axis=0), label=AGE_GROUP_NAMES[i], color=viridis(i))
axes[1,1].set_title('Age groups')
axes[1,1].set_ylabel('Age group population')


plt.tight_layout()
# plt.savefig('Figures/SIR3_rota_demo_pc.png')
plt.show()