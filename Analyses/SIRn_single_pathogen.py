## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import itertools as it
## Period of simulation in months
PERIOD = 12*35

## Population size and age distribution
POP_SIZE = 3.9e7 # California population in 2022
AGE_POP = np.genfromtxt('Data/Processed/US_Census_population_by_age.csv', delimiter=',') # US in 2022
KP_AGE_GROUPS = [range(3),range(3,12),range(12,5*12),range(5*12,18*12),range(18*12,40*12),range(40*12,65*12),range(65*12,100*12)]
AGE_PROPORTION = np.array([np.sum(AGE_POP[group]) for group in KP_AGE_GROUPS])/np.sum(AGE_POP)
KP_AGE_POP = AGE_PROPORTION*POP_SIZE
NAG = len(KP_AGE_GROUPS)

# Rate of aging out of each age group. Last rate informed by US life expectancy at age 65.
AGING_RATE = 1/np.array([3,9,4*12,13*12,22*12,25*12,18.35*12])

# # Birth rate (for California in 2022)
BIRTH_RATE = 3.99e5/(POP_SIZE*12)

# Contact matrix (for US, roughly 2022)
CONTACT = np.genfromtxt('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',')

## Parameters - rotavirus in this example
# Number of susceptibility classes
N_S = 3
# Immunity waning rates, including maternal immunity
WANE = np.array([1/3,1/9,1/9,1/12])
# Recovery rates
REC = np.array([4.3,8.6,8.6])
# Relative susceptability and immunity, for each infection compartment
S_REL = np.array([1,0.62,0.35])
I_REL = np.array([[1],[0.5],[0.1]])
I_REL_3D = np.array([[[1]],[[0.5]],[[0.1]]])
# Probability of detection of cases
P_OBS = 0.041*np.array([0.11,0.029,0])

## Time-varying parameters
# Vaccination of infants
S_VAX = 2 # Which susceptibility class is vaccination equivalent to?
COVERAGE = 0.96*0.8
T_VAX = 300
def VAX(t):
    return COVERAGE if t >= T_VAX else 0
# Force of infection per contact
SEASONALITY = 0.055
OFFSET = 0.636
def BETA_PC(t):
    return 23.25*4.3/12.99 # Pitzer beta divided by average number of contacts
def FOI_PC(t):
    return BETA_PC(t)*(1+SEASONALITY*np.cos(2*np.pi*(t/12-OFFSET)))

## Initial conditions
STATE0 = np.zeros((1+N_S*3)*(NAG))
STATE0[0] = KP_AGE_POP[0] # Infants in maternal immunity compartment
STATE0[NAG+1:2*NAG] = KP_AGE_POP[1:]-1 # Every other age group is all susceptible except
STATE0[2*NAG+1:3*NAG] = 1 # one individual in each age group that is infected.


## Differential equations
def deltas(state,t):
    delta = np.zeros(state.shape)
    pop_size = np.sum(state)
    # Maternal immunity - birth into this compartment, waning, aging
    delta[0:NAG] = (1-VAX(t))*BIRTH_RATE*pop_size - WANE[0]*state[0:NAG]\
        - AGING_RATE*state[0:NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[0:NAG-1]))
    # Susceptible, infected, recovered - waning, aging, infection, recovery for all susceptibility classes
    for i in range(N_S):
        delta[(3*i+1)*NAG:(3*i+2)*NAG] = -S_REL[i]*FOI_PC(t)*(np.dot(CONTACT,np.sum(np.array(([state[(3*j+2)*NAG:(3*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(3*i+1)*NAG:(3*i+2)*NAG] + WANE[i]*state[(3*i)*NAG:(3*i+1)*NAG]\
            - AGING_RATE*state[(3*i+1)*NAG:(3*i+2)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(3*i+1)*NAG:(3*i+2)*NAG-1]))
        if i == S_VAX:
            delta[(3*i+1)*NAG:(3*i+2)*NAG] += VAX(t)*BIRTH_RATE*pop_size
        if i == N_S-1:
            delta[(3*i+1)*NAG:(3*i+2)*NAG] += WANE[-1]*state[(3*i+3)*NAG:(3*i+4)*NAG]
        delta[(3*i+2)*NAG:(3*i+3)*NAG] = S_REL[i]*FOI_PC(t)*(np.dot(CONTACT,np.sum(np.array(([state[(3*j+2)*NAG:(3*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(3*i+1)*NAG:(3*i+2)*NAG] - REC[i]*state[(3*i+2)*NAG:(3*i+3)*NAG]\
            - AGING_RATE*state[(3*i+2)*NAG:(3*i+3)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(3*i+2)*NAG:(3*i+3)*NAG-1]))
        delta[(3*i+3)*NAG:(3*i+4)*NAG] = REC[i]*state[(3*i+2)*NAG:(3*i+3)*NAG] - WANE[i+1]*state[(3*i+3)*NAG:(3*i+4)*NAG]\
            - AGING_RATE*state[(3*i+3)*NAG:(3*i+4)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(3*i+3)*NAG:(3*i+4)*NAG-1]))
    # Enforce stable population size
    if np.sum(delta) > 0:
        delta -= np.sum(delta)*state/np.sum(state)
    return delta

## Integrate the system
result = sp.integrate.odeint(deltas, STATE0, np.arange(0,PERIOD))

## Calculate observations
foi_pc = np.array([FOI_PC(t) for t in range(PERIOD)])
pop_size = np.sum(result,axis=1)
foi = foi_pc*(np.dot(CONTACT,np.sum((np.array([result[:,(3*j+2)*NAG:(3*j+3)*NAG] for j in range(N_S)])*I_REL_3D),axis=0).T)/pop_size)
obs = np.zeros(PERIOD)
for i in range(N_S):
    obs += P_OBS[i]*S_REL[i]*np.sum(foi.T*result[:,(3*i+1)*NAG:(3*i+2)*NAG],axis=1)

## Plot the results
import matplotlib.pyplot as plt
plt.plot(obs, label='Observed cases')
# plt.xlim(0,250)
plt.xlim(200,PERIOD)
plt.ylim(0,3e3)
# plt.plot(np.sum(result[:,0:NAG],axis=1), label='Maternal immunity')
# plt.plot(np.sum(result[:,1*NAG:2*NAG],axis=1), label='First susceptible class')
# plt.plot(np.sum(result[:,2*NAG:3*NAG],axis=1), label='First infections')
# plt.plot(np.sum(result[:,5*NAG:6*NAG],axis=1), label='Second infections')
# plt.plot(np.sum(result[:,8*NAG:9*NAG],axis=1), label='Asymptomatic infections')
# plt.xlim(100,PERIOD)
# plt.ylim(0,2e5)
# plt.plot(np.sum(result[:,1*NAG:2*NAG],axis=1), label='First susceptible')
# plt.plot(np.sum(result[:,4*NAG:5*NAG],axis=1), label='Second susceptible')
# plt.plot(np.sum(result[:,7*NAG:8*NAG],axis=1), label='Asymptomatic susceptible')
# plt.plot(np.sum(result[:,range(0,10*NAG,NAG)],axis=1), label='Newborns')
# plt.plot(np.sum(result[:,range(1,10*NAG,NAG)],axis=1), label='Infants')
# plt.plot(np.sum(result[:,range(2,10*NAG,NAG)],axis=1), label='Young children')
# plt.plot(np.sum(result[:,range(3,10*NAG,NAG)],axis=1), label='Older children')
# plt.plot(np.sum(result[:,range(4,10*NAG,NAG)],axis=1), label='Young adults')
# plt.plot(np.sum(result[:,range(5,10*NAG,NAG)],axis=1), label='Middle-aged adults')
# plt.plot(np.sum(result[:,range(6,10*NAG,NAG)],axis=1), label='Older adults')
plt.legend()
plt.show()