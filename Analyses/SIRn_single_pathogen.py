## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import itertools as it

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
AGE_PROPORTION = np.array([np.sum(AGE_POP[group]) for group in KP_AGE_GROUPS])/np.sum(AGE_POP)
KP_AGE_POP = AGE_PROPORTION*POP_SIZE
NAG = len(KP_AGE_GROUPS)

# Rate of aging out of each age group. Last rate informed by US life expectancy at age 65.
AGING_RATE = 1/(T_FACTOR*np.array([3,9,4*12,13*12,22*12,25*12,18.35*12]))

# Birth rate (for California in 2022)
BIRTH_RATE = 3.99e5/(POP_SIZE*12*T_FACTOR)
# AGING_RATE = BIRTH_RATE/AGE_PROPORTION # Stable population distribution

## Parameters that vary by susceptibility class (for rotavirus in this example)
# Number of susceptibility classes
N_S = 3
# Immunity waning rates for each susceptibility class, with prefixed zero
WANE_UP = np.array([1/9,1/9,0])/T_FACTOR # Waning to higher susceptibility class (last value always 0)
WANE_SAME = np.array([0,0,1/12])/T_FACTOR # Waning to same susceptibility class
# Recovery rates for each susceptibility class
REC = np.array([4.3,8.6,8.6])/T_FACTOR
# Relative susceptability and immunity, for each susceptibility class
S_REL = np.array([1,0.62,0.35])
I_REL = np.array([[1],[0.5],[0.1]])
# Probability of detection of cases for each susceptibility class
P_OBS = 0.041*np.array([0.11,0.029,0])

## Parameters that vary by age group
# Age-specific susceptibility
# S_AGE = np.array([1.555,1.555,1.5125,1,1,1,1])
S_AGE = np.array([1,1,1,1,1,1,1])

## Time-varying parameters
# Vaccination of infants
# S_CLASS is the susceptibility class of the individual
# S_VAX is the susceptibility class of vaccinated individuals
# COVERAGE is the proportion of infants successfully protected
# T_VAX is the time at which vaccination starts
def birth_vax(t,S_CLASS,S_VAX=2,COVERAGE=0.96*0.8,T_VAX=500*T_FACTOR):
    if t < T_VAX:
        return 1 if (S_CLASS == 0) else 0
    else:
        if S_CLASS == S_VAX:
            return COVERAGE
        elif S_CLASS == 0:
            return 1-COVERAGE
        else:
            return 0

# Force of infection per contact
SEASONALITY = 0.05
OFFSET = 0.636*T_FACTOR
BETA_FUDGE_FACTOR = 1.3 # Factor to adjust Pitzer parameters to work with KP contact matrices
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
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, I_REL, P_OBS, birth_vax, INFECTIOUS_CONTACT = params
    # Susceptible, infected, recovered - waning, aging, infection, recovery for all susceptibility classes
    for i in range(N_S):
        delta[(3*i+1)*NAG:(3*i+2)*NAG] = birth_vax(t,i)*BIRTH_RATE*pop_size*np.concatenate((np.ones(1),np.zeros(NAG-1)))\
            -S_REL[i]*S_AGE*(np.dot(INFECTIOUS_CONTACT(t),np.sum(np.array(([state[(3*j+2)*NAG:(3*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(3*i+1)*NAG:(3*i+2)*NAG]\
            + WANE_UP[i-1]*state[(3*i)*NAG:(3*i+1)*NAG] + WANE_SAME[i]*state[(3*i+3)*NAG:(3*i+4)*NAG]\
            - AGING_RATE*state[(3*i+1)*NAG:(3*i+2)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(3*i+1)*NAG:(3*i+2)*NAG-1]))
        delta[(3*i+2)*NAG:(3*i+3)*NAG] = S_REL[i]*S_AGE*(np.dot(INFECTIOUS_CONTACT(t),np.sum(np.array(([state[(3*j+2)*NAG:(3*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(3*i+1)*NAG:(3*i+2)*NAG]\
            - REC[i]*state[(3*i+2)*NAG:(3*i+3)*NAG]\
            - AGING_RATE*state[(3*i+2)*NAG:(3*i+3)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(3*i+2)*NAG:(3*i+3)*NAG-1]))
        delta[(3*i+3)*NAG:(3*i+4)*NAG] = REC[i]*state[(3*i+2)*NAG:(3*i+3)*NAG]\
            - (WANE_UP[i]+WANE_SAME[i])*state[(3*i+3)*NAG:(3*i+4)*NAG]\
            - AGING_RATE*state[(3*i+3)*NAG:(3*i+4)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(3*i+3)*NAG:(3*i+4)*NAG-1]))
    # # Enforce stable population size
    # if np.sum(delta) > 0:
    #     delta -= np.sum(delta)*state/np.sum(state)
    return delta

## Integrate the system
POINTS = 1000
result = sp.integrate.solve_ivp(deltas, [0,PERIOD], STATE0, method='RK45', t_eval=np.linspace(0,PERIOD,POINTS),
 args=((NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, I_REL, P_OBS, birth_vax, INFECTIOUS_CONTACT),))

## Calculate observations
obs = np.zeros(len(result.t))
pop_size = np.sum(result.y,axis=0)
for i_t,t in enumerate(result.t):
    foi = np.dot(INFECTIOUS_CONTACT(t),np.sum((np.array([result.y[(3*j+2)*NAG:(3*j+3)*NAG,i_t] for j in range(N_S)])*I_REL),axis=0))/pop_size[i_t]
    for i in range(N_S):
        obs[i_t] += P_OBS[i]*S_REL[i]*np.sum(foi*result.y[(3*i+1)*NAG:(3*i+2)*NAG,i_t])

## Plot the results
import matplotlib.pyplot as plt

viridis = plt.cm.get_cmap('viridis', 7)

# plt.plot(result.t,obs, label='Observed cases per capita')
# # plt.xlim(300,400)
# # plt.ylim(0,2e-5)
# plt.show()

fig, axes = plt.subplots(2,2,figsize=(6.5,6.5))
axes[0,0].plot(result.t,obs/pop_size, label='Observed cases')
axes[0,0].set_title('Simulation (including burn-in)')
axes[0,0].set_ylabel('Observed incidence')
axes[0,1].plot(result.t,obs/pop_size, label='Observed cases')
axes[0,1].set_xlim(2*PERIOD/3,PERIOD)
# axes[0,1].set_ylim(0,2e-4)
axes[0,1].set_ylim(0,2e-5)
axes[0,1].set_title('Detail after burn-in')
axes[0,1].set_ylabel('Observed incidence')
axes[1,0].plot(result.t,pop_size, label='Population size')
axes[1,0].set_title('Total population')
axes[1,0].set_ylabel('Persons')
axes[1,1].plot(result.t,np.sum(result.y[range(0,10*NAG,NAG),:],axis=0)/pop_size, label='Newborns', color=viridis(0))
axes[1,1].plot(result.t,np.sum(result.y[range(1,10*NAG,NAG),:],axis=0)/pop_size, label='Infants', color=viridis(1))
axes[1,1].plot(result.t,np.sum(result.y[range(2,10*NAG,NAG),:],axis=0)/pop_size, label='Young children', color=viridis(2))
axes[1,1].plot(result.t,np.sum(result.y[range(3,10*NAG,NAG),:],axis=0)/pop_size, label='Older children', color=viridis(3))
axes[1,1].plot(result.t,np.sum(result.y[range(4,10*NAG,NAG),:],axis=0)/pop_size, label='Young adults', color=viridis(4))
axes[1,1].plot(result.t,np.sum(result.y[range(5,10*NAG,NAG),:],axis=0)/pop_size, label='Middle-aged adults', color=viridis(5))
axes[1,1].plot(result.t,np.sum(result.y[range(6,10*NAG,NAG),:],axis=0)/pop_size, label='Older adults', color=viridis(6))
axes[1,1].set_title('Age groups')
axes[1,1].set_ylabel('Proportion of total population')
plt.tight_layout()
# plt.savefig('Figures/SIR3_rota_demo_pc.png')
plt.show()