## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import itertools as it
import contact_model as cm

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
# Immunity waning rates for each susceptibility class, including maternal immunity
WANE_UP = np.array([1/3,1/9,1/9,0]) # Waning to higher susceptibility class
WANE_SAME = np.array([0,0,0,1/12]) # Waning to same susceptibility class
# Recovery rates for each susceptibility class
REC = np.array([4.3,8.6,8.6])
# Relative susceptability and immunity, for each susceptibility class
S_REL = np.array([1,0.62,0.35])
I_REL = np.array([[1],[0.5],[0.1]])
I_REL_3D = np.array([[[1]],[[0.5]],[[0.1]]])
# Probability of detection of cases for each susceptibility class
P_OBS = 0.041*np.array([0.11,0.029,0])

## Time-varying parameters
# Vaccination of infants
S_VAX = 2 # Which susceptibility class is vaccination equivalent to?
COVERAGE = 0.96*0.8
T_VAX = 300
def VAX(t,S_CLASS):
    return COVERAGE if (t >= T_VAX) & (S_CLASS == S_VAX) else 0
# Force of infection per contact
SEASONALITY = 0.055
OFFSET = 0.636
BETA_PC = cm.BETA_PC_STATIC
def INFECTIOUS_CONTACT(t):
    return BETA_PC(t)*(1+SEASONALITY*np.cos(2*np.pi*(t/12-OFFSET)))

## Initial conditions
STATE0 = np.zeros((1+N_S*3)*(NAG))
STATE0[0] = KP_AGE_POP[0] # Infants in maternal immunity compartment
STATE0[NAG+1:2*NAG] = KP_AGE_POP[1:]-1 # Every other age group is all susceptible except
STATE0[2*NAG+1:3*NAG] = 1 # one individual in each age group that is infected.

## Differential equations
def deltas(t,state,params):
    delta = np.zeros(state.shape)
    pop_size = np.sum(state)
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, I_REL, P_OBS, S_VAX, VAX, INFECTIOUS_CONTACT = params
    # Maternal immunity - birth into this compartment, waning, aging
    delta[0:NAG] = (1-VAX(t,S_VAX))*BIRTH_RATE*pop_size - WANE_UP[0]*state[0:NAG]\
        - AGING_RATE*state[0:NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[0:NAG-1]))
    # Susceptible, infected, recovered - waning, aging, infection, recovery for all susceptibility classes
    for i in range(N_S):
        delta[(3*i+1)*NAG:(3*i+2)*NAG] = VAX(t,i)*BIRTH_RATE*pop_size\
            -S_REL[i]*(np.dot(INFECTIOUS_CONTACT(t),np.sum(np.array(([state[(3*j+2)*NAG:(3*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(3*i+1)*NAG:(3*i+2)*NAG]\
            + WANE_UP[i]*state[(3*i)*NAG:(3*i+1)*NAG] + WANE_SAME[i+1]*state[(3*i+3)*NAG:(3*i+4)*NAG]\
            - AGING_RATE*state[(3*i+1)*NAG:(3*i+2)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(3*i+1)*NAG:(3*i+2)*NAG-1]))
        delta[(3*i+2)*NAG:(3*i+3)*NAG] = S_REL[i]*(np.dot(INFECTIOUS_CONTACT(t),np.sum(np.array(([state[(3*j+2)*NAG:(3*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(3*i+1)*NAG:(3*i+2)*NAG]\
            - REC[i]*state[(3*i+2)*NAG:(3*i+3)*NAG]\
            - AGING_RATE*state[(3*i+2)*NAG:(3*i+3)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(3*i+2)*NAG:(3*i+3)*NAG-1]))
        delta[(3*i+3)*NAG:(3*i+4)*NAG] = REC[i]*state[(3*i+2)*NAG:(3*i+3)*NAG]\
            - (WANE_UP[i+1]+WANE_SAME[i+1])*state[(3*i+3)*NAG:(3*i+4)*NAG]\
            - AGING_RATE*state[(3*i+3)*NAG:(3*i+4)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(3*i+3)*NAG:(3*i+4)*NAG-1]))
    # # Enforce stable population size
    # if np.sum(delta) > 0:
    #     delta -= np.sum(delta)*state/np.sum(state)
    return delta

## Integrate the system
POINTS = PERIOD
# result = sp.integrate.odeint(deltas, STATE0, np.linspace(0,PERIOD,POINTS),args=((NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, I_REL, P_OBS, S_VAX, VAX, INFECTIOUS_CONTACT),))
result = sp.integrate.solve_ivp(deltas, [0,PERIOD], STATE0, method='RK45', t_eval=np.linspace(0,PERIOD,POINTS),
 args=((NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, I_REL, P_OBS, S_VAX, VAX, INFECTIOUS_CONTACT),))


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

plt.plot(result.t,obs/pop_size, label='Observed cases per capita')
# plt.xlim(300,400)
# plt.ylim(0,2e-5)
plt.show()

# fig, axes = plt.subplots(2,2,figsize=(6.5,6.5))
# axes[0,0].plot(obs/pop_size, label='Observed cases')
# axes[0,0].set_title('Simulation (including burn-in)')
# axes[0,0].set_ylabel('Observed incidence')
# axes[0,1].plot(obs/pop_size, label='Observed cases')
# axes[0,1].set_xlim(POINTS/2,POINTS)
# axes[0,1].set_ylim(0,8e-5)
# axes[0,1].set_title('Detail after burn-in')
# axes[0,1].set_ylabel('Observed incidence')
# axes[1,0].plot(pop_size, label='Population size')
# axes[1,0].set_title('Total population')
# axes[1,0].set_ylabel('Persons')
# axes[1,1].plot(np.sum(result[:,range(0,10*NAG,NAG)],axis=1)/pop_size, label='Newborns', color=viridis(0))
# axes[1,1].plot(np.sum(result[:,range(1,10*NAG,NAG)],axis=1)/pop_size, label='Infants', color=viridis(1))
# axes[1,1].plot(np.sum(result[:,range(2,10*NAG,NAG)],axis=1)/pop_size, label='Young children', color=viridis(2))
# axes[1,1].plot(np.sum(result[:,range(3,10*NAG,NAG)],axis=1)/pop_size, label='Older children', color=viridis(3))
# axes[1,1].plot(np.sum(result[:,range(4,10*NAG,NAG)],axis=1)/pop_size, label='Young adults', color=viridis(4))
# axes[1,1].plot(np.sum(result[:,range(5,10*NAG,NAG)],axis=1)/pop_size, label='Middle-aged adults', color=viridis(5))
# axes[1,1].plot(np.sum(result[:,range(6,10*NAG,NAG)],axis=1)/pop_size, label='Older adults', color=viridis(6))
# axes[1,1].set_title('Age groups')
# axes[1,1].set_ylabel('Proportion of total population')
# plt.tight_layout()
# plt.savefig('Figures/SIR3_rota_demo_pc.png')