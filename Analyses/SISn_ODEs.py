## Ordidxnary differential equations defining SIR models with n susceptibility classes
## BJS September 2024

from numba import jit
import numpy as np

## Differential equations
def single_pathogen_deltas(t,state,NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, arrivals, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact):
    delta = np.zeros(state.shape)
    pop_size = np.sum(state,dtype=np.float64)
    # Susceptible, infected - waning, aging, infection, recovery for all susceptibility classes
    for i in range(N_S):
        # Susceptibile class i = birth - infection + recovery - waning out + waning in + aging in - aging out +/- vaccination
        delta[(2*i+1)*NAG:(2*i+2)*NAG] = birth_vax(t,i,S_VAX,BCOV,T_VAX)*birth_rate(t)*pop_size*np.concatenate((np.ones(1),np.zeros(NAG-1)))\
            - S_REL[i]*S_AGE*BETA*(np.dot(contact(t,SEASONALITY,OFFSET),np.sum(np.array(([state[(2*j+2)*NAG:(2*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(2*i+1)*NAG:(2*i+2)*NAG]\
            + REC_UP[i-1]*state[(2*i)*NAG:(2*i+1)*NAG] + REC_SAME[i]*state[(2*i+2)*NAG:(2*i+3)*NAG]\
            - WANE[i-1]*state[(2*i+1)*NAG:(2*i+2)*NAG] + WANE[i]*state[(2*i+3)*NAG:(2*i+4)*NAG]\
            - AGING_RATE*state[(2*i+1)*NAG:(2*i+2)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(2*i+1)*NAG:(2*i+2)*NAG-1]))\
            + (all_vax(t,i,ACOV,S_VAX,T_VAX,NAG,N_S,2)*state).reshape((2*N_S+2,NAG)).sum(axis=0)
        # Infectious class i = infection - recovery + aging in - aging out - vaccination + importations
        delta[(2*i+2)*NAG:(2*i+3)*NAG] = S_REL[i]*S_AGE*BETA*(np.dot(contact(t,SEASONALITY,OFFSET),np.sum(np.array(([state[(2*j+2)*NAG:(2*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(2*i+1)*NAG:(2*i+2)*NAG]\
            - (REC_UP[i]+REC_SAME[i])*state[(2*i+2)*NAG:(2*i+3)*NAG]\
            - AGING_RATE*state[(2*i+2)*NAG:(2*i+3)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(2*i+2)*NAG:(2*i+3)*NAG-1]))\
            + IMPORT_RATE*arrivals(t)
    return delta

# ## Numba compliant version
# @jit
# def single_pathogen_deltas(t, state, NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, arrivals, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact):
#     delta = np.zeros(state.shape,dtype=np.float64)
#     pop_size = np.sum(state,dtype=np.float64)
#     # Susceptible, infected - waning, aging, infection, recovery for all susceptibility classes
#     for i in range(N_S):
#         for age in range(NAG):
#             # Susceptibile class i = birth - infection + recovery - waning out + waning in + aging in - aging out +/- vaccination
#             delta[(2*i+1)*NAG+age] = (age==0)*birth_vax(t,i,S_VAX,BCOV,T_VAX)*birth_rate(t)*pop_size\
#                 - S_REL[i]*S_AGE[age]*BETA*(np.dot(contact(t,SEASONALITY,OFFSET),np.sum(np.array([state[j] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=np.float64).reshape((N_S,NAG))*I_REL,axis=0))/pop_size)[age]*state[(2*i+1)*NAG+age]\
#                 + REC_UP[i-1]*state[(2*i)*NAG+age] + REC_SAME[i]*state[(2*i+2)*NAG+age]\
#                 - WANE[i-1]*state[(2*i+1)*NAG+age] + WANE[i]*state[(2*i+3)*NAG+age]\
#                 - AGING_RATE[age]*state[(2*i+1)*NAG+age] + (age>0)*AGING_RATE[age-1]*state[(2*i+1)*NAG+age-1]\
#                 + (all_vax(t,i,ACOV,S_VAX,T_VAX,NAG,N_S,2)*state).reshape((2*N_S+2,NAG)).sum(axis=0)[age]
#             # Infectious class i = infection - recovery + aging in - aging out - vaccination + importations
#             delta[(2*i+2)*NAG+age] = S_REL[i]*S_AGE[age]*BETA*(np.dot(contact(t,SEASONALITY,OFFSET),np.sum(np.array([state[j] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=np.float64).reshape((N_S,NAG))*I_REL,axis=0))/pop_size)[age]*state[(2*i+1)*NAG+age]\
#                 - (REC_UP[i]+REC_SAME[i])*state[(2*i+2)*NAG+age]\
#                 - AGING_RATE[age]*state[(2*i+2)*NAG+age] + (age>0)*AGING_RATE[age-1]*state[(2*i+2)*NAG+age-1]\
#                 + IMPORT_RATE*arrivals(t)
#                                 # + IMPORT_RATE*S_REL[i]*S_AGE[age]*BETA*np.sum(contact(t,SEASONALITY,OFFSET)[age])*arrivals(t)*state[(2*i+1)*NAG+age]\
#     return delta