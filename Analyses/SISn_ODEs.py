## Ordidxnary differential equations defining SIR models with n susceptibility classes
## BJS September 2024

import numpy as np

## Differential equations
def single_pathogen_deltas(t,state,params):
    delta = np.zeros(state.shape)
    pop_size = np.sum(state,dtype=np.float64)
    NAG, N_S, AGING_RATE, births, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, SEASONALITY, OFFSET, contact = params.values()
    # Susceptible, infected - waning, aging, infection, recovery for all susceptibility classes
    for i in range(N_S):
        # Susceptibile class i = birth - infection + recovery - waning out + waning in + aging in - aging out +/- vaccination
        delta[(2*i+1)*NAG:(2*i+2)*NAG] = birth_vax(t,i,S_VAX,BCOV,T_VAX)*births(t)*np.concatenate((np.ones(1),np.zeros(NAG-1)))\
            - S_REL[i]*S_AGE*BETA*(np.dot(contact(t,SEASONALITY,OFFSET),np.sum(np.array(([state[(2*j+2)*NAG:(2*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(2*i+1)*NAG:(2*i+2)*NAG]\
            + REC_UP[i-1]*state[(2*i)*NAG:(2*i+1)*NAG] + REC_SAME[i]*state[(2*i+2)*NAG:(2*i+3)*NAG]\
            - WANE[i-1]*state[(2*i+1)*NAG:(2*i+2)*NAG] + WANE[i]*state[(2*i+3)*NAG:(2*i+4)*NAG]\
            - AGING_RATE*state[(2*i+1)*NAG:(2*i+2)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(2*i+1)*NAG:(2*i+2)*NAG-1]))\
            + (all_vax(t,i,ACOV,S_VAX,T_VAX,NAG,N_S,2)*state).reshape((2*N_S+2,NAG)).sum(axis=0)
        # Infectious class i = infection - recovery + aging in - aging out - vaccination + importations
        delta[(2*i+2)*NAG:(2*i+3)*NAG] = S_REL[i]*S_AGE*BETA*(np.dot(contact(t,SEASONALITY,OFFSET),np.sum(np.array(([state[(2*j+2)*NAG:(2*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(2*i+1)*NAG:(2*i+2)*NAG]\
            - (REC_UP[i]+REC_SAME[i])*state[(2*i+2)*NAG:(2*i+3)*NAG]\
            - AGING_RATE*state[(2*i+2)*NAG:(2*i+3)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(2*i+2)*NAG:(2*i+3)*NAG-1]))\
            + IMPORT[i]
    return delta