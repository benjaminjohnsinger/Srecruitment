## Ordinary differential equations defining SIR models with n susceptibility classes
## BJS September 2024

import numpy as np

## Differential equations
def single_pathogen_deltas(t,state,params):
    delta = np.zeros(state.shape)
    pop_size = np.sum(state)
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params
    # Susceptible, infected, recovered - waning, aging, infection, recovery for all susceptibility classes
    for i in range(N_S):
        # Susceptibile class i = birth - infection + waning + aging in - aging out +/- vaccination
        delta[(3*i+1)*NAG:(3*i+2)*NAG] = birth_vax(t,i,S_VAX,BCOV,T_VAX)*BIRTH_RATE*pop_size*np.concatenate((np.ones(1),np.zeros(NAG-1)))\
            -S_REL[i]*S_AGE*BETA*(np.dot(contact(t),np.sum(np.array(([state[(3*j+2)*NAG:(3*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(3*i+1)*NAG:(3*i+2)*NAG]\
            + WANE_UP[i-1]*state[(3*i)*NAG:(3*i+1)*NAG] + WANE_SAME[i]*state[(3*i+3)*NAG:(3*i+4)*NAG]\
            - AGING_RATE*state[(3*i+1)*NAG:(3*i+2)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(3*i+1)*NAG:(3*i+2)*NAG-1]))\
            + (all_vax(t,i,ACOV,S_VAX,T_VAX,NAG,N_S)*state).reshape((3*N_S+1,NAG)).sum(axis=0)
        # Infectious class i = infection - recovery + aging in - aging out - vaccination + importations
        delta[(3*i+2)*NAG:(3*i+3)*NAG] = S_REL[i]*S_AGE*BETA*(np.dot(contact(t),np.sum(np.array(([state[(3*j+2)*NAG:(3*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(3*i+1)*NAG:(3*i+2)*NAG]\
            - REC[i]*state[(3*i+2)*NAG:(3*i+3)*NAG]\
            - AGING_RATE*state[(3*i+2)*NAG:(3*i+3)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(3*i+2)*NAG:(3*i+3)*NAG-1]))\
            + IMPORT
        # Recovered class i = recovery - waning + aging in - aging out - vaccination
        delta[(3*i+3)*NAG:(3*i+4)*NAG] = REC[i]*state[(3*i+2)*NAG:(3*i+3)*NAG]\
            - (WANE_UP[i]+WANE_SAME[i])*state[(3*i+3)*NAG:(3*i+4)*NAG]\
            - AGING_RATE*state[(3*i+3)*NAG:(3*i+4)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(3*i+3)*NAG:(3*i+4)*NAG-1]))
    return delta