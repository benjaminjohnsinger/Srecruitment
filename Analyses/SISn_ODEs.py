## Ordidxnary differential equations defining SIR models with n susceptibility classes
## BJS September 2024

import numpy as np

## Differential equations
def single_pathogen_deltas(t,state,params):
    delta = np.zeros(state.shape)
    pop_size = np.sum(state)
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params.values()
    # Susceptible, infected - waning, aging, infection, recovery for all susceptibility classes
    # First susceptible class = birth - infection - waning out + waning in + aging in - aging out +/- vaccination
    delta[0:NAG] = birth_vax(t,0,S_VAX,BCOV,T_VAX)*BIRTH_RATE*pop_size*np.concatenate((np.ones(1),np.zeros(NAG-1)))\
        - S_REL[0]*S_AGE*BETA*(np.dot(contact(t),np.sum(np.array(([state[(2*j+1)*NAG:(2*j+2)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[0:NAG]\
        - WANE[0]*state[0:NAG] + WANE[1]*state[2*NAG:3*NAG]\
        - AGING_RATE*state[0:NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[0:NAG-1]))\
        + (all_vax(t,0,ACOV,S_VAX,T_VAX,NAG,N_S)*state).reshape((2*N_S,NAG)).sum(axis=0)
    # First infectious classs = infection - recovery + aging in - aging out - vaccination + importations
    delta[NAG:2*NAG] = S_REL[0]*S_AGE*BETA*(np.dot(contact(t),np.sum(np.array(([state[(2*j+1)*NAG:(2*j+2)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[0:NAG]\
        - REC[0]*state[NAG:2*NAG]\
        - AGING_RATE*state[NAG:2*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[NAG:2*NAG-1]))\
        + IMPORT[0]
    for i in range(1,N_S-1):
        # Susceptibile class i = birth - infection + recovery - waning out + waning in + aging in - aging out +/- vaccination
        delta[(2*i)*NAG:(2*i+1)*NAG] = birth_vax(t,i,S_VAX,BCOV,T_VAX)*BIRTH_RATE*pop_size*np.concatenate((np.ones(1),np.zeros(NAG-1)))\
            - S_REL[i]*S_AGE*BETA*(np.dot(contact(t),np.sum(np.array(([state[(2*j+1)*NAG:(2*j+2)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(2*i)*NAG:(2*i+1)*NAG]\
            - WANE[i]*state[(2*i)*NAG:(2*i+1)*NAG] + WANE[i+1]*state[(2*i+2)*NAG:(2*i+3)*NAG]\
            + REC[i-1]*state[(2*(i-1)+1)*NAG:(2*(i-1)+2)*NAG]\
            - AGING_RATE*state[(2*i)*NAG:(2*i+1)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(2*i)*NAG:(2*i+1)*NAG-1]))\
            + (all_vax(t,i,ACOV,S_VAX,T_VAX,NAG,N_S)*state).reshape((2*N_S,NAG)).sum(axis=0)
        # Infectious class i = infection - recovery + aging in - aging out - vaccination + importations
        delta[(2*i+1)*NAG:(2*i+2)*NAG] = S_REL[i]*S_AGE*BETA*(np.dot(contact(t),np.sum(np.array(([state[(2*j+1)*NAG:(2*j+2)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(2*i)*NAG:(2*i+1)*NAG]\
            - REC[i]*state[(2*i+1)*NAG:(2*i+2)*NAG]\
            - AGING_RATE*state[(2*i+1)*NAG:(2*i+2)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(2*i+1)*NAG:(2*i+2)*NAG-1]))\
            + IMPORT[i]
    idx = N_S-1
    # Last susceptibile class = birth - infection - waning out + aging in - aging out +/- vaccination
    delta[(2*idx)*NAG:(2*idx+1)*NAG] = birth_vax(t,idx,S_VAX,BCOV,T_VAX)*BIRTH_RATE*pop_size*np.concatenate((np.ones(1),np.zeros(NAG-1)))\
        - S_REL[idx]*S_AGE*BETA*(np.dot(contact(t),np.sum(np.array(([state[(2*j+1)*NAG:(2*j+2)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(2*idx)*NAG:(2*idx+1)*NAG]\
        - WANE[idx]*state[(2*idx)*NAG:(2*idx+1)*NAG]\
        + REC[idx-1]*state[(2*(idx-1)+1)*NAG:(2*(idx-1)+2)*NAG] + REC[idx]*state[(2*idx+1)*NAG:(2*idx+2)*NAG]\
        - AGING_RATE*state[(2*idx)*NAG:(2*idx+1)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(2*idx)*NAG:(2*idx+1)*NAG-1]))\
        + (all_vax(t,idx,ACOV,S_VAX,T_VAX,NAG,N_S)*state).reshape((2*N_S,NAG)).sum(axis=0)
    # Last infectious class = infection - recovery + aging in - aging out - vaccination + importations
    delta[(2*idx+1)*NAG:(2*idx+2)*NAG] = S_REL[idx]*S_AGE*BETA*(np.dot(contact(t),np.sum(np.array(([state[(2*j+1)*NAG:(2*j+2)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(2*idx)*NAG:(2*idx+1)*NAG]\
        - REC[idx]*state[(2*idx+1)*NAG:(2*idx+2)*NAG]\
        - AGING_RATE*state[(2*idx+1)*NAG:(2*idx+2)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(2*idx+1)*NAG:(2*idx+2)*NAG-1]))\
        + IMPORT[idx]
    return delta