## Ordidxnary differential equations defining SIR models with n susceptibility classes
## BJS September 2024

from numba import jit
import numpy as np
N_C = 2

# ## Differential equations
# def single_pathogen_deltas(t,state,NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, arrivals, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact):
#     delta = np.zeros(state.shape)
#     pop_size = np.sum(state,dtype=np.float64)
#     age_pops = np.array([np.sum(state[range(i_age,(2*N_S+1)*NAG,NAG)],axis=0) for i_age in range(NAG)])
#     # Susceptible, infected - waning, aging, infection, recovery for all susceptibility classes
#     for i in range(N_S):
#         # Susceptibile class i = birth - infection + recovery - waning out + waning in + aging in - aging out +/- vaccination
#         delta[(2*i+1)*NAG:(2*i+2)*NAG] = birth_vax(t,i,S_VAX,BCOV,T_VAX)*birth_rate(t)*pop_size*np.concatenate((np.ones(1),np.zeros(NAG-1)))\
#             - S_REL[i]*S_AGE*BETA*(np.dot(contact(t,SEASONALITY,OFFSET),np.sum(np.array(([state[(2*j+2)*NAG:(2*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(2*i+1)*NAG:(2*i+2)*NAG]\
#             + REC_UP[i-1]*state[(2*i)*NAG:(2*i+1)*NAG] + REC_SAME[i]*state[(2*i+2)*NAG:(2*i+3)*NAG]\
#             - WANE[i-1]*state[(2*i+1)*NAG:(2*i+2)*NAG] + WANE[i]*state[(2*i+3)*NAG:(2*i+4)*NAG]\
#             - AGING_RATE*state[(2*i+1)*NAG:(2*i+2)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(2*i+1)*NAG:(2*i+2)*NAG-1]))\
#             + (all_vax(t,i,ACOV,S_VAX,NAG,N_S,2,[S_REL*P_OBS,age_pops,AGING_RATE])*state).reshape((2*N_S+2,NAG)).sum(axis=0)\
#             - IMPORT_RATE*regional_positivity(t)*S_REL[i]*S_AGE*BETA*np.sum(contact(t,SEASONALITY,OFFSET),axis=0)*arrivals(t)*state[(2*i+1)*NAG:(2*i+2)*NAG]
#         # Infectious class i = infection - recovery + aging in - aging out - vaccination + importations
#         delta[(2*i+2)*NAG:(2*i+3)*NAG] = S_REL[i]*S_AGE*BETA*(np.dot(contact(t,SEASONALITY,OFFSET),np.sum(np.array(([state[(2*j+2)*NAG:(2*j+3)*NAG] for j in range(N_S)]))*I_REL,axis=0))/pop_size)*state[(2*i+1)*NAG:(2*i+2)*NAG]\
#             - (REC_UP[i]+REC_SAME[i])*state[(2*i+2)*NAG:(2*i+3)*NAG]\
#             - AGING_RATE*state[(2*i+2)*NAG:(2*i+3)*NAG] + np.concatenate((np.zeros(1), AGING_RATE[:-1]*state[(2*i+2)*NAG:(2*i+3)*NAG-1]))\
#             + IMPORT_RATE*regional_positivity(t)*S_REL[i]*S_AGE*BETA*np.sum(contact(t,SEASONALITY,OFFSET),axis=0)*arrivals(t)*state[(2*i+1)*NAG:(2*i+2)*NAG]
#     return delta

@jit
def delta_helper(state,NAG,N_S,REC_UP,REC_SAME,WANE):
    delta = np.zeros(state.shape)
    for i in range(N_S):
        delta[(2*i+1)*NAG:(2*i+2)*NAG] = REC_UP[i-1]*state[(2*i)*NAG:(2*i+1)*NAG] + REC_SAME[i]*state[(2*i+2)*NAG:(2*i+3)*NAG]\
            - WANE[i-1]*state[(2*i+1)*NAG:(2*i+2)*NAG] + WANE[i]*state[(2*i+3)*NAG:(2*i+4)*NAG]
        delta[(2*i+2)*NAG:(2*i+3)*NAG] = -(REC_UP[i]+REC_SAME[i])*state[(2*i+2)*NAG:(2*i+3)*NAG]
    return delta

def single_pathogen_deltas(t,state,NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, arrivals, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact):
    delta = np.zeros(state.shape)
    pop_size = np.sum(state,dtype=np.float64)
    age_pops = np.array([np.sum(state[range(i_age,(2*N_S+1)*NAG,NAG)],axis=0) for i_age in range(NAG)])
    # births either straight into youngest age group in first susceptibility class or split between first and S_VAX
    delta += birth_vax(t,S_VAX,BCOV,T_VAX,NAG,N_S,N_C)*birth_rate(t)*pop_size
    # infections are negative for susceptibles and positive for infected
    contact_t = contact(t,SEASONALITY,OFFSET)
    infectious_contact = np.dot(contact_t,np.sum(np.array([state[j] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=np.float64).reshape((N_S,NAG))*I_REL,axis=0))/pop_size
    import_contact = IMPORT_RATE*regional_positivity(t)*arrivals(t)*np.sum(contact_t,axis=0)
    delta[NAG:-NAG] += np.repeat(S_REL,NAG*N_C)*np.tile(S_AGE,N_S*N_C)*BETA*np.tile(infectious_contact+import_contact,N_S*N_C)*np.repeat(np.tile(np.array([-1,1]+[0]*(N_C-2)),N_S),NAG)*np.array([np.tile(state[(2*i+1)*NAG:(2*i+2)*NAG],N_C) for i in range(N_S)]).flatten()
    # # recovery is positive for susceptibles and negative for infected 
    # delta[NAG:-NAG] += np.repeat(REC_SAME,NAG*N_C)*np.repeat(np.tile(np.array([1]+[0]*(N_C-2)+[-1]),N_S),NAG)*np.array([np.tile(state[(2*i+2)*NAG:(2*i+3)*NAG],N_C) for i in range(N_S)]).flatten()
    # delta[N_C*NAG:-2*NAG] += np.repeat(REC_UP[:-1],NAG*N_C)*np.repeat(np.tile(np.array([-1,1]+[0]*(N_C-2)),N_S-1),NAG)*np.array([np.tile(state[(2*i+2)*NAG:(2*i+3)*NAG],N_C) for i in range(N_S-1)]).flatten()
    # # waning immunity - this is unfortuantely less efficient than the for loop
    # delta[(N_C+1)*NAG:-NAG] -= np.repeat(WANE[:-1],NAG*N_C)*np.repeat(np.tile(np.array([1]+[0]*(N_C-1)),N_S-1),NAG)*np.array([np.tile(state[(2*i+1)*NAG:(2*i+2)*NAG],N_C) for i in range(1,N_S)]).flatten()
    # delta[NAG:-(N_C+1)*NAG] += np.repeat(WANE[:-1],NAG*N_C)*np.repeat(np.tile(np.array([1]+[0]*(N_C-1)),N_S-1),NAG)*np.array([np.tile(state[(2*i+1)*NAG:(2*i+2)*NAG],N_C) for i in range(1,N_S)]).flatten()
    delta += delta_helper(state,NAG,N_S,REC_UP,REC_SAME,WANE)
    # aging
    delta[NAG:-NAG] -= (AGING_RATE*state[NAG:-NAG].reshape((N_C*N_S,NAG))).flatten()
    temp_age = AGING_RATE.copy()
    temp_age[-1] = 0
    delta[(NAG+1):-(NAG-1)] += (temp_age*state[NAG:-NAG].reshape((N_C*N_S,NAG))).flatten()
    # vaccination
    delta[NAG:-NAG] -= (ACOV(t,S_REL*P_OBS,age_pops,AGING_RATE)*np.array([(i%N_C==0 and i//N_C!=S_VAX)*state[(i+1)*NAG:(i+2)*NAG] for i in range(N_S*N_C)])).flatten()
    delta[(S_VAX*N_C+1)*NAG:(S_VAX*N_C+2)*NAG] += ACOV(t,S_REL*P_OBS,age_pops,AGING_RATE)*np.sum(np.array([(i%N_C==0 and i//N_C!=S_VAX)*state[(i+1)*NAG:(i+2)*NAG] for i in range(N_S*N_C)],dtype=np.float64).reshape((N_S*N_C,NAG)),axis=0)
    return delta


# ## Numba compliant version - this actually runs slower, numpy vectorization is more powerful than numba it seems
# @jit
# def single_pathogen_deltas(t, state, NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, arrivals, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact):
#     delta = np.zeros(state.shape,dtype=np.float64)
#     pop_size = np.sum(state,dtype=np.float64)
#     age_pops = np.zeros(NAG)
#     for age in range(NAG):
#         for i in range(2*N_S+1):
#             age_pops[age] += state[i*NAG+age]
#     # Susceptible, infected - waning, aging, infection, recovery for all susceptibility classes
#     for i in range(N_S):
#         for age in range(NAG):
#             # Susceptibile class i = birth - infection + recovery - waning out + waning in + aging in - aging out +/- vaccination
#             delta[(2*i+1)*NAG+age] = (age==0)*birth_vax(t,i,S_VAX,BCOV,T_VAX)*birth_rate(t)*pop_size\
#                 - S_REL[i]*S_AGE[age]*BETA*(np.dot(contact(t,SEASONALITY,OFFSET),np.sum(np.array([state[j] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=np.float64).reshape((N_S,NAG))*I_REL,axis=0))/pop_size)[age]*state[(2*i+1)*NAG+age]\
#                 + REC_UP[i-1]*state[(2*i)*NAG+age] + REC_SAME[i]*state[(2*i+2)*NAG+age]\
#                 - WANE[i-1]*state[(2*i+1)*NAG+age] + WANE[i]*state[(2*i+3)*NAG+age]\
#                 - AGING_RATE[age]*state[(2*i+1)*NAG+age] + (age>0)*AGING_RATE[age-1]*state[(2*i+1)*NAG+age-1]\
#                 + (all_vax(t,i,ACOV,S_VAX,NAG,N_S,2,[S_REL*P_OBS,age_pops,AGING_RATE],age)*state).reshape((2*N_S+2,NAG)).sum(axis=0)[age]\
#                 - IMPORT_RATE*regional_positivity(t)*S_REL[i]*S_AGE[age]*BETA*np.sum(contact(t,SEASONALITY,OFFSET),axis=0)[age]*arrivals(t)*state[(2*i+1)*NAG+age]
#             # Infectious class i = infection - recovery + aging in - aging out - vaccination + importations
#             delta[(2*i+2)*NAG+age] = S_REL[i]*S_AGE[age]*BETA*(np.dot(contact(t,SEASONALITY,OFFSET),np.sum(np.array([state[j] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=np.float64).reshape((N_S,NAG))*I_REL,axis=0))/pop_size)[age]*state[(2*i+1)*NAG+age]\
#                 - (REC_UP[i]+REC_SAME[i])*state[(2*i+2)*NAG+age]\
#                 - AGING_RATE[age]*state[(2*i+2)*NAG+age] + (age>0)*AGING_RATE[age-1]*state[(2*i+2)*NAG+age-1]\
#                 + IMPORT_RATE*regional_positivity(t)*S_REL[i]*S_AGE[age]*BETA*np.sum(contact(t,SEASONALITY,OFFSET),axis=0)[age]*arrivals(t)*state[(2*i+1)*NAG+age]
#                 # + IMPORT_RATE*arrivals(t)
#                                 # + IMPORT_RATE*S_REL[i]*S_AGE[age]*BETA*np.sum(contact(t,SEASONALITY,OFFSET)[age])*arrivals(t)*state[(2*i+1)*NAG+age]\
#     return delta