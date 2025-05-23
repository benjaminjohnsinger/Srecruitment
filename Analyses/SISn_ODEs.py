## Ordidxnary differential equations defining SIR models with n susceptibility classes
## BJS September 2024

from numba import jit
import numpy as np
N_C = 2 # two types of compartment

# separate function for recovery and waning immunity, since this is fasted in a numba-compiled for loop
@jit
def delta_helper(state,nag,ns,REC_UP,REC_SAME,WANE):
    delta = np.zeros(state.shape)
    for i in range(ns):
        delta[(2*i+1)*nag:(2*i+2)*nag] = REC_UP[i-1]*state[(2*i)*nag:(2*i+1)*nag] + REC_SAME[i]*state[(2*i+2)*nag:(2*i+3)*nag]\
            - WANE[i-1]*state[(2*i+1)*nag:(2*i+2)*nag] + WANE[i]*state[(2*i+3)*nag:(2*i+4)*nag]
        delta[(2*i+2)*nag:(2*i+3)*nag] = -(REC_UP[i]+REC_SAME[i])*state[(2*i+2)*nag:(2*i+3)*nag]
    return delta

# vectorized ODEs for SIS model with 3 susceptibility classes
def single_pathogen_deltas(t,state,NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, arrivals, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact):
    delta = np.zeros(state.shape)
    pop_size = np.sum(state,dtype=np.float64)
    age_pops = np.array([np.sum(state[range(i_age,(2*N_S+1)*NAG,NAG)],axis=0) for i_age in range(NAG)])
    # births either straight into youngest age group in first susceptibility class or split between first and S_VAX
    delta += birth_vax(t,BCOV,S_VAX,NAG,N_S,N_C)*birth_rate(t)*pop_size
    # infections are negative for susceptibles and positive for infected
    contact_t = contact(t,SEASONALITY,OFFSET)
    infectious_contact = np.dot(contact_t,np.sum(np.array([state[j] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=np.float64).reshape((N_S,NAG))*I_REL,axis=0))/pop_size
    import_contact = IMPORT_RATE*regional_positivity(t)*arrivals(t)*np.dot(contact_t,age_pops)/pop_size
    delta[NAG:-NAG] += np.repeat(S_REL,NAG*N_C)*np.tile(S_AGE,N_S*N_C)*BETA*np.tile(infectious_contact+import_contact,N_S*N_C)*np.repeat(np.tile(np.array([-1,1]+[0]*(N_C-2)),N_S),NAG)*np.array([np.tile(state[(2*i+1)*NAG:(2*i+2)*NAG],N_C) for i in range(N_S)]).flatten()
    # recovery and waning immunity, which have references to zero buffers at beginning and end of state
    delta += delta_helper(state,NAG,N_S,REC_UP,REC_SAME,WANE)
    # aging
    delta[NAG:-NAG] -= (AGING_RATE*state[NAG:-NAG].reshape((N_C*N_S,NAG))).flatten() # aging out of age groups
    temp_age = AGING_RATE.copy()
    temp_age[-1] = 0
    delta[(NAG+1):-(NAG-1)] += (temp_age*state[NAG:-NAG].reshape((N_C*N_S,NAG))).flatten() # aging into age groups
    # vaccination
    delta[NAG:-NAG] -= (ACOV(t,S_REL*P_OBS,age_pops,AGING_RATE)*np.array([(i%N_C==0 and i//N_C!=S_VAX)*state[(i+1)*NAG:(i+2)*NAG] for i in range(N_S*N_C)])).flatten()
    delta[(S_VAX*N_C+1)*NAG:(S_VAX*N_C+2)*NAG] += ACOV(t,S_REL*P_OBS,age_pops,AGING_RATE)*np.sum(np.array([(i%N_C==0 and i//N_C!=S_VAX)*state[(i+1)*NAG:(i+2)*NAG] for i in range(N_S*N_C)],dtype=np.float64).reshape((N_S*N_C,NAG)),axis=0)
    return delta

def two_pathogen_deltas(t, state, NAG, N_S, AGING_RATE, birth_rate, arrivals, contact, INTERFERENCE,
                        WANE1, REC_UP1, REC_SAME1, S_REL1, S_AGE1, I_REL1, P_OBS1, birth_vax1, all_vax1, S_VAX1, ACOV1, BCOV1, regional_positivity1, IMPORT_RATE1, BETA1, SEASONALITY1, OFFSET1,
                        WANE2, REC_UP2, REC_SAME2, S_REL2, S_AGE2, I_REL2, P_OBS2, birth_vax2, all_vax2, S_VAX2, ACOV2, BCOV2, regional_positivity2, IMPORT_RATE2, BETA2, SEASONALITY2, OFFSET2):
    state1 = state[:(N_C*N_S+1+(3-N_C))*NAG]
    pop_size = np.sum(state1,dtype=np.float64)
    age_pops = np.array([np.sum(state1[range(i_age,(N_C*N_S+1)*NAG,NAG)],axis=0) for i_age in range(NAG)])
    state2 = state[(N_C*N_S+1+(3-N_C))*NAG:]
    delta1 = single_pathogen_deltas(t, state1, NAG, N_S, AGING_RATE, birth_rate, WANE1, REC_UP1, REC_SAME1, S_REL1, S_AGE1, I_REL1, P_OBS1, birth_vax1, all_vax1, S_VAX1, ACOV1, BCOV1, arrivals, regional_positivity1, IMPORT_RATE1, BETA1, SEASONALITY1, OFFSET1, contact)
    delta2 = single_pathogen_deltas(t, state2, NAG, N_S, AGING_RATE, birth_rate, WANE2, REC_UP2, REC_SAME2, S_REL2, S_AGE2, I_REL2, P_OBS2, birth_vax2, all_vax2, S_VAX2, ACOV2, BCOV2, arrivals, regional_positivity2, IMPORT_RATE2, BETA2, SEASONALITY2, OFFSET2, contact)
    # Interference term
    age_infections1 = np.array([np.sum(state1[range(2*NAG+i_age,(N_C*N_S+1)*NAG,N_C*NAG)],axis=0) for i_age in range(NAG)])
    age_infections2 = np.array([np.sum(state2[range(2*NAG+i_age,(N_C*N_S+1)*NAG,N_C*NAG)],axis=0) for i_age in range(NAG)])
    # instead of 1 - the interference term, use the interference term directly to generate a correction to the existing deltas
    s_age_correction1 = S_AGE1*INTERFERENCE[1]*age_infections2/age_pops
    s_age_correction2 = S_AGE2*INTERFERENCE[0]*age_infections1/age_pops
    # then the deltas are decremented by the correction
    contact_t1 = contact(t,SEASONALITY1,OFFSET1)
    infectious_contact1 = np.dot(contact_t1,np.sum(np.array([state1[j] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=np.float64).reshape((N_S,NAG))*I_REL1,axis=0))/pop_size
    import_contact1 = IMPORT_RATE1*regional_positivity1(t)*arrivals(t)*np.dot(contact_t1,age_pops)/pop_size
    delta1[NAG:-NAG] -= np.repeat(S_REL1,NAG*N_C)*np.tile(s_age_correction1,N_S*N_C)*BETA1*np.tile(infectious_contact1+import_contact1,N_S*N_C)*np.repeat(np.tile(np.array([-1,1]+[0]*(N_C-2)),N_S),NAG)*np.array([np.tile(state1[(2*i+1)*NAG:(2*i+2)*NAG],N_C) for i in range(N_S)]).flatten()
    contact_t2 = contact(t,SEASONALITY2,OFFSET2)
    infectious_contact2 = np.dot(contact_t2,np.sum(np.array([state2[j] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=np.float64).reshape((N_S,NAG))*I_REL2,axis=0))/pop_size
    import_contact2 = IMPORT_RATE2*regional_positivity2(t)*arrivals(t)*np.dot(contact_t2,age_pops)/pop_size
    delta2[NAG:-NAG] -= np.repeat(S_REL2,NAG*N_C)*np.tile(s_age_correction2,N_S*N_C)*BETA2*np.tile(infectious_contact2+import_contact2,N_S*N_C)*np.repeat(np.tile(np.array([-1,1]+[0]*(N_C-2)),N_S),NAG)*np.array([np.tile(state2[(2*i+1)*NAG:(2*i+2)*NAG],N_C) for i in range(N_S)]).flatten()

    return np.concatenate((delta1, delta2))