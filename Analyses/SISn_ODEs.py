## Ordidxnary differential equations defining SIR models with n susceptibility classes
## BJS September 2024

from numba import jit
# # import jax.numpy as jnp
import numpy as np
# import contact_model as cm
N_C = 2 # two types of compartment

# separate function for recovery and waning immunity, since this is faster in a numba-compiled for loop
# @jit
def delta_helper(state,nag,ns,REC_UP,REC_SAME,WANE):
    delta = np.zeros(state.shape)
    for i in range(ns):
        # delta = delta.at[(2*i+1)*nag:(2*i+2)*nag].set(REC_UP[i-1]*state[(2*i)*nag:(2*i+1)*nag] + REC_SAME[i]*state[(2*i+2)*nag:(2*i+3)*nag]\
        #     - WANE[i-1]*state[(2*i+1)*nag:(2*i+2)*nag] + WANE[i]*state[(2*i+3)*nag:(2*i+4)*nag])
        delta[(2*i+1)*nag:(2*i+2)*nag] = REC_UP[i-1]*state[(2*i)*nag:(2*i+1)*nag] + REC_SAME[i]*state[(2*i+2)*nag:(2*i+3)*nag]\
            - WANE[i-1]*state[(2*i+1)*nag:(2*i+2)*nag] + WANE[i]*state[(2*i+3)*nag:(2*i+4)*nag]
        # delta = delta.at[(2*i+2)*nag:(2*i+3)*nag].set(-(REC_UP[i]+REC_SAME[i])*state[(2*i+2)*nag:(2*i+3)*nag])
        delta[(2*i+2)*nag:(2*i+3)*nag] = -(REC_UP[i]+REC_SAME[i])*state[(2*i+2)*nag:(2*i+3)*nag]
    return delta

# vectorized ODEs for SIS model with 3 susceptibility classes
def single_pathogen_deltas(t, state, NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, S_VAX, ACOV, BCOV, arrivals, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact):
    delta = np.zeros(state.shape)
    pop_size = np.sum(state)
    age_pops = np.sum(state.reshape(((2*N_S+2),NAG)),axis=0)
    # births either straight into youngest age group in first susceptibility class or split between first and S_VAX
    delta = delta + birth_vax(t,BCOV,S_VAX,NAG,N_S,N_C)*birth_rate(t)*pop_size
    # infections are negative for susceptibles and positive for infected
    contact_t = contact(t,SEASONALITY,OFFSET)
    infectious_contact = np.dot(contact_t,np.sum(np.array([state[j] for j in range(NAG,(N_C*N_S+1)*NAG) if ((j//NAG)-1)%N_C==1]).reshape((N_S,NAG))*I_REL,axis=0))/pop_size
    import_contact = IMPORT_RATE*regional_positivity(t)*arrivals(t)*np.dot(contact_t,age_pops)/pop_size
    infection = np.repeat(S_REL,NAG*N_C)*np.tile(S_AGE,N_S*N_C)*BETA*np.tile(infectious_contact+import_contact,N_S*N_C)*np.repeat(np.tile(np.array([-1,1]+[0]*(N_C-2)),N_S),NAG)*np.array([np.tile(state[(2*i+1)*NAG:(2*i+2)*NAG],N_C) for i in range(N_S)]).flatten()
    # recovery and waning immunity, which have references to zero buffers at beginning and end of state
    delta = delta + delta_helper(state,NAG,N_S,REC_UP,REC_SAME,WANE)
    # aging
    aging_out = (AGING_RATE*state[NAG:-NAG].reshape((N_C*N_S,NAG))).flatten() # aging out of age groups
    temp_age = AGING_RATE.copy()
    # temp_age = temp_age.at[-1].set(0)
    temp_age[-1] = 0
    aging_in = (temp_age*state[NAG:-NAG].reshape((N_C*N_S,NAG))).flatten() # aging into age groups
    # delta = delta.at[(NAG+1):-(NAG-1)].add(aging_in)
    delta[(NAG+1):-(NAG-1)] += aging_in
    # vaccination
    vax_out = (ACOV(t,S_REL*P_OBS,age_pops,AGING_RATE)*np.array([(i%N_C==0 and i//N_C!=S_VAX)*state[(i+1)*NAG:(i+2)*NAG] for i in range(N_S*N_C)])).flatten()
    vax_in = ACOV(t,S_REL*P_OBS,age_pops,AGING_RATE)*np.sum(np.array([(i%N_C==0 and i//N_C!=S_VAX)*state[(i+1)*NAG:(i+2)*NAG] for i in range(N_S*N_C)]).reshape((N_S*N_C,NAG)),axis=0)
    # delta = delta.at[(S_VAX*N_C+1)*NAG:(S_VAX*N_C+2)*NAG].add(vax_in)
    delta[(S_VAX*N_C+1)*NAG:(S_VAX*N_C+2)*NAG] += vax_in
    
    # delta = delta.at[NAG:-NAG].add(infection - aging_out - vax_out)
    delta[NAG:-NAG] += infection - aging_out - vax_out
    return delta