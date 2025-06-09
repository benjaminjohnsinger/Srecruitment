## Ordidxnary differential equations defining SIR models with n susceptibility classes
## BJS September 2024

# from numba import jit
import jax.numpy as jnp
import numpy as np
N_C = 2 # two types of compartment

# separate function for recovery and waning immunity, since this is fasted in a numba-compiled for loop
# @jit
def delta_helper(state,nag,ns,REC_UP,REC_SAME,WANE):
    delta = jnp.zeros(state.shape)
    for i in range(ns):
        delta = delta.at[(2*i+1)*nag:(2*i+2)*nag].set(REC_UP[i-1]*state[(2*i)*nag:(2*i+1)*nag] + REC_SAME[i]*state[(2*i+2)*nag:(2*i+3)*nag]\
            - WANE[i-1]*state[(2*i+1)*nag:(2*i+2)*nag] + WANE[i]*state[(2*i+3)*nag:(2*i+4)*nag])
        delta = delta.at[(2*i+2)*nag:(2*i+3)*nag].set(-(REC_UP[i]+REC_SAME[i])*state[(2*i+2)*nag:(2*i+3)*nag])
    return delta

# vectorized ODEs for SIS model with 3 susceptibility classes
def single_pathogen_deltas(t, state, args):
    NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, S_VAX, ACOV, BCOV, arrivals, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact = args
    delta = jnp.zeros(state.shape)
    pop_size = jnp.sum(state)
    age_pops = jnp.sum(state.reshape(((2*N_S+2),NAG)),axis=0)
    # births either straight into youngest age group in first susceptibility class or split between first and S_VAX
    delta = delta + birth_vax(t,BCOV,S_VAX,NAG,N_S,N_C)*birth_rate(t)*pop_size
    # infections are negative for susceptibles and positive for infected
    contact_t = contact(t,SEASONALITY,OFFSET)
    infectious_contact = jnp.dot(contact_t,jnp.sum(jnp.array([state[j] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0]).reshape((N_S,NAG))*I_REL,axis=0))/pop_size
    import_contact = IMPORT_RATE*regional_positivity(t)*arrivals(t)*jnp.dot(contact_t,age_pops)/pop_size
    infection = jnp.repeat(S_REL,NAG*N_C)*jnp.tile(S_AGE,N_S*N_C)*BETA*jnp.tile(infectious_contact+import_contact,N_S*N_C)*jnp.repeat(jnp.tile(jnp.array([-1,1]+[0]*(N_C-2)),N_S),NAG)*jnp.array([jnp.tile(state[(2*i+1)*NAG:(2*i+2)*NAG],N_C) for i in range(N_S)]).flatten()
    # recovery and waning immunity, which have references to zero buffers at beginning and end of state
    delta = delta + delta_helper(state,NAG,N_S,REC_UP,REC_SAME,WANE)
    # aging
    aging_out = (AGING_RATE*state[NAG:-NAG].reshape((N_C*N_S,NAG))).flatten() # aging out of age groups
    temp_age = AGING_RATE.copy()
    temp_age = temp_age.at[-1].set(0)
    aging_in = (temp_age*state[NAG:-NAG].reshape((N_C*N_S,NAG))).flatten() # aging into age groups
    delta = delta.at[(NAG+1):-(NAG-1)].set(delta[(NAG+1):-(NAG-1)] + aging_in)
    # vaccination
    vax_out = (ACOV(t,S_REL*P_OBS,age_pops,AGING_RATE)*jnp.array([(i%N_C==0 and i//N_C!=S_VAX)*state[(i+1)*NAG:(i+2)*NAG] for i in range(N_S*N_C)])).flatten()
    vax_in = ACOV(t,S_REL*P_OBS,age_pops,AGING_RATE)*jnp.sum(jnp.array([(i%N_C==0 and i//N_C!=S_VAX)*state[(i+1)*NAG:(i+2)*NAG] for i in range(N_S*N_C)]).reshape((N_S*N_C,NAG)),axis=0)
    delta = delta.at[(S_VAX*N_C+1)*NAG:(S_VAX*N_C+2)*NAG].set(delta[(S_VAX*N_C+1)*NAG:(S_VAX*N_C+2)*NAG] + vax_in)
    
    delta = delta.at[NAG:-NAG].set(delta[NAG:-NAG] + infection - aging_out - vax_out)
    return delta

# def two_pathogen_deltas(t, state, NAG, N_S, AGING_RATE, birth_rate, arrivals, contact, INTERFERENCE,
#                         WANE1, REC_UP1, REC_SAME1, S_REL1, S_AGE1, I_REL1, P_OBS1, birth_vax1, all_vax1, S_VAX1, ACOV1, BCOV1, regional_positivity1, IMPORT_RATE1, BETA1, SEASONALITY1, OFFSET1,
#                         WANE2, REC_UP2, REC_SAME2, S_REL2, S_AGE2, I_REL2, P_OBS2, birth_vax2, all_vax2, S_VAX2, ACOV2, BCOV2, regional_positivity2, IMPORT_RATE2, BETA2, SEASONALITY2, OFFSET2):
#     state1 = state[:(N_C*N_S+1+(3-N_C))*NAG]
#     pop_size = jnp.sum(state1)
#     age_pops = jnp.array([jnp.sum(state1[range(i_age,(N_C*N_S+1)*NAG,NAG)],axis=0) for i_age in range(NAG)])
#     state2 = state[(N_C*N_S+1+(3-N_C))*NAG:]
#     delta1 = single_pathogen_deltas(t, state1, NAG, N_S, AGING_RATE, birth_rate, WANE1, REC_UP1, REC_SAME1, S_REL1, S_AGE1, I_REL1, P_OBS1, birth_vax1, all_vax1, S_VAX1, ACOV1, BCOV1, arrivals, regional_positivity1, IMPORT_RATE1, BETA1, SEASONALITY1, OFFSET1, contact)
#     delta2 = single_pathogen_deltas(t, state2, NAG, N_S, AGING_RATE, birth_rate, WANE2, REC_UP2, REC_SAME2, S_REL2, S_AGE2, I_REL2, P_OBS2, birth_vax2, all_vax2, S_VAX2, ACOV2, BCOV2, arrivals, regional_positivity2, IMPORT_RATE2, BETA2, SEASONALITY2, OFFSET2, contact)
#     # Interference term
#     age_infections1 = jnp.array([jnp.sum(state1[range(2*NAG+i_age,(N_C*N_S+1)*NAG,N_C*NAG)],axis=0) for i_age in range(NAG)])
#     age_infections2 = jnp.array([jnp.sum(state2[range(2*NAG+i_age,(N_C*N_S+1)*NAG,N_C*NAG)],axis=0) for i_age in range(NAG)])
#     # instead of 1 - the interference term, use the interference term directly to generate a correction to the existing deltas
#     s_age_correction1 = S_AGE1*INTERFERENCE[1]*age_infections2/age_pops
#     s_age_correction2 = S_AGE2*INTERFERENCE[0]*age_infections1/age_pops
#     # then the deltas are decremented by the correction
#     contact_t1 = contact(t,SEASONALITY1,OFFSET1)
#     infectious_contact1 = jnp.dot(contact_t1,jnp.sum(jnp.array([state1[j] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0]).reshape((N_S,NAG))*I_REL1,axis=0))/pop_size
#     import_contact1 = IMPORT_RATE1*regional_positivity1(t)*arrivals(t)*jnp.dot(contact_t1,age_pops)/pop_size
#     delta1[NAG:-NAG] -= jnp.repeat(S_REL1,NAG*N_C)*jnp.tile(s_age_correction1,N_S*N_C)*BETA1*jnp.tile(infectious_contact1+import_contact1,N_S*N_C)*jnp.repeat(jnp.tile(jnp.array([-1,1]+[0]*(N_C-2)),N_S),NAG)*jnp.array([jnp.tile(state1[(2*i+1)*NAG:(2*i+2)*NAG],N_C) for i in range(N_S)]).flatten()
#     contact_t2 = contact(t,SEASONALITY2,OFFSET2)
#     infectious_contact2 = jnp.dot(contact_t2,jnp.sum(jnp.array([state2[j] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0]).reshape((N_S,NAG))*I_REL2,axis=0))/pop_size
#     import_contact2 = IMPORT_RATE2*regional_positivity2(t)*arrivals(t)*jnp.dot(contact_t2,age_pops)/pop_size
#     delta2[NAG:-NAG] -= jnp.repeat(S_REL2,NAG*N_C)*jnp.tile(s_age_correction2,N_S*N_C)*BETA2*jnp.tile(infectious_contact2+import_contact2,N_S*N_C)*jnp.repeat(jnp.tile(jnp.array([-1,1]+[0]*(N_C-2)),N_S),NAG)*jnp.array([jnp.tile(state2[(2*i+1)*NAG:(2*i+2)*NAG],N_C) for i in range(N_S)]).flatten()

#     return jnp.concatenate((delta1, delta2))