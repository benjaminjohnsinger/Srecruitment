import numpy as np

# Vaccination of infants (a proportion)
# S_VAX is the susceptibility class of vaccinated individuals
# COVERAGE is the proportion of infants vaccinated
# T_VAX is the time at which vaccination starts
def birth_vax(t,s_class,S_VAX=2,COVERAGE=0,T_VAX=0):
    if t < T_VAX:
        return 1 if (s_class == 0) else 0
    else:
        if s_class == S_VAX:
            return COVERAGE
        elif s_class == 0:
            return 1-COVERAGE
        else:
            return 0
# Annual mass vaccination (a rate)
# S_VAX is the susceptibility class of vaccinated individuals
# coverage is the time-varying proportion of the population vaccinated each month - this can be an age-dependent vector
def all_vax(t,s_class,coverage,S_VAX=2,T_VAX=0,NAG=7,N_S=3,N_C=3):
    if s_class != S_VAX:
        # (3-N_C) here is a really hacky way of making this work with SIS model, which needs to reference an extra empty compartment
        vec = np.zeros((N_C*N_S+1+(3-N_C))*NAG)
        vec[(N_C*s_class+1)*NAG:(N_C*s_class+2)*NAG] = -1
        return 0 if t < T_VAX else coverage(t,T_VAX)*vec
    else:
        vec = np.zeros((N_C*N_S+1+(3-N_C))*NAG)
        for i in range(N_S):
            if i != s_class:
                vec[(i*N_C+1)*NAG:(i*N_C+2)*NAG] = 1
        return 0 if t < T_VAX else coverage(t,T_VAX)*vec