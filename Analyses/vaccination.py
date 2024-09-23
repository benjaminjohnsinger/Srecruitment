import numpy as np

# Vaccination of infants
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
# Annual mass vaccination 
# S_VAX is the susceptibility class of vaccinated individuals
# COVERAGE is the proportion of the population vaccinated each month - this can be an age-dependent vector
# T_VAX is the time at which vaccination starts
def all_vax(t,compartment,S_VAX=2,COVERAGE=0,T_VAX=0,NAG=7,N_S=3):
    if compartment != S_VAX*3+1:
        vec = np.zeros((3*N_S+1)*NAG)
        vec[compartment*NAG:(compartment+1)*NAG] = -1
        return 0 if t < T_VAX else COVERAGE*vec
    else:
        vec = np.ones((3*N_S+1)*NAG)
        vec[compartment*NAG:(compartment+1)*NAG] = 0
        return 0 if t < T_VAX else COVERAGE*vec