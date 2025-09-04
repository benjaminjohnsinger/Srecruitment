## Generic disease parameters
## Blank canvas epi parameters with SIS model in mind

# import jax.numpy as jnp
from numba import jit
from utils import age_detection

NAG = 7

## Parameters that vary by susceptibility class
# Number of susceptibility classes
N_S = 3
# Waning rates for susceptibles into lower susceptibilty class - for index plus one, i.e. [0,1,0] means only last class wanes
WANE = 1/30*jnp.array([0.0,1.0,0.0])/365
# Recovery rates for each susceptibility class
REC_UP = jnp.array([1/5,1/5,0.0]) # Recovery to higher susceptibility class
REC_SAME = jnp.array([0.0,0.0,1/5]) # Recovery to same susceptibility class
# Relative susceptability and infectiousness, for each susceptibility class
ACQUIRED_IMMUNITY = 0.25
S_REL = jnp.linspace(1,(1-(N_S-1)*ACQUIRED_IMMUNITY),N_S)
I_REL = jnp.ones((N_S,1))
# Probability of detection of cases for each susceptibility class
P_OBS = 0.02*jnp.ones(N_S)
## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = jnp.ones(NAG)
# Age-specific probability of detection
AGE_DISEASE_REDUCTION = 0
# OBS_AGE = jnp.linspace(1,(1-(NAG-1)*AGE_DISEASE_REDUCTION),NAG)
OBS_AGE = jnp.array([1,0.6,0.2,0.05,0.1,0.4,0.7])
## Other
# Seasonality parameters
SEASONALITY = 0.05
OFFSET = 0.2
# Infectiousness per contact
BETA = 1/10
# Vaccination paramters
S_VAX, BCOV = 2, 0 
# @jit
def ACOV(t,T_VAX):
    return 0.0