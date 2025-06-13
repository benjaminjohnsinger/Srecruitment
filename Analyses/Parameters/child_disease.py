## Child disease parameters
## Epidemiological parameters for a rota-like disease

import jax.numpy as jnp

# ## Rota-like parameters
## Parameters that vary by susceptibility class
# Number of susceptibility classes
N_S = 3
# Immunity waning rates for each susceptibility class, with prefixed zero
WANE_UP = jnp.array([1/9,1/9,0]) # Waning to higher susceptibility class (last value always 0)
WANE_SAME = jnp.array([0,0,1/12]) # Waning to same susceptibility class
# Recovery rates for each susceptibility class
REC = jnp.array([4.3,8.6,8.6])
# Relative susceptability and infectiousness, for each susceptibility class
S_REL = jnp.array([1,0.62,0.35])
I_REL = jnp.array([[1],[0.5],[0.1]])
# Probability of detection of cases for each susceptibility class
P_OBS = 0.041*jnp.array([0.11,0.029,0])
## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = jnp.array([1,1,1,1,1,1,1])
# Age-specific probability of detection
OBS_AGE = jnp.array([1,1,1,1,1,1,1])
## Other parameters
# Seasonality parameters
SEASONALITY = 0.05
OFFSET = 0
# Infectiousness
BETA = 130
# Vaccination paramters
S_VAX, BCOV = 2, 0.8*0.96
def ACOV(t,T_VAX):
    return 0