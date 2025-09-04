## Adult disease parameters
## Epidemiological parameters for a flu-like disease

# import jax.numpy as jnp

## Flu-like parameters
## Parameters that vary by susceptibility class
# Number of susceptibility classes
N_S = 3
# Immunity waning rates for each susceptibility class, with prefixed zero
WANE = 1/3
WANE_UP = WANE*jnp.array([1,1,0]) # Waning to higher susceptibility class (last value always 0)
WANE_SAME = WANE*jnp.array([0,0,1]) # Waning to same susceptibility class
# Recovery rates for each susceptibility class
REC = 10*jnp.ones(3)
# Relative susceptability and infectiousness, for each susceptibility class
S_REL = jnp.array([1,0.8,0.6])
I_REL = jnp.ones((N_S,1))
# Probability of detection of cases for each susceptibility class
P_OBS = 0.01*jnp.array([4,2,2])
## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = jnp.ones(7)
# Age-specific probability of detection
OBS_AGE = jnp.ones(7)
## Other 
# Seasonality parameters
SEASONALITY = 0.05
OFFSET = 0
# Infectiousness
BETA = 55
# Vaccination paramters
S_VAX, BCOV = 2, 0
def ACOV(t,T_VAX):
    return 0.04 if t > T_VAX else 0