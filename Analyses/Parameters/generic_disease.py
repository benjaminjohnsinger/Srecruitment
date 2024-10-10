## Generic disease parameters
## Blank canvas epi parameters with SIS model in mind

import numpy as np

## Parameters that vary by susceptibility class
# Number of susceptibility classes
N_S = 3
# Immunity waning rates for each susceptibility class
WANE = 1/12*np.array([0,0,1,0])
# Recovery rates for each susceptibility class
REC = 10*np.ones(N_S)
# Relative susceptability and infectiousness, for each susceptibility class
S_REL = np.ones(N_S)
I_REL = np.ones((N_S,1))
# Probability of detection of cases for each susceptibility class
P_OBS = 0.01*np.ones(N_S)
## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = np.ones(7)
# Age-specific probability of detection
OBS_AGE = np.ones(7)
## Other 
# Seasonality parameters
SEASONALITY = 0.05
OFFSET = 0
# Infectiousness
BETA = 55
# Vaccination paramters
S_VAX, BCOV = 2, 0
def ACOV(t,T_VAX):
    return 0