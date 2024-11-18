## Pitzer RSV parameters
## Parameters from Pitzer et al. 2015 in PLOS Pathogens

import numpy as np
from numba import jit

NAG = 7

## Parameters that vary by susceptibility class
# Number of susceptibility classes
N_S = 3
# Waning rates for susceptibles into lower susceptibilty class - for index plus one, i.e. [0,1,0] means only last class wanes
WANE = np.array([0.0,1/(30*365),0.0])
# Recovery rates for each susceptibility class
REC_UP = np.array([1/10,1/5,0.0]) # Recovery to higher susceptibility class
REC_SAME = np.array([0.0,0.0,1/5]) # Recovery to same susceptibility class
# Relative susceptability and infectiousness, for each susceptibility class
S_REL = np.array([1,0.76,0.4])
I_REL = np.array([[1],[1],[1]])
# Probability of detection of cases for each susceptibility class
P_OBS = 0.06*np.array([1,0.75,0])
## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = np.ones(NAG)
# Age-specific probability of detection
OBS_AGE = np.array([0.5,0.37,0.12,0.005,0.005,0.01,0.1])
## Other 
# Seasonality parameters
SEASONALITY = 0.2
OFFSET = 0
# Infectiousness
BETA = 1/7
# Vaccination paramters
S_VAX, BCOV = 2, 0
@jit
def ACOV(t,T_VAX):
    return 0