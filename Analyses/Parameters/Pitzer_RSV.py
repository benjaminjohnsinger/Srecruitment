## Pitzer RSV parameters
## Parameters from Pitzer et al. 2015 in PLOS Pathogens

import numpy as np
from numba import jit

NAG = 25

## Parameters that vary by susceptibility class
# Number of susceptibility classes
N_S = 4
# Waning rates for susceptibles into lower susceptibilty class - for index plus one, i.e. [0,1,0] means only last class wanes
WANE = np.zeros(N_S)
# Recovery rates for each susceptibility class
REC_UP = np.array([1/10,1/7,1/5,0.0]) # Recovery to higher susceptibility class
REC_SAME = np.array([0.0,0.0,0.0,1/5]) # Recovery to same susceptibility class
# Relative susceptability and infectiousness, for each susceptibility class
S_REL = np.array([1,0.76,0.6,0.4])
I_REL = np.array([[1],[0.75],[0.51],[0.51]])
# Probability of detection of cases for each susceptibility class
P_OBS = 0.032*np.array([1,0.75,0,0])
## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = np.ones(NAG)
# Age-specific probability of detection
OBS_AGE = np.concatenate((np.array([0.5,0.3,0.2,0.1,0.1,0.1]),0.1*np.ones(NAG-6)))
## Other 
# Seasonality parameters
SEASONALITY = 0.2
OFFSET = 0
# Infectiousness
BETA = 0.0562 # = 8.9/(10*15.85)
# Vaccination paramters
S_VAX, BCOV = 2, 0
@jit
def ACOV(t,T_VAX):
    return 0