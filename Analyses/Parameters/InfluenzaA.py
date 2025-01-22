## Influenza parameters
## Literature paramdeters on flu, and guesses to match KPSC data

import numpy as np
from numba import jit
from utils import age_detection

NAG = 7

## Parameters that vary by susceptibility class
# Number of susceptibility classes
N_S = 3
# Waning rates for susceptibles into lower susceptibilty class - for index plus one, i.e. [0,1,0] means only last class wanes
WANE = np.array([0.0,1/270,0.0]) # Ferguson 2003
# WANE = np.array([0.0,1/365,0.0])
# Recovery rates for each susceptibility class
REC_UP = np.array([1/3,1/3,0.0]) # Bjornstad 2016
REC_SAME = np.array([0.0,0.0,1/3]) # Bjornstad 2016
# Relative susceptability and infectiousness, for each susceptibility class
S_REL = np.array([1,0.9,0.8]) # maximum flu vaccine effectiveness is 60% (CDC)
I_REL = np.array([[1],[1],[1]])
# Probability of detection of cases for each susceptibility class
P_OBS_REL = np.array([1,0.89,0.4]) # simplest assumption in absence of data
P_OBS_MAX = 0.03
P_OBS = P_OBS_MAX*P_OBS_REL
## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = np.ones(NAG)
# Age-specific relative probability of detection
OBS_AGE = np.array([0.2,0.15,0.1,0.05,0.05,0.2,1])
## Other
# Seasonality parameters
SEASONALITY = 0.1
OFFSET = 0.8
# Infectiousness
BETA = 0.145
# Vaccination paramters
S_VAX, BCOV = 2, 0
@jit
def ACOV(t,S_REL):
    return 0

IMPORT_RATE = 1e-12