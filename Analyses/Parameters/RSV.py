## RSV parameters
## Parameters from Pitzer et al. 2015 edited to roughly fit KPSC observations

import numpy as np
from numba import jit
from utils import age_detection

NAG = 7

## Parameters that vary by susceptibility class
# Number of susceptibility classes
N_S = 3
# Waning rates for susceptibles into lower susceptibilty class - for index plus one, i.e. [0,1,0] means only last class wanes
WANE = np.array([0.0,1/(30*365),0.0])
# WANE = 7.82131488e-05*np.array([0,1,0])
# Recovery rates for each susceptibility class
REC_UP = np.array([1/4.9,1/4.1,0.0]) # Recovery to higher susceptibility class - Okiro 2010
REC_SAME = np.array([0.0,0.0,1/4.1]) # Recovery to same susceptibility class - Okiro 2010
# Relative susceptability and infectiousness, for each susceptibility class
S_REL = np.array([1,0.559,0.333]) # Glezen 1986
I_REL = np.array([[1],[1],[1]])
# Probability of detection of cases for each susceptibility class
P_OBS_REL = np.array([1,0.46,0.31]) # Henderson 1979
# P_OBS_MAX = 2.29911325e-02
P_OBS_MAX = 0.03
P_OBS = P_OBS_MAX*P_OBS_REL
## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = np.ones(NAG)
# Age-specific relative probability of detection
OBS_AGE = np.array([1,0.75,0.5,0.05,0.05,0.15,1])
# OBS_AGE = np.array([1.,0.38763672,0.15026224,0.05825276,0.02450205,0.66893976])
## Other
# Seasonality parameters
# SEASONALITY = 0.1
SEASONALITY = 9.32959247e-02
OFFSET = 0.85
# OFFSET = 9.08955903e-02
# Infectiousness
BETA = 0.14
# BETA = 1.23970813e-01
# BETA = 0.14
# Vaccination paramters
S_VAX, BCOV = 2, 0
@jit
def ACOV(t,T_VAX):
    return 0