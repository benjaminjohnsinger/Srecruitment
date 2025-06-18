## CDA example disease parameters
## Epidemiological parameters for a disease with annual outbreaks

import numpy as np
import pandas as pd
from numba import jit
# from Analyses.utils import age_detection

NAG = 7

## Parameters that vary by susceptibility class
# Number of susceptibility classes
N_S = 3
# Waning rates for susceptibles into lower susceptibilty class - for index plus one, i.e. [0,1,0] means only last class wanes
WANE = np.array([0.0,1/365,0.0])
# Recovery rates for each susceptibility class
REC_UP = np.array([1/4,1/4,0.0]) # Recovery to higher susceptibility class - Okiro 2010
REC_SAME = np.array([0.0,0.0,1/4]) # Recovery to same susceptibility class - Okiro 2010
# Relative susceptability and infectiousness, for each susceptibility class
S_REL = np.array([1,0.5,0.25])
I_REL = np.array([[1],[1],[1]])
# Probability of detection of cases for each susceptibility class
P_OBS_REL = np.array([1,0.5,0.3]) 
P_OBS_MAX = 0.01
P_OBS = P_OBS_MAX*P_OBS_REL
## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = np.ones(NAG)
# Age-specific relative probability of detection
OBS_AGE = np.array([1,0.75,0.5,0.05,0.05,0.15,1])
## Other
# Seasonality parameters
SEASONALITY = 0.2
OFFSET = 0.2
# Infectiousness
BETA = 0.238
# Vaccination paramters
S_VAX, BCOV = 2, 0
@jit
def BCOV(t):
    return 0
@jit
def ACOV(t,SP,ap,AR):
    return 0

IMPORT_RATE = 0

@jit
def regional_positivity(t):
    return(0)