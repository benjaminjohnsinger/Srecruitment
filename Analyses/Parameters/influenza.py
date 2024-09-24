import numpy as np

## Flu-like parameters
## Parameters that vary by susceptibility class
# Number of susceptibility classes
N_S = 3
# Immunity waning rates for each susceptibility class, with prefixed zero
WANE_UP = np.array([1/3,1/3,0]) # Waning to higher susceptibility class (last value always 0)
WANE_SAME = np.array([0,0,1/3]) # Waning to same susceptibility class
# Recovery rates for each susceptibility class
REC = np.array([4.3,8.6,8.6])
# Relative susceptability and infectiousness, for each susceptibility class
S_REL = np.array([1,0.8,0.6])
I_REL = np.array([[1],[0.9],[0.8]])
# Probability of detection of cases for each susceptibility class
P_OBS = 0.4*np.array([0.1,0.05,0.05])
## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = np.array([1,1,1,1,1,1,1])
# Age-specific probability of detection
OBS_AGE = np.array([0.5,0.5,0.1,0.1,0.1,0.1,1])
## Other parameters
# Factor to adust FOI
BETA_FUDGE_FACTOR = 0.7
# Vaccination paramters
S_VAX, BCOV = 2, 0
def ACOV(t,T_VAX):
    return 0.04 if t > T_VAX else 0