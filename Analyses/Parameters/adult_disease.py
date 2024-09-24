import numpy as np

## Flu-like parameters
## Parameters that vary by susceptibility class
# Number of susceptibility classes
N_S = 3
# Immunity waning rates for each susceptibility class, with prefixed zero
WANE = 1/3
WANE_UP = WANE*np.array([1,1,0]) # Waning to higher susceptibility class (last value always 0)
WANE_SAME = WANE*np.array([0,0,1]) # Waning to same susceptibility class
# Recovery rates for each susceptibility class
REC = 5*np.ones(3)
# Relative susceptability and infectiousness, for each susceptibility class
S_REL = np.array([1,0.8,0.6])
I_REL = np.array([[1],[0.9],[0.8]])
# Probability of detection of cases for each susceptibility class
P_OBS = 0.01*np.array([4,2,2])
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
BETA = 30
# Vaccination paramters
S_VAX, BCOV = 2, 0
def ACOV(t,T_VAX):
    return 0.04 if t > T_VAX else 0