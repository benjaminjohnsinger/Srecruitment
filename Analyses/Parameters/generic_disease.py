## Generic disease parameters
## Blank canvas epi parameters with SIS model in mind

import numpy as np

NAG = 7

## Parameters that vary by susceptibility class
# Number of susceptibility classes
N_S = 3
# Waning rates for susceptibles into lower susceptibilty class - for index plus one, i.e. [0,1,0] means only last class wanes
WANE = 1/30*np.array([0.0,1.0,0.0])/30.44
# Recovery rates for each susceptibility class
REC_UP = 5*np.array([1.0,1.0,0.0])/30.44 # Recovery to higher susceptibility class
REC_SAME = 5*np.array([0.0,0.0,1.0])/30.44 # Recovery to same susceptibility class
# Relative susceptability and infectiousness, for each susceptibility class
AQUIRED_IMMUNITY = 0.25
S_REL = np.linspace(1,(1-(N_S-1)*AQUIRED_IMMUNITY),N_S)
I_REL = np.ones((N_S,1))
# Probability of detection of cases for each susceptibility class
P_OBS = 0.01*np.ones(N_S)
## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = np.ones(NAG)
# Age-specific probability of detection
AGE_DISEASE_REDUCTION = 0
OBS_AGE = np.linspace(1,(1-(NAG-1)*AGE_DISEASE_REDUCTION),NAG)
## Other 
# Seasonality parameters
SEASONALITY = 0.05
OFFSET = 0
# Infectiousness per contact
BETA = 1/10
# Vaccination paramters
S_VAX, BCOV = 2, 0
def ACOV(t,T_VAX):
    return 0