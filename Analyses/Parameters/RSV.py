## RSV parameters
## Parameters from literature cited by Pitzer et al. 2015, and guesses to match KPSC data

import numpy as np
import pandas as pd
from numba import jit
# from Analyses.utils import age_detection

NAG = 7

## Parameters that vary by susceptibility class
# Number of susceptibility classes
N_S = 3
# Waning rates for susceptibles into lower susceptibilty class - for index plus one, i.e. [0,1,0] means only last class wanes
WANE = np.array([0.0,1/(30*365),0.0])
# WANE = np.array([0,1.195e-01,0])/365
# Recovery rates for each susceptibility class
REC_UP = np.array([1/4.9,1/4.1,0.0]) # Recovery to higher susceptibility class - Okiro 2010
REC_SAME = np.array([0.0,0.0,1/4.1]) # Recovery to same susceptibility class - Okiro 2010
# Relative susceptability and infectiousness, for each susceptibility class
S_REL = np.array([1,0.559,0.333]) # Glezen 1986
# S_REL = np.array([1,0.25,0.025]) # Nokes 2008
I_REL = np.array([[1],[1],[1]])
# Probability of detection of cases for each susceptibility class
P_OBS_REL = np.array([1,0.46,0.31]) # Henderson 1979
# P_OBS_MAX = 3.396e-02
P_OBS_MAX = 0.03
P_OBS = P_OBS_MAX*P_OBS_REL
## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = np.ones(NAG)
# Age-specific relative probability of detection
OBS_AGE = np.array([1,0.75,0.5,0.05,0.05,0.15,1])
# OBS_AGE = np.array([1,0.8229,0.6458,0.4687,0.2916,0.1375,0.8626])
## Other
# Seasonality parameters
SEASONALITY = 0.25
# SEASONALITY = 1.727e-01
OFFSET = 0.85
# OFFSET = 8.584e-01
# Infectiousness
BETA = 0.11
# BETA = 6.983e-02
# BETA = 0.18
# Vaccination paramters
S_VAX, BCOV = 2, 0
@jit
def BCOV(t):
    return 0
@jit
def ACOV(t,SP,ap,AR):
    return 0

# IMPORT_RATE = 7.255e-13
IMPORT_RATE = 1e-11

PP = pd.read_csv("Data/Processed/RSV_PercentPositive_Regions.csv")
PP.index = pd.to_datetime(PP["Date"], format="%Y-%m-%d")
PP.index = (PP.index - pd.to_datetime("1970-01-01")).days
# PP_NP = np.array(PP['Region 9'])
PP_NP = np.array(PP[['Region '+str(i) for i in range(1,9)] + ['Region 10']].mean(axis=1))
PP_IDX = np.array(PP.index)
@jit
def regional_positivity(t):
    return PP_NP[np.argmax(PP_IDX>=t)]