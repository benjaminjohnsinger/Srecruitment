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
WANE = np.array([0.0,1/365,0.0])
# Recovery rates for each susceptibility class
REC_UP = np.array([1/3,1/3,0.0])
REC_SAME = np.array([0.0,0.0,1/3])
# Relative susceptability and infectiousness, for each susceptibility class
S_REL = np.array([1,0.25,0.025])
I_REL = np.array([[1],[1],[1]])
# Probability of detection of cases for each susceptibility class
P_OBS_REL = np.array([1,0.5,0.25])
P_OBS_MAX = 0.03
P_OBS = P_OBS_MAX*P_OBS_REL
## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = np.ones(NAG)
# Age-specific relative probability of detection
OBS_AGE = np.array([1,0.75,0.5,0.05,0.05,0.15,1])
## Other
# Seasonality parameters
SEASONALITY = 0.4
OFFSET = 0.2
# Infectiousness
BETA = 0.5
# Vaccination paramters
S_VAX, BCOV = 2, 0
@jit
def BCOV(t):
    return 0
@jit
def ACOV(t,SP,ap,AR):
    return 0

IMPORT_RATE = 0

PP = pd.read_csv("Data/Processed/NREVSS_PCR_PercentPositive_PIV3.csv")
PP.index = pd.to_datetime(PP["Date"], format="%Y-%m-%d")
PP.index = (PP.index - pd.to_datetime("1970-01-01")).days
# PP_NP = np.array(PP['Region 9'])/100
PP_NP = np.array(PP[['Region '+str(i) for i in range(1,9)] + ['Region 10']].mean(axis=1))/100
PP_IDX = np.array(PP.index)
@jit
def regional_positivity(t):
    if t < 14883 or t > PP_IDX[-1]:
        day_in_season = (t + 92)%365
        time_2010 = 14883 + day_in_season
        return PP_NP[np.argmax(PP_IDX>=time_2010)]
    return PP_NP[np.argmax(PP_IDX>=t)]