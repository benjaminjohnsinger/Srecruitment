## Influenza parameters
## Literature paramdeters on flu, and guesses to match KPSC data

# import jax.numpy as jnp

import pandas as pd
# from Analyses.utils import age_detection

NAG = 7

## Parameters that vary by susceptibility class
# Number of susceptibility classes
N_S = 3
# Waning rates for susceptibles into lower susceptibilty class - for index plus one, i.e. [0,1,0] means only last class wanes
WANE = jnp.array([0.0,1/500,0.0]) 
# Recovery rates for each susceptibility class
REC_UP = jnp.array([1/3,1/3,0.0]) # Bjornstad 2016
REC_SAME = jnp.array([0.0,0.0,1/3]) # Bjornstad 2016
# Relative susceptability and infectiousness, for each susceptibility class
S_REL = jnp.array([1,1,0.4]) # maximum flu vaccine effectiveness is 60% (CDC)
I_REL = jnp.array([[1],[1],[1]])
# Probability of detection of cases for each susceptibility class
P_OBS_REL = jnp.array([1,1,1]) # simplest assumption in absence of data
P_OBS_MAX = 0.03
P_OBS = P_OBS_MAX*P_OBS_REL
## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = jnp.ones(NAG)
# Age-specific relative probability of detection
# OBS_AGE = jnp.array([0.2,0.15,0.1,0.05,0.05,0.2,1])
OBS_AGE = jnp.ones(NAG)
## Other
# Seasonality parameters
SEASONALITY = 0.2
OFFSET = 0.8
# Infectiousness
BETA = 0.0884
# Vaccination paramters
S_VAX = 2
# @jit
def BCOV(t):
    return 0
# @jit
def ACOV(t):
    return 0

IMPORT_RATE = 0

PP = pd.read_csv("Data/Processed/FluView_PercentPositive_Regions_B.csv")
PP.index = pd.to_datetime(PP["Date"], format="%Y-%m-%d")
PP.index = (PP.index - pd.to_datetime("1970-01-01")).days
# PP_NP = jnp.array(PP['Region 9'])/100
PP_NP = jnp.array(PP[['Region '+str(i) for i in range(1,9)] + ['Region 10']].mean(axis=1))/100
PP_IDX = jnp.array(PP.index)
# @jit
def regional_positivity(t, PP_NP=PP_NP, PP_IDX=PP_IDX):
    if t < PP_IDX[0] or t > PP_IDX[-1]:
        day_in_season = (t + 92)%365
        time_1998 = 10500 + day_in_season
        return PP_NP[jnp.argmax(PP_IDX>=time_1998)]
    return PP_NP[jnp.argmax(PP_IDX>=t)]