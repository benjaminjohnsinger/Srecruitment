## RSV parameters
## Parameters from Lowensteyn et al. 2023

# import jax.numpy as jnp
import pandas as pd

import contact_model as cm
from utils import date_to_t

NAG = 7

## Parameters that vary by susceptibility class
# Number of susceptibility classes
N_S = 3
# Waning rates for susceptibles into lower susceptibilty class - for index plus one, i.e. [0,1,0] means only last class wanes
WANE = jnp.array([0.0,1/365,0.0])
# Recovery rates for each susceptibility class
REC_UP = jnp.array([1/10,1/7,0.0])
REC_SAME = jnp.array([0.0,0.0,1/5])
# Relative susceptability and infectiousness, for each susceptibility class
S_REL = jnp.array([1,0.76,0.4])
I_REL = jnp.array([[1],[0.75],[0.51]])
# Probability of detection of cases for each susceptibility class
P_OBS_REL = jnp.array([1,0.4,0])
P_OBS_MAX = 0.072*0.45
P_OBS = P_OBS_MAX*P_OBS_REL
## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = jnp.ones(NAG)
# Age-specific relative probability of detection
OBS_AGE = jnp.array([0.072,0.016,0.007,0.001,0.001,0.00001,0.00004])/(0.072*0.45)
## Other
# Seasonality parameters
SEASONALITY = 0.13
OFFSET = (274/365)+3.65/(2*jnp.pi)
# Infectiousness
# BETA = 0.054
BETA = 0.046
# BETA = 0.045

# Vaccination paramters
S_VAX, BCOV = 2, 0
# @jit
def BCOV(t):
    return 0
# @jit
def ACOV(t,SP,ap,AR):
    return 0

CONTACT = jnp.genfromtxt('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', dtype=jnp.float64)
Ts = jnp.array([date_to_t('1970-01-01'),
date_to_t('2020-03-19'), # Newsom announces stay-at-home order
date_to_t('2021-04-27'), # CDC amends mask guidance to allow vaccinated individuals to go maskless
date_to_t('2021-12-15'), # CDC reinstates mask guidance
date_to_t('2022-03-01')]) # End of mask mandate in California
Fs = jnp.array([1,0.72,1,0.72,1]) # 0.2 minimum relative contact rate between COMIX and POLYMOD
# @jit
def contact(t,seasonality,offset):
    return cm.piecewise(t,Ts,Fs)*(1+seasonality*jnp.cos(2*jnp.pi*((t-274)/365-offset)))*CONTACT
@jit
# def contact(t,seasonality,offset):
#     return cm.google_prestige_work(t)*(1+seasonality*jnp.cos(2*jnp.pi*((t-274)/365-offset)))*CONTACT


# IMPORT_RATE = 20/30.44
IMPORT_RATE = 0.01
# IMPORT_RATE = 0

PP = pd.read_csv("Data/Processed/RSV_PercentPositive_Regions.csv")
PP.index = pd.to_datetime(PP["Date"], format="%Y-%m-%d")
PP.index = (PP.index - pd.to_datetime("1970-01-01")).days
# PP_NP = jnp.array(PP['Region 9'])/100
PP_NP = jnp.array(PP[['Region '+str(i) for i in range(1,9)] + ['Region 10']].mean(axis=1))/100
PP_IDX = jnp.array(PP.index)
# @jit
def regional_positivity(t):
    if t < PP_IDX[0] or t > PP_IDX[-1]:
        day_in_season = (t + 92)%365
        time_2010 = 14883 + day_in_season
        return PP_NP[jnp.argmax(PP_IDX>=time_2010)]
    return PP_NP[jnp.argmax(PP_IDX>=t)]
@jit
# def regional_positivity(t):
#     return 0.01