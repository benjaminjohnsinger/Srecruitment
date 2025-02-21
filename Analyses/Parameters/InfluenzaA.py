## Influenza parameters
## Literature paramdeters on flu, and guesses to match KPSC data

import numpy as np
from numba import jit
import pandas as pd
# from utils import age_detection

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

## Immunity
# Immunity relationships determined by vaccine parameters
# Free parameters between 0 and 1
IMM_ABOVE_MIN = 0 # The immunity to infection and disease on exposure above the minimum level informed by data
FIRST_IMM_MAG = 0 # The immunity to infection and disease after first infection above the minimum level informed by data
FIRST_DIS_INF_FACTOR = 0.39 # Determines relative infection and disease immunity after first infection
# ratio of protection against disease to protection against infection (Basta et al 2008)
R = 0.29/0.43 
# ratio of vaccine effectiveness in children to that in adults
CHILD_EFF_RATIO = 1.54 
# maximum lower CI of vaccine effectiveness in adults is 0.39, ESP should be at least this large
# cannot be larger than 1/CHILD_EFF_RATIO
ESP = 0.39 + IMM_ABOVE_MIN*(1/CHILD_EFF_RATIO-0.39)
factor = ((1 - R) + np.sqrt(1 + R**2 + 2*R*(1-2*ESP)))/2
# Relative susceptability and infectiousness, for each susceptibility class
# S1 has to be >= (1-1.54*ESP)/(1-ESP) for D1 to be >= 1
S1 = ((1-CHILD_EFF_RATIO*ESP)/(1-ESP) + FIRST_IMM_MAG*(1-(1-CHILD_EFF_RATIO*ESP)/(1-ESP)))**FIRST_DIS_INF_FACTOR
S2 = S1*(1-ESP)/factor
# Probability of detection of cases for each susceptibility class
# plug this into the expressions for S2 and D2 and you get D1 = (1-CHILD_EFF_RATIO*ESP)/((1-ESP)*S1)
# D1 should be at least as big as that value
D1 = ((1-CHILD_EFF_RATIO*ESP)/(1-ESP) + FIRST_IMM_MAG*(1-(1-CHILD_EFF_RATIO*ESP)/(1-ESP)))**(1-FIRST_DIS_INF_FACTOR)
D2 = D1*factor

S_REL = np.array([1,S1,S2])
I_REL = np.array([[1],[1],[1]])
P_OBS_REL = np.array([1,D1,D2])

P_OBS_MAX = 0.05
P_OBS = P_OBS_MAX*P_OBS_REL

## Parameters that vary by age group
# Age-specific susceptibility
S_AGE = np.ones(NAG)
# Age-specific relative probability of detection
OBS_AGE = np.array([0.2,0.15,0.1,0.05,0.05,0.2,1])
## Other
# Seasonality parameters
SEASONALITY = 0.04
OFFSET = 0.25
# Infectiousness
BETA = 0.11
# Vaccination paramters
S_VAX = 2
@jit
def BCOV(t):
    return 0

IMPORT_RATE = 1e-11
PP = pd.read_csv("Data/Processed/FluView_PercentPositive_Regions_A.csv")
PP.index = pd.to_datetime(PP["Date"], format="%Y-%m-%d")
PP.index = (PP.index - pd.to_datetime("1970-01-01")).days
state_centroids = pd.read_csv("Data/Raw/state_centroids.csv")# # gravity model based on distance of states to california
from scipy.spatial import distance
dist = distance.cdist(state_centroids[['latitude','longitude']],state_centroids[['latitude','longitude']])
# gravity model
gravity = 2**(-dist/5)
# normalize
gravity = gravity/gravity.sum(axis=1)[:,np.newaxis]
# set diagonal to zero
np.fill_diagonal(gravity,0)
PP_NP_ALL = np.dot(gravity,PP[state_centroids['state']].values.T)
# get just the value for California
PP_NP = PP_NP_ALL[4]
PP_IDX = np.array(PP.index)
@jit
def regional_positivity(t):
    return PP_NP[np.argmax(PP_IDX>=t)]