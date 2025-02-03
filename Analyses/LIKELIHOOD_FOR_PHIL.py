## BJS Feb 2025
## Fitting models to data using surrogate model

# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import sys

# Importing the necessary functions to connect to data sources
from vaccination import birth_vax, all_vax, flu_rate, flu_eff_coverage
import contact_model as cm
from SISn_ODEs import single_pathogen_deltas as sis_deltas
from Parameters.census_population import *
from Parameters.times_and_contacts import *
from Parameters.InfluenzaA import *

from utils import *
from demography import *
from mobility_and_import import *
from plotting import *
from fit_MCMC import *

# Data on delayed reporting
p_time_to_obs = np.genfromtxt("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)

## Initial conditions
STATE0 = np.zeros((2*N_S+2)*NAG)
STATE0[NAG:2*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

# Default parameters for the ODE, as taken from Parameters/InfluenzaA.py
params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': flu_rate, 'BCOV': BCOV, 'T_VAX': T_VAX,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
'contact': contact}

# KPSC data on which to evaluate fit
incidence = pd.read_csv("Data/Processed/KPSC_Influenza_A_incidence_age_daily.csv",index_col=0)

# Parameters from command line
x = np.array(sys.argv[1:],dtype=float)

# Set model parameters accordingly
sim_params = params.copy()
sim_params["WANE"] = np.array([0.0,x[0],0.0])
sim_params["SEASONALITY"] = x[1]
sim_params["OFFSET"] = x[2]
sim_params["BETA"] = x[3]
sim_params["IMPORT_RATE"] = x[4]
srel, pobsrel = constrained_immunity(x[5],x[6],x[7])
sim_params["S_REL"] = srel
sim_params["P_OBS"] = x[8]*pobsrel
obs_age = age_detection(NAG,x[9],x[10],x[11])

# Print negative log likelihood
print(-SIS_likelihood(incidence,sim_params,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True))

## or:
# def likelihood(x):
#     sim_params = params.copy()
#     sim_params["WANE"] = np.array([0.0,x[0],0.0])
#     sim_params["SEASONALITY"] = x[1]
#     sim_params["OFFSET"] = x[2]
#     sim_params["BETA"] = x[3]
#     sim_params["IMPORT_RATE"] = x[4]
#     srel, pobsrel = constrained_immunity(x[5],x[6],x[7])
#     sim_params["S_REL"] = srel
#     sim_params["P_OBS"] = x[8]*pobsrel
#     obs_age = age_detection(NAG,x[9],x[10],x[11])
#     return -SIS_likelihood(incidence,sim_params,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)