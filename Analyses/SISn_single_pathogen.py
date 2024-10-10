## SIS model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import time
import itertools as it
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle

from vaccination import birth_vax, all_vax
import contact_model as cm
from SISn_ODEs import single_pathogen_deltas as deltass
from Parameters.test_population import *
from Parameters.generic_disease import *

from plotting import *

## Period of simulation in months
PERIOD = 12*50

## Contacts and force of infection
IMPORT = 0.01*np.ones(N_S)
# Contact matrix for all contact types
CONTACT = np.genfromtxt('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',')
CONTACT /= 12.99 # transform to contact proportions
# Lockdown and other mobility changes
T_LOCKDOWN = 37*12
LOCKDOWN_DURATION = 12
LOCKDOWN_REDUCTION = 0.4
shape_step = lambda t : cm.STEP(t,T_LOCKDOWN,LOCKDOWN_DURATION,LOCKDOWN_REDUCTION)
shape_static = lambda t : 1
def contact(t,shape,seasonality=SEASONALITY,offset=OFFSET,c_rate=CONTACT):
    return shape(t)*(1+seasonality*np.cos(2*np.pi*(t/12-offset)))*c_rate

## Initial conditions
STATE0 = np.zeros(2*N_S*NAG)
STATE0[0:NAG] = KP_AGE_POP-1 # Everyone is susceptible except
STATE0[NAG:2*NAG] = 1 # one individual in each age group that is infected.

## Integrate the system
# POINTS = np.concat((np.zeros(1),np.arange(T_LOCKDOWN-5*12,T_LOCKDOWN+LOCKDOWN_DURATION+5*12,0.1),np.ones(1)*PERIOD))
POINTS = np.arange(0,PERIOD+1,1)
T_VAX = PERIOD

# BETA *= 0.8

# Parameters for the ODE
params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': BIRTH_RATE, 'WANE': WANE, 'REC': REC, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV, 'T_VAX': T_VAX,
'IMPORT': IMPORT, 'BETA': BETA, 'contact':lambda t : contact(t,shape_step)}
result = sp.integrate.solve_ivp(deltass,(0,PERIOD),STATE0,args=(params,),t_eval=POINTS,method='RK45')

plt.plot(result.y.T)
plt.show()