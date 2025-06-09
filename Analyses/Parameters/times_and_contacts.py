import pandas as pd
import jax.numpy as jnp
import numpy as np
from numba import jit
import contact_model as cm
from utils import date_to_t

## Period of simulation
EPOCH = pd.to_datetime('1970-01-01')
START = pd.to_datetime('2015-07-04') # 89 days before start of KPSC data, to account for delay in detection
END = pd.to_datetime('2023-10-01') # end of KPSC data
PERIOD = pd.date_range(start=START, end=END, freq='D')

## Contacts and force of infection
# Contact matrix for all contact types
CONTACT = jnp.asarray(pd.read_csv('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', header=None).values)

# Lockdown and other mobility changes
Ts = jnp.array([date_to_t(EPOCH),
date_to_t('2020-03-19'), # Newsom announces stay-at-home order
date_to_t('2021-04-27'), # CDC amends mask guidance to allow vaccinated individuals to go maskless
date_to_t('2021-12-15'), # CDC reinstates mask guidance
date_to_t('2022-03-01')]) # End of mask mandate in California
# date_to_t('2021-12-20')])
Fs = jnp.array([1,0.2,1,0.2,1]) # 0.2 minimum relative contact rate between COMIX and POLYMOD
# @jit
def contact(t,seasonality,offset,CONTACT=CONTACT,Ts=Ts,Fs=Fs):
    return cm.piecewise(t,Ts,Fs)*(1+seasonality*jnp.cos(2*jnp.pi*((t-274)/365-offset)))*CONTACT

T_VAX = date_to_t('2035-01-01')

POINTS = jnp.array(date_to_t(PERIOD))