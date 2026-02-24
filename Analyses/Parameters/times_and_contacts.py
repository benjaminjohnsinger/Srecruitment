import pandas as pd
# # import jax.numpy as jnp
import numpy as np

import contact_model as cm

def date_to_t(date, start_date=pd.to_datetime("1970-01-01")):
    """
    Convert date to time index
    """
    date_time = pd.to_datetime(date)
    return (date_time - start_date).days

## Period of simulation
EPOCH = pd.to_datetime('1970-01-01')
START = pd.to_datetime('2015-07-04') # 89 days before start of KPSC data, to account for delay in detection
END = pd.to_datetime('2025-05-01') # end of KPSC data
PERIOD = pd.date_range(start=START, end=END, freq='D')

## Contacts and force of infection
# Contact matrix for all contact types
CONTACT = np.asarray(pd.read_csv('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', header=None).values)

# Lockdown and other mobility changes
TT = np.array([date_to_t(EPOCH),
date_to_t('2020-03-19'), # Newsom announces stay-at-home order
date_to_t('2021-04-27'), # CDC amends mask guidance to allow vaccinated individuals to go maskless
date_to_t('2021-12-15'), # CDC reinstates mask guidance
date_to_t('2022-03-01')]) # End of mask mandate in California
# date_to_t('2021-12-20')])
FF = np.array([1,0.2,1,0.2,1]) # 0.2 minimum relative contact rate between COMIX and POLYMOD
# # @jit
# Ts = np.array([date_to_t('1970-01-01'), date_to_t('2020-03-19'), date_to_t('2021-03-07'), date_to_t('2021-08-29'), date_to_t('2022-04-30')])
# Fs = [1, 0.74032097, 0.94517362, 0.78748583, 0.96393216]
def contact(t,seasonality,offset,CONTACT=CONTACT,Ts=TT,Fs=FF):
    return cm.piecewise(t,Ts,Fs)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT

# POINTS = np.array(date_to_t(PERIOD), dtype=jnp.float32)