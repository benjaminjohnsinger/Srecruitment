import numpy as np
import pandas as pd
from numba import jit

@jit
def STATIC(t):
    return 1

@jit
def STEP(t,t_lockdown,duration,reduction):
    return 1-reduction if t < t_lockdown + duration and t > t_lockdown else 1

@jit
def RAMP(t,t_lockdown,duration,recovery_duration,reduction):
    return 1-reduction if t < t_lockdown + duration and t > t_lockdown else 1 + reduction*((t-t_lockdown-duration)/recovery_duration - 1) if t > t_lockdown and t < t_lockdown + duration + recovery_duration else 1

@jit
def piecewise(t,ts,fs):
    return fs[np.where(ts <= t)[0][-1]]

MOBILITY2020 = pd.read_csv('Data/Raw/Google_mobility_reports/2020_US_Region_Mobility_Report.csv', delimiter=',')
MOBILITY2021 = pd.read_csv('Data/Raw/Google_mobility_reports/2021_US_Region_Mobility_Report.csv', delimiter=',')
MOBILITY2022 = pd.read_csv('Data/Raw/Google_mobility_reports/2022_US_Region_Mobility_Report.csv', delimiter=',')
MOBILITY = pd.concat([MOBILITY2020, MOBILITY2021, MOBILITY2022])
MOBILITY_CA = MOBILITY.loc[MOBILITY['iso_3166_2_code'] == 'US-CA']
MOBILITY_CA = MOBILITY_CA.sort_values(by='date')
MOBILITY_CA.index = (pd.to_datetime(MOBILITY_CA['date'])-pd.to_datetime('1970-01-01')).dt.days
MOBILITY_WORK = 1+MOBILITY_CA['workplaces_percent_change_from_baseline']/100
MOBILITY_WORK_MA = MOBILITY_WORK.rolling(window=28).mean()
MOBILITY_WORK_MA = MOBILITY_WORK_MA.fillna(method='bfill')
RELATIVE_CONTACT_WORK = 1.3169 - 4.7718*MOBILITY_WORK_MA + 5.7062*MOBILITY_WORK_MA**2
IDX = np.array(MOBILITY_CA.index)
RELATIVE_CONTACT_WORK_NP = np.array(RELATIVE_CONTACT_WORK)
# print(RELATIVE_CONTACT_WORK_NP)
@jit
def google_prestige_work(t):
    # if t<18355: # if before first dip below baseline
    #     return 1
    # elif t<18952: # if before first recovery to 95% of baseline
    # if t < max(IDX):
    return np.minimum(1,RELATIVE_CONTACT_WORK_NP[np.argmin(IDX<=t)])
    # else:
    #     return RELATIVE_CONTACT_WORK_NP[-1]
    # else:
    #     return 1
# print([RELATIVE_CONTACT_WORK_NP[np.argmin(IDX<=t)] for t in [18383,18506,18809,19024,19631]])