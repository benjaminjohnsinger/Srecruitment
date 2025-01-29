import numpy as np
import pandas as pd
import scipy as sp
from numba import jit

from matplotlib import pyplot as plt
from matplotlib import cm as colormaps

# Vaccination of infants (a proportion)
# S_VAX is the susceptibility class of vaccinated individuals
# COVERAGE is the proportion of infants vaccinated
# T_VAX is the time at which vaccination starts

def birth_vax(t,s_class,S_VAX=2,COVERAGE=0,T_VAX=0):
    if t < T_VAX:
        return 1 if (s_class == 0) else 0
    else:
        if s_class == S_VAX:
            return COVERAGE
        elif s_class == 0:
            return 1-COVERAGE
        else:
            return 0

# Annual mass vaccination (a rate)
# S_VAX is the susceptibility class of vaccinated individuals
# coverage is the time-varying proportion of the population vaccinated each month - this can be an age-dependent vector

def all_vax(t,s_class,rate,S_VAX=2,NAG=7,N_S=3,N_C=3,cov_args=[np.ones(2),np.ones(7),np.ones(7)]):
    if s_class != S_VAX:
        # (3-N_C) here is a really hacky way of making this work with SIS model, which needs to reference an extra empty compartment
        vec = np.zeros((N_C*N_S+1+(3-N_C))*NAG)
        vec[(N_C*s_class+1)*NAG:(N_C*s_class+2)*NAG] = -1
    else:
        vec = np.zeros((N_C*N_S+1+(3-N_C))*NAG)
        for i in range(N_S):
            if i != s_class:
                vec[(i*N_C+1)*NAG:(i*N_C+2)*NAG] = 1
    rate_val = rate(t,cov_args[0],cov_args[1],cov_args[2])
    if isinstance(rate_val,(int,float)):
        return rate_val*vec
    else:
        return np.tile(rate_val,N_C*N_S+1+(3-N_C))*vec

# Flu vaccination coverage
VAX_FLU = pd.read_csv('Data/Processed/KPSC_vaccinated_proportion_ages_monthly.csv',index_col=0)
VAX_FLU = VAX_FLU[['<3m','3-11m','1-4y','5-17y','18-39y','40-64y','>=65y']]
VAX_FLU.index = pd.to_datetime(VAX_FLU.index,format='%Y-%m')
VAX_FLU.index = (VAX_FLU.index - pd.to_datetime('1970-01-01')).days
KPSC_RATIO = 0.455/0.5793469238813651 # ratio of 2022 CDC vaccine coverage data for California to age-adjusted coverage in 2022/23 season KPSC data
VAX_FLU_ADJUSTED = VAX_FLU*KPSC_RATIO
VAX_FLU_IDX = np.array(VAX_FLU_ADJUSTED.index)
VAX_FLU_NP = np.array(VAX_FLU_ADJUSTED)

# Flu efficacy by year
# EFF_ALL = (1/100)*np.array([42, 56, 60, 47, 49, 52, 19, 48, 40, 38, 29, 39, 42, 36, 30, 42, 42]) # from CDC, Data/Raw/vaccine-effectiveness-chart-2024.xlsx, with missing data filled in with mean (42)
# EFF_ALL_CI_MIN = (1/100)*np.array([0,22.7,54,36,29,44,10,41,32,31,21,32,0,21,-9,29,0])
# EFF_ALL_CI_MAX = (1/100)*np.array([100,74.8,66,56,47,59,27,55,46,43,35,44,100,48,54,53,100])
# EFF_CHILD = (1/100) * np.array([50, 41, 58, 59, 51, 48, 25, 51, 57, 68, 48, 34, 50, 51, 50, 52, 50]) # VE in youngest age group from studies that went into Data/Raw/vaccine-effectiveness-chart-2024.xlsx, with missing data filled in with mean (50)
EFF = (1/100)*np.array([37, 61, 51, 44, 39, 53, 7, 52, 19, 33, 25, 34, 37, 32, 23, 41, 37]) # VE in 18-49yo (or age group containing this range) from studies that went into Data/Raw/vaccine-effectiveness-chart-2024.xlsx, with missing data filled in with mean (37)
# EFF_CI_MIN = (1/100)*np.array([0,-1,36,21,16,39,-12,39,0,21,10,23,0,3,-29,34,0])
# EFF_CI_MAX = (1/100)*np.array([100,84.8,62,60,48,64,33,61,34,44,37,44,100,52,54,47,100])
EFF_IDX = np.array([(pd.to_datetime('2008-10-01') + pd.DateOffset(years=i) - pd.to_datetime('1970-01-01')).days for i in range(17)])

@jit
def flu_eff_coverage(t,protection,eff_cap=True):
    '''
    Calculate the effective coverage (proportion of people protected) of the flu vaccine at time t
    t: time in days since 1970-01-01
    protection: relative risk of disease given exposure for each susceptibility class
    eff_cap: whether to cap the efficacy at 1
    '''
    max_eff = (protection[-2]-protection[-1])/protection[-2] # this is assuming that most people are in the last two susceptibility classes
    raw_eff = EFF[np.argmin(EFF_IDX<=t)]
    adj_eff = raw_eff/max_eff
    if adj_eff > 1 and eff_cap:
        adj_eff = 1
    # If time steps are less than one year, this code can be used to repeat seasonal patterns outside of data scope
    # if time is before 2015-10-01, corresponding month in 2015-10-01 to 2016-09-30 is used
    if t < 16709: # 16709 is the number of days since 1970-01-01 to 2015-10-01
        day_in_season = (t - 16709)%365
        time_2015 = 16709 + day_in_season
        return adj_eff*VAX_FLU_NP[np.argmax(VAX_FLU_IDX>=time_2015)]
    # if time is after 2022-10-01, corresponding month in 2022-10-01 to 2023-09-30 is used
    elif t > 19266:
        day_in_season = (t - 19266)%365
        time_2022 = 19266 + day_in_season
        return adj_eff*VAX_FLU_NP[np.argmax(VAX_FLU_IDX>=time_2022)]
    else:
        return adj_eff*VAX_FLU_NP[np.argmax(VAX_FLU_IDX>=t)]

# # Flu vaccination rate
@jit
def flu_rate(t,protection,pops,aging_rate,eff_cap=True):
    v = flu_eff_coverage(t,protection)
    v_next_month = flu_eff_coverage(t+31,protection)
    pop_shift = np.concatenate((np.zeros(1),pops[:-1]))
    v_shift = np.concatenate((np.zeros(1),v[:-1]))
    age_rate_shift = np.concatenate((np.zeros(1),aging_rate[:-1]))
    rate = (1/(1-v))*((v_next_month-v)/31 + (1/365)*v + age_rate_shift*(pop_shift/pops)*(v-v_shift))
    return np.maximum(0,rate)