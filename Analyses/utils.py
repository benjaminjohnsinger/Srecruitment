# import jax.numpy as np
from random import seed

import numpy as np
import pandas as pd
import re
import jax
import jax.numpy as jnp
import contact_model as cm
import pickle
import sys
import time
from functools import partial

class FakeOpt:
    def __init__(self, x):
        self.x = x


####### Utility functions #######


def date_to_t(date, start_date=pd.to_datetime("1970-01-01")):
    """
    Convert date to time index
    """
    date_time = pd.to_datetime(date)
    return (date_time - start_date).days

def t_to_date(t, start_date=pd.to_datetime("1970-01-01")):
    """
    Convert time index to date
    """
    return start_date + pd.DateOffset(days=t)

def real_to_p(number):
    """
    Map real number to (0,1) interval
    """
    return 1/(1+np.exp(-number))

def p_to_real(probability):
    """
    Map (0,1) interval to real number
    """
    return np.log(probability/(1-probability))

def increment_to_vec(increment,length):
    """
    Convert increment to vector
    """
    return np.array([max(0,1-i*increment) for i in range(length)])

def to_increment(vec):
    """
    Convert vector to increment, or just return value
    """
    if isinstance(vec,np.ndarray):
        return 1-vec[1]
    else:
        return vec

def calculate_population_size(values, N_S=3, NAG=7):
    """
    Calculate population size from model output
    """
    population_size = jnp.sum(values[1:-NAG].reshape(2*N_S, NAG, -1), axis=0).T
    population_size = population_size.at[:,0].add(values[0,:])
    return population_size

def age_detection(NAG,young_immunity,old_immunity,young_old,maternal_immunity=None,linear=True,min_obs=0.05,n_infant_groups=2):
    """
    Calculate age-specific relative probabilty of detection from parameters.
    young_immunity: immunity gained by aging an age group
    old_immunity: immunity lost by aging an age group
    young_old: half the ratio of young to old susceptibility
    maternal_immunity: extra protection given to infants (optional)
    """
    if maternal_immunity is None:
        x = np.arange(NAG)
        if not linear:
            OBS = np.minimum(1,2*young_old)*young_immunity**x+np.minimum(1,2*(1-young_old))*old_immunity**(NAG-1-x)
        elif linear:
            OBS = np.max((min_obs*np.ones(NAG),np.minimum(1,2*young_old)-young_immunity*x,np.minimum(1,2*(1-young_old))-old_immunity*(NAG-1-x)),axis=0)
    else:
        x = np.arange(NAG-n_infant_groups)
        OBS_noninfant = np.max((min_obs*np.ones(NAG-n_infant_groups),np.minimum(1,2*young_old)-young_immunity*x,np.minimum(1,2*(1-young_old))-old_immunity*(NAG-1-n_infant_groups-x)),axis=0)
        OBS_infant = np.max((min_obs*np.ones(n_infant_groups),np.min((np.ones(n_infant_groups),np.minimum(1,2*young_old)-2*(maternal_immunity-0.5)*np.arange(n_infant_groups,0,-1)),axis=0)),axis=0)
        OBS = np.concatenate((OBS_infant,OBS_noninfant))
    return OBS/np.max(OBS)

    # else:
    #     if maternal_immunity <= young_immunity:
    #         OBS_infant = np.max((min_obs*np.ones(n_infant_groups),np.minimum(1,2*young_old)-(young_immunity-maternal_immunity)*np.arange(n_infant_groups),axis=0))
    #         young_start = np.minimum(1,2*young_old)-n_infant_groups*(young_immunity-maternal_immunity)
    #     else:
    #         m_i = (maternal_immunity-young_immunity)/(1-young_immunity)
    #         OBS_infant = np.max((min_obs*np.ones(n_infant_groups),np.minimum(1,2*young_old)-mi*np.arange(n_infant_groups,0,-1),axis=0))
    #         young_start = np.minimum(1,2*young_old)
    #     x = np.arange(NAG-n_infant_groups)
    #     OBS_noninfant = np.max((min_obs*np.ones(NAG-n_infant_groups),young_start-young_immunity*x,np.minimum(1,2*(1-young_old))-old_immunity*(NAG-1-n_infant_groups-x)),axis=0)
        

# print(age_detection(7, 1,0.59,0.67,0.615,linear=True,min_obs=0.025,n_infant_groups=1))
# print(age_detection(7, 1,0.9,0.05,0.475,linear=True,min_obs=0.025,n_infant_groups=1))

# DIS_INF_RATIO = 0.29/0.43 is the ratio of protection against disease to protection against infection from Basta et al 2008
# MIN_EFF = 0.39 is the maximum lower CI of vaccine effectiveness in adults, so ESP (protection against infection and illness in adults, terminology from Basta et al 2008) should be at least this large - from CDC
# CHILD_EFF_RATIO = 1.54 is the ratio of vaccine effectiveness in children to that in adults, so ESP should be at most 1/CHILD_EFF_RATIO
def constrained_immunity(extra_immunity,first_immunity,first_dis_inf_factor,DIS_INF_RATIO=0.674,MIN_EFF=0.39,CHILD_EFF_RATIO=1.54):
    """
    Convert free parameters between 0 and 1 to immunity parameters constrained by flu vaccine data
    """
    total_eff = MIN_EFF + extra_immunity*(1/CHILD_EFF_RATIO - MIN_EFF)
    child_eff = (1-CHILD_EFF_RATIO*total_eff)/(1-total_eff)
    base_immunity = child_eff + first_immunity*(1-child_eff)
    S1 = base_immunity**first_dis_inf_factor
    D1 = base_immunity**(1-first_dis_inf_factor)

    factor = ((1 - DIS_INF_RATIO) + jnp.sqrt(1 + DIS_INF_RATIO**2 + 2*DIS_INF_RATIO*(1-2*total_eff)))/2
    S2 = S1*(1-total_eff)/factor
    D2 = D1*factor
    return jnp.array([1,S1,S2]), jnp.array([1,D1,D2])

def scalars_to_params(scalar_values_dict, params, NAG=7, N_S=3, N_C=2):
    for name in scalar_values_dict.keys():
        if name in ["NAG","N_S","BIRTH_RATE","S_VAX","ACOV","BCOV","T_VAX","IMPORT_RATE","BETA","SEASONALITY","OFFSET"]:
            params[name] = scalar_values_dict[name]
        elif name == "WANE2":
            params[name] = scalar_values_dict[name]*np.array([0]+[1]*(N_S-2)+[0])
        elif name == "REC_UP":
            params[name] = scalar_values_dict[name]*np.array([1]*(N_S-1)+[0])
        elif name == "REC_SAME":
            params[name] = scalar_values_dict[name]*np.array([0]*(N_S-1)+[1])
        elif name == "REC":
            params["REC_UP"] = scalar_values_dict[name]*np.array([1]*(N_S-1)+[0])
            params["REC_SAME"] = scalar_values_dict[name]*np.array([0]*(N_S-1)+[1])
        elif name == "P_OBS":
            params[name] = scalar_values_dict[name]*params[name]/np.max(params[name])
        elif name == "P_OBS_REL":
            params["P_OBS"] = np.max(params[name])*increment_to_vec(scalar_values_dict[name],N_S)
        elif name in ["S_REL","I_REL","S_AGE"]:
            params[name] = increment_to_vec(scalar_values_dict[name],N_S)
        elif name == "ACQUIRED_IMMUNITY":
            params["S_REL"] = increment_to_vec(scalar_values_dict[name],N_S)
        elif re.search(r"S_REL\d+",name):
            i = int(name[5:])
            params["S_REL"][i] = np.prod(np.array([scalar_values_dict["S_REL"+str(j)] for j in range(1,i+1)]))
        elif re.search(r"D_REL\d+",name):
            i = int(name[5:])
            pobsrel = np.array([1]+[np.prod(np.array([scalar_values_dict["D_REL"+str(j)] for j in range(1,i+1)])) for i in range(1,N_S)])
            params["P_OBS"] = pobsrel*scalar_values_dict["P_OBS"]
        elif re.search(r"ACQUIRED_IMMUNITY\d+",name):
            params["S_REL"][int(name[17:])] = scalar_values_dict[name]
        elif re.search(r"P_OBS\d+",name):
            params["P_OBS"][int(name[5:])] = scalar_values_dict[name]
        elif name in ["EXTRA_IMMUNITY","FIRST_IMMUNITY","FIRST_DIS_INF_FACTOR"]:
            params["S_REL"], obs_rel = constrained_immunity(scalar_values_dict["EXTRA_IMMUNITY"],scalar_values_dict["FIRST_IMMUNITY"],scalar_values_dict["FIRST_DIS_INF_FACTOR"])
            params["P_OBS"] = obs_rel*scalar_values_dict["P_OBS"]
        elif name in ["DT1","DT2","DT3","F1","F2","F3","F4"]:
            Ts = np.array([date_to_t('1970-01-01'),date_to_t('2020-03-19'),date_to_t('2020-03-19')+scalar_values_dict["DT1"]*365,date_to_t('2020-03-19')+(scalar_values_dict["DT1"]+scalar_values_dict["DT2"])*365,date_to_t('2020-03-19')+(scalar_values_dict["DT1"]+scalar_values_dict["DT2"]+scalar_values_dict["DT3"])*365])
            F1 = scalar_values_dict["F1"] # value between 0 and 1 (first lockdown)
            F2 = F1 + scalar_values_dict["F2"] - F1*scalar_values_dict["F2"] # value between x_temp[0+3] and 1 (inter-lockdown)
            F3 = F2*scalar_values_dict["F3"] # value less than F2 (second lockdown)
            F4 = F2 + scalar_values_dict["F4"] - F2*scalar_values_dict["F4"] # value between F2 and 1 (post-lockdown)
            Fs = np.array([1,F1,F2,F3,F4])
            CONTACT = np.asarray(pd.read_csv('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', dtype=np.float32, header=None).values)
            # @jit
            def contact(t,seasonality,offset):
                return cm.piecewise(t,Ts,Fs)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
            params["contact"] = contact
    #     elif re.search(r"OBS_AGE\d+",name):
    #         params["OBS_AGE"][int(name[8:])] = scalar_values_dict[name]
    # if "OBS_AGE_YOUNG" in scalar_values_dict.keys() or "OBS_AGE_OLD" in scalar_values_dict.keys() or "OBS_AGE_YOUNG_OLD" in scalar_values_dict.keys():
    #     if not all([key in scalar_values_dict.keys() for key in ["OBS_AGE_YOUNG","OBS_AGE_OLD","OBS_AGE_YOUNG_OLD"]]):
    #         raise ValueError("If you want to set OBS_AGE with parameters, you need to set all of OBS_AGE_YOUNG, OBS_AGE_OLD, and OBS_AGE_YOUNG_OLD")
    #     params["OBS_AGE"] = age_immunity(np.arange(NAG),scalar_values_dict["OBS_AGE_YOUNG"],scalar_values_dict["OBS_AGE_OLD"],scalar_values_dict["OBS_AGE_YOUNG_OLD"])
    return params

def params_to_scalars(param_dict,scalar_names):
    scalar_dict = {}
    for name in scalar_names:
        if name in ["NAG","N_S","BIRTH_RATE","S_VAX","ACOV","BCOV","T_VAX","IMPORT_RATE","BETA","SEASONALITY","OFFSET"]:
            scalar_dict[name] = param_dict[name]
        elif name == "WANE2":
            scalar_dict[name] = param_dict[name][1]
        elif name in ["WANE2","REC_UP","REC_SAME","P_OBS"]:
            scalar_dict[name] = param_dict[name][0]
        elif name in ["S_REL","I_REL","S_AGE","OBS_AGE"]:
            scalar_dict[name] = 1-param_dict[name][1]
        elif name == "P_OBS_REL":
            scalar_dict[name] = param_dict["P_OBS"][1]/np.max(param_dict["P_OBS"])
        elif name == "ACQUIRED_IMMUNITY":
            scalar_dict[name] = 1-param_dict["S_REL"][1]
        elif re.search(r"S_REL\d+",name):
            scalar_dict[name] = param_dict["S_REL"][int(name[5:])]
        elif re.search(r"ACQUIRED_IMMUNITY\d+",name):
            scalar_dict[name] = param_dict["S_REL"][int(name[17:])]
        elif re.search(r"P_OBS\d+",name):
            scalar_dict[name] = param_dict["P_OBS"][int(name[5:])]
    return scalar_dict

def pathogen_parameters(pathogen, import_multiplier=1e-9, incidence_data=False, smoothed=False, skip_incidence=False, hosp=True, NAG=7, dedup=False):
    ARRIVALS = jnp.asarray(np.genfromtxt('Data/Processed/arrivals_daily.csv', delimiter=','))
    if 'RSV' in pathogen:
        REC_UP = jnp.array([1/4.9,1/4.1,0.0])
        REC_SAME = jnp.array([0.0,0.0,1/4.1])
        IMPORT_STRENGTH = import_multiplier*ARRIVALS*jnp.asarray(np.genfromtxt('Data/Processed/RSV_positivity_daily.csv', delimiter=','))
        p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
        if incidence_data:
            if incidence_data=="Old":
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_unsalvage_panel_positive_RSV_matched_noncovid_ARI_hospitalizations_proportional_incidence.csv",index_col=0).values)
            elif incidence_data=="orig":
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_ARI_RSV_incidence_age_daily.csv",index_col=0))
            elif smoothed:
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_panel_proportion_positive_ARI_nonCOVID_RSV_incidence_age"+['', '_hosp'][hosp]+"_daily_smootheds14fp1.csv",index_col=0).values)
            else:
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_panel_proportion_positive_ARI_nonCOVID_RSV_incidence_age"+['', '_hosp'][hosp]+"_daily.csv",index_col=0).values)
        else:
            positives = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_RSV_positive_counts'+['', '_hospday'][hosp]+['', '_split'][NAG>7]+['', '_dedup'][dedup]+'.csv', header=None).values)
            total_tests = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_RSV_total_counts'+['', '_hospday'][hosp]+['', '_split'][NAG>7]+['', '_dedup'][dedup]+'.csv', header=None).values)
    elif 'InfluenzaA' in pathogen:
        REC_UP = jnp.array([1/3.0,1/3.0,0.0])
        REC_SAME = jnp.array([0.0,0.0,1/3.0])
        IMPORT_STRENGTH = import_multiplier*ARRIVALS*jnp.asarray(np.genfromtxt('Data/Processed/InfluenzaA_positivity_daily.csv', delimiter=','))
        p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
        if incidence_data:
            if incidence_data=="Old":
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_unsalvage_panel_positive_InfluenzaA_matched_noncovid_ARI_hospitalizations_proportional_incidence.csv",index_col=0).values)
            elif incidence_data=="orig":
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_ARI_InfluenzaA_incidence_age_daily.csv",index_col=0))
            elif smoothed:
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_panel_proportion_positive_ARI_nonCOVID_InfluenzaA_incidence_age"+['', '_hosp'][hosp]+"_daily_smootheds14fp1.csv",index_col=0).values)
            else:
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_panel_proportion_positive_ARI_nonCOVID_InfluenzaA_incidence_age"+['', '_hosp'][hosp]+"_daily.csv",index_col=0).values)
        else:
            positives = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_InfluenzaA_positive_counts'+['', '_hospday'][hosp]+['', '_split'][NAG>7]+['', '_dedup'][dedup]+'.csv', header=None).values)
            total_tests = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_InfluenzaA_total_counts'+['', '_hospday'][hosp]+['', '_split'][NAG>7]+['', '_dedup'][dedup]+'.csv', header=None).values)
    elif 'InfluenzaB' in pathogen:
        REC_UP = jnp.array([1/3.0,1/3.0,0.0])
        REC_SAME = jnp.array([0.0,0.0,1/3.0])
        IMPORT_STRENGTH = import_multiplier*ARRIVALS*jnp.asarray(np.genfromtxt('Data/Processed/InfluenzaB_positivity_daily.csv', delimiter=','))
        p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/Influenza_B_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
        if incidence_data:
            if incidence_data=="Old":
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_unsalvage_panel_positive_InfluenzaB_matched_noncovid_ARI_hospitalizations_proportional_incidence.csv",index_col=0).values)
            elif incidence_data=="orig":
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_ARI_InfluenzaB_incidence_age_daily.csv",index_col=0))
            elif smoothed:
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_panel_proportion_positive_ARI_nonCOVID_InfluenzaB_incidence_age"+['', '_hosp'][hosp]+"_daily_smootheds14fp1.csv",index_col=0).values)
            else:
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_panel_proportion_positive_ARI_nonCOVID_InfluenzaB_incidence_age"+['', '_hosp'][hosp]+"_daily.csv",index_col=0).values)
        else:
            positives = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_InfluenzaB_positive_counts'+['', '_hospday'][hosp]+['', '_split'][NAG>7]+['', '_dedup'][dedup]+'.csv', header=None).values)
            total_tests = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_InfluenzaB_total_counts'+['', '_hospday'][hosp]+['', '_split'][NAG>7]+['', '_dedup'][dedup]+'.csv', header=None).values)
    elif 'Parainfluenza3' in pathogen:
        REC_UP = jnp.array([1/3.0,1/3.0,0.0])
        REC_SAME = jnp.array([0.0,0.0,1/3.0])
        IMPORT_STRENGTH = import_multiplier*ARRIVALS*jnp.asarray(np.genfromtxt('Data/Processed/Parainfluenza3_positivity_daily.csv', delimiter=','))
        p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
        if incidence_data:
            if incidence_data=="Old":
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_unsalvage_panel_positive_Parainfluenza3_matched_noncovid_ARI_hospitalizations_proportional_incidence.csv",index_col=0).values)
            elif incidence_data=="orig":
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_ARI_Parainfluenza3_incidence_age_daily.csv",index_col=0))
            elif smoothed:
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_panel_proportion_positive_ARI_nonCOVID_Parainfluenza3_incidence_age"+['', '_hosp'][hosp]+"_daily_smootheds14fp1.csv",index_col=0).values)
            else:
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_panel_proportion_positive_ARI_nonCOVID_Parainfluenza3_incidence_age"+['', '_hosp'][hosp]+"_daily.csv",index_col=0).values)
        else:
            positives = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_Parainfluenza3_positive_counts'+['', '_hospday'][hosp]+['', '_split'][NAG>7]+['', '_dedup'][dedup]+'.csv', header=None).values)
            total_tests = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_Parainfluenza3_total_counts'+['', '_hospday'][hosp]+['', '_split'][NAG>7]+['', '_dedup'][dedup]+'.csv', header=None).values)
    elif 'Adenovirus' in pathogen:
        REC_UP = jnp.array([1/3.0,1/3.0,0.0])
        REC_SAME = jnp.array([0.0,0.0,1/3.0])
        IMPORT_STRENGTH = import_multiplier*ARRIVALS*jnp.asarray(np.genfromtxt('Data/Processed/Adenovirus_positivity_daily.csv', delimiter=','))
        p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
        if incidence_data:
            if incidence_data=="Old":
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_unsalvage_panel_positive_Adenovirus_matched_noncovid_ARI_hospitalizations_proportional_incidence.csv",index_col=0).values)
            elif incidence_data=="orig":
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_ARI_Adenovirus_incidence_age_daily.csv",index_col=0))
            elif smoothed:
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_panel_proportion_positive_ARI_nonCOVID_Adenovirus_incidence_age"+['', '_hosp'][hosp]+"_daily_smootheds14fp1.csv",index_col=0).values)
            else:
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_panel_proportion_positive_ARI_nonCOVID_Adenovirus_incidence_age"+['', '_hosp'][hosp]+"_daily.csv",index_col=0).values)
        else:
            positives = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_Adenovirus_positive_counts'+['', '_hospday'][hosp]+['', '_split'][NAG>7]+['', '_dedup'][dedup]+'.csv', header=None).values)
            total_tests = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_Adenovirus_total_counts'+['', '_hospday'][hosp]+['', '_split'][NAG>7]+['', '_dedup'][dedup]+'.csv', header=None).values)
    elif 'Metapneumovirus' in pathogen:
        REC_UP = np.array([1/4.9,1/4.1,0.0]) # based on RSV - Okiro 2010
        REC_SAME = np.array([0.0,0.0,1/4.1]) # based on RSV - Okiro 2010
        IMPORT_STRENGTH = import_multiplier*ARRIVALS*jnp.asarray(np.genfromtxt('Data/Processed/Metapneumovirus_positivity_daily.csv', delimiter=','))
        p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
        if incidence_data:
            if incidence_data=="Old":
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_unsalvage_panel_positive_Metapneumovirus_matched_noncovid_ARI_hospitalizations_proportional_incidence.csv",index_col=0).values)
            elif incidence_data=="orig":
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_ARI_Metapneumovirus_incidence_age_daily.csv",index_col=0))
            elif smoothed:
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_panel_proportion_positive_ARI_nonCOVID_Metapneumovirus_incidence_age"+['', '_hosp'][hosp]+"_daily_smootheds14fp1.csv",index_col=0).values)
            else:
                data = jnp.asarray(pd.read_csv("Data/Processed/KPSC_panel_proportion_positive_ARI_nonCOVID_Metapneumovirus_incidence_age"+['', '_hosp'][hosp]+"_daily.csv",index_col=0).values)
        else:
            positives = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_Metapneumovirus_positive_counts'+['', '_hospday'][hosp]+['', '_split'][NAG>7]+['', '_dedup'][dedup]+'.csv', header=None).values)
            total_tests = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_Metapneumovirus_total_counts'+['', '_hospday'][hosp]+['', '_split'][NAG>7]+['', '_dedup'][dedup]+'.csv', header=None).values)
    elif "test" in pathogen:
        REC_UP = jnp.array([1/3.0,1/3.0,0.0])
        REC_SAME = jnp.array([0.0,0.0,1/3.0])
        IMPORT_STRENGTH = import_multiplier*ARRIVALS*jnp.asarray(np.genfromtxt('Data/Processed/Metapneumovirus_positivity_daily.csv', delimiter=','))
        p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
    if skip_incidence:
        return REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs
    # join positives and total_tests into a single array with shape (time, age_group, 2)
    if not incidence_data:
        data = jnp.stack((total_tests, positives), axis=-1)
    return REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, data


# @partial(jax.jit, static_argnames=['pathogen','lockdown','option1','option2'])

# jax-safe importations for x_to_params
from Parameters.times_and_contacts import EPOCH, TT as defaultTT
from Parameters.times_and_contacts import FF as defaultFF
from new_vax import rsv_eff_vax_rate
from new_vax import rsv_maternal_immunity
from new_vax import flu_eff_vax_rate
def x_to_params(x, pathogen, lockdown, option1, option2, fixed_params = None, import_multiplier=1e-9, end_date='2025-05-01', print_params=False, rescale=None, return_contact=False, NAG=7, wrong_aging=False, birth_rate_multiplier=1.0):
    true_NAG = 7
    if "months" in option1:
        NAG = true_NAG = 65
    elif "split" in option1:
        NAG = true_NAG = 8
    if fixed_params is None:
        if "months" in option1:
            from Parameters.census_population import AGING_RATE_months as AGING_RATE
        elif NAG>7:
            from Parameters.census_population import AGING_RATE_split as AGING_RATE
        else:
            from Parameters.census_population import AGING_RATE
        if wrong_aging:
            AGING_RATE = AGING_RATE.at[4].set(1/(22*365))
        if "months" in option1:
            CONTACT_MATRIX = jnp.asarray(pd.read_csv('Data/Processed/contact_matrices/MONTHS_contact_all_US_Census.csv', delimiter=',', header=None).values)
        else:
            CONTACT_MATRIX = jnp.asarray(pd.read_csv('Data/Processed/contact_matrices/KP'+['', '_mod']["cmod" in option2]+['', '_split'][NAG>7]+'_contact_all_US_Census.csv', delimiter=',', header=None).values)
        BIRTH_RATE = birth_rate_multiplier * jnp.asarray(np.genfromtxt('Data/Processed/birth_rate_daily.csv', delimiter=','))
        if pathogen == "sim":
            _, _, IMPORT_STRENGTH, _ = pathogen_parameters("test", import_multiplier=import_multiplier, skip_incidence=True)
        else:
            REC_UP, REC_SAME, IMPORT_STRENGTH, _ = pathogen_parameters(pathogen, import_multiplier=import_multiplier, skip_incidence=True)
    else:
        FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX = fixed_params[:4]
        REC_UP, REC_SAME, IMPORT_STRENGTH = fixed_params[-3:]
    
    if "clike" in option2:
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            print(CONTACT_MATRIX[12])
            print(CONTACT_MATRIX[-4])
        print(option2)
        match = re.search(r'(\d)clike(\d)', option2)
        print(match, match.group(1), match.group(2))
        if match:
            i = int(match.group(1))
            j = int(match.group(2))
            if true_NAG==65:
                age_map = [jnp.arange(0, 3), jnp.arange(3, 12), jnp.arange(12, 5*12)] + [jnp.array([5*12+k]) for k in range(7 + ("split" in option1) - 3)]
                target_idx = age_map[i]
                source_idx = age_map[j]
                source_rows = CONTACT_MATRIX[source_idx[0], :]
                if source_rows.ndim == 1:
                    source_rows = source_rows[None, :]
                if source_rows.shape[0] == 1 and target_idx.shape[0] > 1:
                    source_rows = jnp.repeat(source_rows, target_idx.shape[0], axis=0)
                print(target_idx, source_idx)
                print(source_rows)
                CONTACT_MATRIX = CONTACT_MATRIX.at[target_idx, :].set(source_rows)
            else:
                CONTACT_MATRIX = CONTACT_MATRIX.at[i,:].set(CONTACT_MATRIX[j,:])
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            print(CONTACT_MATRIX[12])
            print(CONTACT_MATRIX[-4])
    if "cboost" in option2:
        match = re.search(r'cboost(\d+)', option2)
        CBOOST = int(match.group(1))
        if true_NAG==65:
            print("Contact boost:", CBOOST)
            CONTACT_MATRIX = CONTACT_MATRIX.at[12:5*12,12:5*12].set(CONTACT_MATRIX[12:5*12,12:5*12]*CBOOST)
        else:
            CONTACT_MATRIX = CONTACT_MATRIX.at[2,2].set(CONTACT_MATRIX[2,2]*CBOOST)

    if rescale is not None:
        x = rescale[:,0] + x * (rescale[:,1] - rescale[:,0])
    
    N_S = 3
    EPOCH = pd.to_datetime('1970-01-01')
    END = pd.to_datetime(end_date)
    FULL_PERIOD = pd.date_range(start=EPOCH, end=END, freq='D')
    FULL_POINTS = jnp.array(date_to_t(FULL_PERIOD))

    n = 0
    if pathogen == "sim":
        REC_UP = jnp.array([x[n],x[n+1],0.0])
        REC_SAME = jnp.array([0.0,0.0,x[n+1]])
        n += 2
    if "fixbetap" in option2:
        match = re.search(r'fixbetap(\d+)', option2)
        BETA = int(match.group(1)) / (10 ** len(match.group(1)))
    else:
        BETA = x[n]
        n+=1
    if "seasp" in option2:
        match = re.search(r'seasp(\d+)', option2)
        SEASONALITY = int(match.group(1)) / (10 ** len(match.group(1)))
    else:
        SEASONALITY = x[n]
        n+=1
    if "phasep" in option2:
        match = re.search(r'phasep(\d+)', option2)
        OFFSET = int(match.group(1)) / (10 ** len(match.group(1)))
    else:
        OFFSET = x[n]
        n+=1
    MATERNAL_IMMUNITY = jnp.zeros((len(FULL_POINTS), N_S))
    if "wane" in option1:
        WANE = jnp.array([0.0,x[n],x[n+1]])
        n += 2
    elif "SIRS" in option1:
        WANE = jnp.array([0.0,x[n],0.0])
        n += 1
    else:
        WANE = jnp.array([0.0,0.0,x[n]])
        n += 1
    if "SIRS" in option1:
        S_REL = jnp.zeros(N_S)
        S_REL = S_REL.at[0].set(1)
        pobsrel = jnp.ones(N_S)
    elif ("Influenza" in pathogen) and ("free" not in pathogen) and ('nr' not in option2):
        srel, pobsrel = constrained_immunity(x[n],x[n+1],x[n+2])
        S_REL = srel
        n += 3
    elif (pathogen == 'RSV') and ('nr' not in option2):
        S_REL = jnp.array([1,x[n],x[n]*x[n+1]])
        pobsrel = jnp.array([1,0.46,0.31]) # Henderson 1979
        n += 2
    else:
        S_REL = jnp.array([1,x[n],x[n]*x[n+1]])
        pobsrel = jnp.array([1,x[n+2],x[n+2]*x[n+3]])
        n += 4
    if ('flexage' not in option2) and ('maxagep' not in option2):
        P_OBS = x[n]*pobsrel
        n += 1
    else:
        P_OBS = pobsrel
    if "irel" in option1:
        I_REL = jnp.array([1,x[n],x[n]*x[n+1]])
        n += 2
    else:
        I_REL = jnp.array([1,1,1])
    if "maxmimm" in option1:
        MIMM = 1
        MATERNAL_IMMUNITY = MATERNAL_IMMUNITY.at[:,-1].set(MIMM)
    elif "totalmimm" in option1:
        MIMM = 1
        MATERNAL_IMMUNITY = MATERNAL_IMMUNITY.at[:,:].set(MIMM)
    elif "mimm" in option1:
        MIMM = x[n]
        MATERNAL_IMMUNITY = MATERNAL_IMMUNITY.at[:,-1].set(MIMM)
        n += 1
    if "RSV" in pathogen:
        MATERNAL_IMMUNITY = jnp.minimum(1, MATERNAL_IMMUNITY + rsv_maternal_immunity(FULL_POINTS).reshape(-1,1))
    if 'pathogen' not in option1:
        if lockdown == 'Default':
            PIECEWISE_CONTACT = jax.vmap(lambda t: cm.piecewise(t, defaultTT, defaultFF, steepness=0.2))(FULL_POINTS)
            RELATIVE_CONTACT = PIECEWISE_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
            n+=7
        elif lockdown == 'FlexStepwise':
            TT = jnp.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[n]*365,date_to_t('2020-03-19')+(x[n]+x[n+1])*365,date_to_t('2020-03-19')+(x[n]+x[n+1]+x[n+2])*365])
            # Fs - element 2 must be bigger than element 1, element 3 must be smaller than element 2, element 4 must be bigger than element 2
            F1 = x[n+3] # value between 0 and 1 (first lockdown)
            F2 = F1 + x[n+4] - F1*x[n+4] # value between x[n+3] and 1 (inter-lockdown)
            F3 = F2*x[n+5] # value less than F2 (second lockdown)
            F4 = F2 + x[n+6] - F2*x[n+6] # value between F2 and 1 (post-lockdown)
            FF = jnp.array([1,F1,F2,F3,F4])
            PIECEWISE_CONTACT = jax.vmap(lambda t: cm.piecewise(t, TT, FF, steepness=0.2))(FULL_POINTS)
            RELATIVE_CONTACT = PIECEWISE_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
            n += 7
        elif lockdown == 'Mobility':
            FF = [x[n],x[n+1],]
            contact_factor = 1 + FF[0]*cm.MOBILITY_CHANGE_JAX
            MOBILITY_CONTACT = jnp.ones(len(FULL_POINTS))
            MOBILITY_CONTACT = MOBILITY_CONTACT.at[cm.MOBILITY_START:cm.MOBILITY_END+1].set(contact_factor)
            MOBILITY_CONTACT = MOBILITY_CONTACT.at[cm.MOBILITY_END+1:].set(FF[1])
            RELATIVE_CONTACT = MOBILITY_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
            n += 2
        elif lockdown == 'Mobility2':
            FF = [x[n],x[n+1],x[n+2],]
            contact_factor = 1 + FF[0]*cm.MOBILITY_CHANGE_JAX + FF[1]*cm.MOBILITY_CHANGE_JAX**2
            MOBILITY_CONTACT = jnp.ones(len(FULL_POINTS))
            MOBILITY_CONTACT = MOBILITY_CONTACT.at[cm.MOBILITY_START:cm.MOBILITY_END+1].set(contact_factor)
            MOBILITY_CONTACT = MOBILITY_CONTACT.at[cm.MOBILITY_END+1:].set(FF[2])
            RELATIVE_CONTACT = MOBILITY_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
            n += 3
        elif lockdown == "Taube":
            contact_factor = (2.63692872 + 1.96488539*cm.MOBILITY_CHANGE_JAX + -0.03890451*cm.MOBILITY_CHANGE_JAX**2)/2.63692872
            MOBILITY_CONTACT = jnp.ones(len(FULL_POINTS))
            MOBILITY_CONTACT = MOBILITY_CONTACT.at[cm.MOBILITY_START:cm.MOBILITY_END+1].set(contact_factor)
            RELATIVE_CONTACT = MOBILITY_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
        elif "ExponentialFixed" in lockdown:
            FF = [1,0.2]
            TT = [date_to_t(EPOCH), date_to_t('2020-03-19')]
            RR = [0.005,]
            EXPONENTIAL_CONTACT = cm.exponential_recovery(FULL_POINTS, TT, FF, RR)
            RELATIVE_CONTACT = EXPONENTIAL_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
        elif "ExponentialInOut" in lockdown:
            FF = [1,x[n]]
            TT = [date_to_t(EPOCH), date_to_t('2020-01-20'), date_to_t('2020-03-19')]
            RR = [x[n+1],]
            EXPONENTIAL_CONTACT = cm.exponential_in_and_out(FULL_POINTS, TT, FF, RR)
            RELATIVE_CONTACT = EXPONENTIAL_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
            n += 2
        elif "ExponentialByAge" in lockdown:
            match = re.search(r'\d', lockdown)
            age_partition = int(match.group()) if match else 6
            FF = [1,x[n]]
            TT = [date_to_t(EPOCH), date_to_t('2020-03-19')]
            RR = jnp.array([[x[n+1],x[n+2],],])
            EXPONENTIAL_CONTACT = cm.exponential_recovery_byage(FULL_POINTS, TT, FF, RR, age_partition=age_partition, NAG=NAG)
            RELATIVE_CONTACT = EXPONENTIAL_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS.reshape(-1,1)-274)/365-OFFSET)))
            n += 3
        elif "Exponential2" in lockdown:
            FF = [1,x[n],x[n+1]]
            TT = [date_to_t(EPOCH), date_to_t('2020-03-19'), date_to_t('2020-03-19')+x[n+2]*365]
            RR = [x[n+3],x[n+4]]
            EXPONENTIAL_CONTACT = cm.exponential_recovery(FULL_POINTS, TT, FF, RR)
            RELATIVE_CONTACT = EXPONENTIAL_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
            n += 5
        elif "Exponential" in lockdown:
            FF = [1,x[n]]
            TT = [date_to_t(EPOCH), date_to_t('2020-03-19')]
            RR = [x[n+1],]
            EXPONENTIAL_CONTACT = cm.exponential_recovery(FULL_POINTS, TT, FF, RR)
            RELATIVE_CONTACT = EXPONENTIAL_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
            n += 2
        elif lockdown == "Sigmoid":
            FF = [1, x[n]]
            TT = [date_to_t(EPOCH), date_to_t('2020-03-19'), date_to_t('2020-03-19')+x[n+1]*365]
            RR = [x[n+2],]
            SIGMOID_CONTACT = cm.sigmoid_recovery(FULL_POINTS, TT, FF, RR)
            RELATIVE_CONTACT = SIGMOID_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
            n += 3
        elif lockdown == "RSV0415":
            TT = jnp.array([date_to_t('1970-01-01'), date_to_t('2020-03-19'), date_to_t('2020-09-30'), date_to_t('2021-08-18'), date_to_t('2022-06-15')])
            F1 = x[n] # value between 0 and 1 (first lockdown)
            F2 = F1 + x[n+1] - F1*x[n+1] # value between F1 and 1 (inter-lockdown)
            F3 = F2*x[n+2] # value less than F2 (second lockdown)
            F4 = F2 + x[n+3] - F2*x[n+3] # value between F2 and 1 (post-lockdown)
            FF = jnp.array([1,F1,F2,F3,F4])
            PIECEWISE_CONTACT = jax.vmap(lambda t: cm.piecewise(t, TT, FF, steepness=0.2))(FULL_POINTS)
            RELATIVE_CONTACT = PIECEWISE_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
            n += 4
        elif lockdown == "PolicyDates":
            TT = np.array([date_to_t(EPOCH),
                date_to_t('2020-03-19'), # Newsom announces stay-at-home order
                date_to_t('2021-04-27'), # CDC amends mask guidance to allow vaccinated individuals to go maskless
                date_to_t('2021-12-15'), # CDC reinstates mask guidance
                date_to_t('2022-03-01')]) # End of mask mandate in California
            F1 = x[n] # value between 0 and 1 (first lockdown)
            F2 = F1 + x[n+1] - F1*x[n+1] # value between F1 and 1 (inter-lockdown)
            F3 = F2*x[n+2] # value less than F2 (second lockdown)
            F4 = F2 + x[n+3] - F2*x[n+3] # value between F2 and 1 (post-lockdown)
            FF = jnp.array([1,F1,F2,F3,F4])
            PIECEWISE_CONTACT = jax.vmap(lambda t: cm.piecewise(t, TT, FF, steepness=0.2))(FULL_POINTS)
            RELATIVE_CONTACT = PIECEWISE_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
            n += 4
    if re.search(r'\d{6}',lockdown):
        try:
            with open("Data/Processed/DE_cm_opt_"+str(lockdown)+".pickle","rb") as f:
                opt = pickle.load(f)
        except:
            with open("Data/Processed/DE_cm_opt_"+str(lockdown)+"intermediate.pickle","rb") as f:
                opt = pickle.load(f)
        x_lockdown = opt.x
        TT = jnp.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x_lockdown[0]*365,date_to_t('2020-03-19')+(x_lockdown[0]+x_lockdown[1])*365,date_to_t('2020-03-19')+(x_lockdown[0]+x_lockdown[1]+x_lockdown[2])*365])
        # Fs - element 2 must be bigger than element 1, element 3 must be smaller than element 2, element 4 must be bigger than element 2
        F1 = x_lockdown[3] # value between 0 and 1 (first lockdown)
        F2 = F1 + x_lockdown[4] - F1*x_lockdown[4] # value between F1 and 1 (inter-lockdown)
        F3 = F2*x_lockdown[5] # value less than F2 (second lockdown)
        F4 = F2 + x_lockdown[6] - F2*x_lockdown[6] # value between F2 and 1 (post-lockdown)
        FF = jnp.array([1,F1,F2,F3,F4])
        PIECEWISE_CONTACT = jax.vmap(lambda t: cm.piecewise(t, TT, FF, steepness=0.2))(FULL_POINTS)
        RELATIVE_CONTACT = PIECEWISE_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
        if pathogen != "sim":
            n += 7
    elif 'pathogen' in option1:
        RELATIVE_CONTACT = fixed_params[9]
    if "ODipTune" in lockdown:
        FO = FF[1] + x[n] - FF[1]*x[n]
        dipdates = jnp.array([date_to_t(EPOCH),
                    date_to_t('2021-12-15'),
                    date_to_t('2022-03-01')])
        dipvalues = jnp.array([1, 1-FO, 1])
        ODIP_CONTACT = jax.vmap(lambda t: cm.piecewise(t, dipdates, dipvalues, steepness=0.2))(FULL_POINTS)
        RELATIVE_CONTACT = ODIP_CONTACT[:, None] * RELATIVE_CONTACT if RELATIVE_CONTACT.ndim == 2 else ODIP_CONTACT * RELATIVE_CONTACT
    elif "ODipLinear" in lockdown:
        FO = 0.75*FF[1] + 0.25
        dipdates = jnp.array([date_to_t(EPOCH),
                    date_to_t('2021-12-15'),
                    date_to_t('2022-03-01')])
        dipvalues = jnp.array([1, 1-FO, 1])
        ODIP_CONTACT = jax.vmap(lambda t: cm.piecewise(t, dipdates, dipvalues, steepness=0.2))(FULL_POINTS)
        RELATIVE_CONTACT = ODIP_CONTACT[:, None] * RELATIVE_CONTACT if RELATIVE_CONTACT.ndim == 2 else ODIP_CONTACT * RELATIVE_CONTACT
    elif "ODipEqual" in lockdown:
        FO = FF[1]
        dipdates = jnp.array([date_to_t(EPOCH),
                    date_to_t('2021-12-15'),
                    date_to_t('2022-03-01')])
        dipvalues = jnp.array([1, 1-FO, 1])
        ODIP_CONTACT = jax.vmap(lambda t: cm.piecewise(t, dipdates, dipvalues, steepness=0.2))(FULL_POINTS)
        RELATIVE_CONTACT = ODIP_CONTACT[:, None] * RELATIVE_CONTACT if RELATIVE_CONTACT.ndim == 2 else ODIP_CONTACT * RELATIVE_CONTACT
    if "months" in option1:
        true_OBS_AGE = jnp.zeros(true_NAG)
        NAG = 7 + ("split" in option1)
    if ('maxagep' in option2) & ('dynamic' not in option1):
        if "RSV" in pathogen:
            fixed_age_index = 0
        else:
            fixed_age_index = -1
        # search for numbers after maxagep in option2, that number divided by 100 is the value of OBS_AGE for the fixed age group
        match = re.search(r'maxagep(\d+)', option2)
        obs_age_max = int(match.group(1)) / (10 ** len(match.group(1)))
        OBS_AGE = jnp.array([0.0]*NAG)
        OBS_AGE = OBS_AGE.at[fixed_age_index].set(obs_age_max)
        # at all other indexes, use x[n:n+NAG-1] in order
        OBS_AGE = OBS_AGE.at[1+fixed_age_index:NAG+fixed_age_index].set(obs_age_max * x[n:n+NAG-1])
        n += NAG-1
    elif ('flexage' in option2) & ('dynamic' not in option1):
        OBS_AGE = x[n:n+NAG]
        n += NAG
    elif 'dynamic' in option1:
        OBS_AGE = fixed_params[7]
    if "months" in option1:
        true_OBS_AGE = true_OBS_AGE.at[0:3].set(OBS_AGE[0])
        true_OBS_AGE = true_OBS_AGE.at[3:12].set(OBS_AGE[1])
        true_OBS_AGE = true_OBS_AGE.at[12:12*5].set(OBS_AGE[2])
        true_OBS_AGE = true_OBS_AGE.at[12*5:].set(OBS_AGE[3:])
    if (("Influenza" in pathogen) and ("nvax" not in option1)):
        protection_param = S_REL*P_OBS
        max_eff = (protection_param[-2]-protection_param[-1])/protection_param[-2]
        VAX_RATE = flu_eff_vax_rate(FULL_POINTS, max_eff)
        # if NAG is greater than 7, duplicate the fifth row (index 4) to fill out the additional age groups
        if true_NAG == 8:
            VAX_RATE = jnp.concatenate((VAX_RATE[:, :5], jnp.tile(VAX_RATE[:, 4:5], (1, NAG-7)), VAX_RATE[:, 5:]), axis=1)
        elif true_NAG == 65:
            VAX_RATE = jnp.concatenate((jnp.tile(VAX_RATE[:, 0:1], (1, 3)), jnp.tile(VAX_RATE[:, 1:2], (1, 9)), jnp.tile(VAX_RATE[:, 2:3], (1, 12*4)), VAX_RATE[:, 3:5], jnp.tile(VAX_RATE[:, 4:5], (1, 1)), VAX_RATE[:, 5:]), axis=1)
    elif (("RSV" in pathogen) and ("nvax" not in option1)) or ("rsvvax" in option1):
        protection_param = S_REL*P_OBS
        max_eff0 = 1 - protection_param[-1]
        max_eff1 = (protection_param[-2]-protection_param[-1])/protection_param[-2]
        VAX_RATE = rsv_eff_vax_rate(FULL_POINTS, max_eff0, max_eff1)
        if true_NAG == 8:
            VAX_RATE = jnp.concatenate((VAX_RATE[:, :5], jnp.tile(VAX_RATE[:, 4:5], (1, NAG-7)), VAX_RATE[:, 5:]), axis=1)
        elif true_NAG == 65:
            VAX_RATE = jnp.concatenate((jnp.tile(VAX_RATE[:, 0:1], (1, 3)), jnp.tile(VAX_RATE[:, 1:2], (1, 9)), jnp.tile(VAX_RATE[:, 2:3], (1, 12*4)), VAX_RATE[:, 3:5], jnp.tile(VAX_RATE[:, 4:5], (1, 1)), VAX_RATE[:, 5:]), axis=1)
    else:
        if "months" in option1:
            VAX_RATE = jnp.zeros((len(FULL_POINTS),true_NAG))
        else:
            VAX_RATE = jnp.zeros((len(FULL_POINTS),NAG))

    # if relative contact is 1-dimensional, copy it across all age groups
    if RELATIVE_CONTACT.ndim == 1:
        if "months" in option1:
            RELATIVE_CONTACT = jnp.sqrt(jnp.tile(RELATIVE_CONTACT.reshape(-1,1), (1,true_NAG)))
        else:
            RELATIVE_CONTACT = jnp.sqrt(jnp.tile(RELATIVE_CONTACT.reshape(-1,1), (1,NAG)))
    
    if "daycare" in option2:
        parent_labor = [60.98262489901673, 61.5063564713625, 60.84281207451028, 60.768041939091475, 60.5037381418266, 61.10016011287247, 61.148622423778086, 62.99393722242363, 63.99372275576328, 63.5827420015764, 64.12485171621293, 64.66696143084945, 66.22302672280338, 67.85271452382989, 68.58049720957207]
        parent_labor = jnp.asarray(parent_labor)/100
        relative_parent_labor = (parent_labor-0.64124852) # scale by 2020 value
        # times are epoch then first of each year from 2010 to 2024
        parent_labor_times = jnp.array([date_to_t(pd.to_datetime(d)) for d in [f'{y}-01-01' for y in range(2010, 2025)]])
        # interpolate relative_parent_labor to FULL_POINTS
        relative_parent_labor_interp = jnp.interp(FULL_POINTS, parent_labor_times, relative_parent_labor)
        # scale all contact between the first four age groups by relative_parent_labor_interp
        # x[n] is the proportion of all contacts between these age groups that occur in daycare
        # if daycare is followed by p, then a number, set the value of DAYCARE to be that number divided by 10
        match = re.search(r'daycarep(\d+)', option2)
        if match:
            DAYCARE = int(match.group(1))/10
        else:
            DAYCARE = x[n]
            n += 1
        RELATIVE_CONTACT = RELATIVE_CONTACT.at[:, 1:4].set(RELATIVE_CONTACT[:, 1:4] * (1 + DAYCARE * relative_parent_labor_interp.reshape(-1,1)))

    if "months" in option1:
        params = (FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX,
                BETA, WANE, S_REL, I_REL, P_OBS, true_OBS_AGE, RELATIVE_CONTACT, VAX_RATE, MATERNAL_IMMUNITY,
                REC_UP, REC_SAME, IMPORT_STRENGTH)
    else:
        params = (FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX,
                BETA, WANE, S_REL, I_REL, P_OBS, OBS_AGE, RELATIVE_CONTACT, VAX_RATE, MATERNAL_IMMUNITY,
                REC_UP, REC_SAME, IMPORT_STRENGTH)
    
    if print_params:
        param_names = ["BETA","WANE","SEASONALITY","OFFSET","S_REL","I_REL","P_OBS","OBS_AGE"]
        if "daycare" in option2:
            param_names += ["DAYCARE"]
        if lockdown != "Taube":
            param_names += ["FF"]
            if "ODip" in lockdown:
                param_names += ["FO"]
            if "Mobility" not in lockdown:
                param_names += ["TT"]
            if "Exponential" in lockdown or "Sigmoid" in lockdown:
                param_names += ["RR"]
        for i in range(len(param_names)):
            if param_names[i] == "TT" and "Mobility" not in lockdown:
                print("TT: " + [t_to_date(t).strftime('%Y-%m-%d') for t in eval(param_names[i])].__str__())
            else:
                print(param_names[i]+": "+eval(param_names[i]).__str__())
        if "mimm" in option1 or "maxmimm" in option1:
            print("MATERNAL_IMMUNITY: "+MIMM.__str__())
    
    if "ODip" in lockdown:
        contact_multiplier = ODIP_CONTACT
    else:
        contact_multiplier = 1

    if return_contact:
        if 'Exponential' in lockdown:
            return params, contact_multiplier * EXPONENTIAL_CONTACT
        if 'Sigmoid' in lockdown:
            return params, SIGMOID_CONTACT*contact_multiplier
        elif 'Mobility' in lockdown:
            return params, MOBILITY_CONTACT*contact_multiplier
        elif 'Taube' in lockdown:
            return params, MOBILITY_CONTACT*contact_multiplier
        else:
            return params, PIECEWISE_CONTACT*contact_multiplier
    else:
        return params

def parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=7):
    bounds_dict = {"WANE2": [0,1e-2]}
    if "seasp" not in option2:
        bounds_dict["SEASONALITY"] = [0,1]
    if "phasep" not in option2:
        bounds_dict["OFFSET"] = [0,1]
    if "fixbetap" not in option2:
        bounds_dict["BETA"] = [0,0.3]
    if "wane" in option1:
        bounds_dict["WANE1"] = [0,1e-2]
    if option1 == "nb":
        bounds_dict["OVERDISPERSION"] = [-5,10]
    elif ("mimm" in option1) & ("maxmimm" not in option1):
        bounds_dict["MATERNAL_IMMUNITY"] = [0,1]
    if "SIRS" not in option1:
        if ("Influenza" in pathogen) and ("free" not in pathogen) and ("nr" not in option2):
            bounds_dict["EXTRA_IMMUNITY"] = [0,1]
            bounds_dict["FIRST_IMMUNITY"] = [0.1,1]
            bounds_dict["FIRST_DIS_INF_FACTOR"] = [0,1]
        elif ("RSV" in pathogen) and ("nr" not in option2):
            bounds_dict["S_REL1"] = bounds_dict["S_REL2"] = [0.1,1]
        else:
            bounds_dict["S_REL1"] = bounds_dict["S_REL2"] = bounds_dict["D_REL1"] = bounds_dict["D_REL2"] = [0.1,1]
    if "irel" in option1:
        bounds_dict["I_REL1"] = bounds_dict["I_REL2"] = [0.1,1]
    if "dynamic" not in option1 and "pp" not in option2:
        if "months" in option1:
            NAG = 7 + ("split" in option1)
            true_NAG = 65
        if "flexage" in option2:
            if option2 == "flexage":
                upper_bound = 0.005
            else:
                # Extract decimal value from option2 (e.g., "flexagep01" -> 0.01, "flexagep5" -> 0.5)
                match = re.search(r'flexagep(\d+)', option2)
                if match:
                    upper_bound = int(match.group(1)) / (10 ** len(match.group(1)))
                else:
                    upper_bound = 0.01
            for i in range(1,NAG+1):
                bounds_dict[f"AGE_OBS_{i}"] = [0, upper_bound]
        elif "maxagep" in option2:
            start_index = "RSV" in pathogen
            for i in range(1+start_index,NAG+start_index):
                bounds_dict[f"AGE_OBS_{i}"] = [0, 1]
    if "pp" in option2:
        if option2 == "ppflexage":
            lower_bound = 0.05
        else:
            # Extract decimal value from option2 (e.g., "flexagep01" -> 0.01, "flexagep5" -> 0.5)
            match = re.search(r'ppflexagep(\d+)', option2)
            if match:
                lower_bound = int(match.group(1)) / (10 ** len(match.group(1)))
            else:
                lower_bound = 0.05
        for i in range(1,NAG+1):
            bounds_dict[f"AGE_OBS_{i}"] = [lower_bound, 1]
    if "pathogen" not in option1:
        if lockdown == "Mobility":
            bounds_dict["F1"] = bounds_dict["F2"] = [0,2]
        elif lockdown == "Mobility2":
            bounds_dict["F1"] = bounds_dict["F2"] = bounds_dict["F3"] = [0,2]
        elif ("Exponential" in lockdown) and (("ByAge" not in lockdown) and ("2" not in lockdown)):
            if "Max" in lockdown:
                if "Maxp" in lockdown:
                    match = re.search(r'Maxp(\d+)', lockdown)
                    upper_bound = int(match.group(1)) / (10 ** len(match.group(1)))
                else:
                    upper_bound = 0.7
                bounds_dict["F1"] = [0, upper_bound]
            else:
                bounds_dict["F1"] = [0,1]
            bounds_dict["R1"] = [0.002,0.01]
        elif "ExponentialByAge" in lockdown:
            bounds_dict["F1"] = [0,1]
            bounds_dict["R1"] = bounds_dict["R2"] = [0.001,0.01]
        elif lockdown == "Exponential2":
            bounds_dict["F1"] = bounds_dict["F2"] = [0,1]
            bounds_dict["DT1"] = [0,2]
            bounds_dict["R1"] = bounds_dict["R2"] = [0.002,0.05]
        elif lockdown == "Sigmoid":
            bounds_dict["F1"] = [0,1]
            bounds_dict["DT1"] = [0,3]
            bounds_dict["R1"] = [0.002,0.01]
        elif lockdown == "RSV0415":
            bounds_dict["F1"] = bounds_dict["F2"] = bounds_dict["F3"] = bounds_dict["F4"] = [0,1]
        elif lockdown != "Taube":
            bounds_dict["DT1"] = bounds_dict["DT2"] = bounds_dict["DT3"] = bounds_dict["F1"] = bounds_dict["F2"] = bounds_dict["F3"] = bounds_dict["F4"] = [0,1]
        if "ODipTune" in lockdown:
            bounds_dict["FO"] = [0,1]
    if ("daycare" in option2) and ("daycarep" not in option2):
        bounds_dict["DAYCARE"] = [0,1]
    # reorder bounds_dict to match order in x
    bounds_dict = {key: bounds_dict[key] for key in ["BETA","SEASONALITY","OFFSET","WANE1","WANE2","IMPORT_RATE","EXTRA_IMMUNITY","FIRST_IMMUNITY","FIRST_DIS_INF_FACTOR","S_REL1","S_REL2","D_REL1","D_REL2","I_REL1","I_REL2","P_OBS","MATERNAL_IMMUNITY","F1","F2","F3","F4","DT1","DT2","DT3","R1","R2","FO","OVERDISPERSION","AGE_OBS_YOUNG","AGE_OBS_OLD","AGE_OBS_YOUNG_OLD","AGE_OBS_MATERNAL","AGE_OBS_1","AGE_OBS_2","AGE_OBS_3","AGE_OBS_4","AGE_OBS_5","AGE_OBS_6","AGE_OBS_7","AGE_OBS_8","AGE_OBS_9","DAYCARE"]\
        if key in bounds_dict.keys()}
    bounds = jnp.array(list(bounds_dict.values()))
    param_names = list(bounds_dict.keys())
    
    return param_names, bounds

def parameters_from_DE(pathogen, lockdown, option1, option2, seed, lockdown_x=None, import_multiplier=1e-9):
    base_path = "Data/Processed/results"+str(seed)[:6]+"/"
    filename_pattern = pathogen+lockdown+option1+option2+str(seed)+".pickle"
    opt = None
    prefix = ""
    for test_prefix in ["DE_opt_", "evosax_DE_"]:
        filepath = base_path + test_prefix + filename_pattern
        try:
            with open(filepath, "rb") as f:
                opt = pickle.load(f)
            print(f"Loaded: {test_prefix}{filename_pattern}")
            prefix = test_prefix
            break
        except FileNotFoundError:
            continue

    if opt is None:
        print('File not found with either prefix (DE_opt_ or evosax_DE_)')
        sys.exit()

    # Detect file type and extract results accordingly
    if prefix == "evosax_DE_":
        # evosax_DE format
        x = opt["final_population"][np.argmin(opt["final_fitness"])]
    else:
        # scipy.optimize.differential_evolution format
        if opt.success:
            print("Optimization converged")
        else:
            print("Optimization did not converge")
            print(opt.message)
            print(opt.x)
            sys.exit()
        x = opt.x
    
    if "split" in option1:
        from Parameters.census_population import AGING_RATE_split as AGING_RATE
    else:
        from Parameters.census_population import AGING_RATE
    REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, tests = pathogen_parameters(pathogen, import_multiplier=import_multiplier)
    CONTACT_MATRIX = jnp.asarray(pd.read_csv('Data/Processed/contact_matrices/KP'+['', '_split'][ "split" in option1] +'_contact_all_US_Census.csv', delimiter=',', header=None).values)
    BIRTH_RATE = jnp.asarray(np.genfromtxt('Data/Processed/birth_rate_daily.csv', delimiter=','))

    end_date = '2025-05-01'
    EPOCH = pd.to_datetime('1970-01-01')
    END = pd.to_datetime(end_date)
    FULL_PERIOD = pd.date_range(start=EPOCH, end=END, freq='D')
    FULL_POINTS = np.array(date_to_t(FULL_PERIOD))

    param_names, bounds = parameters_names_bounds(pathogen, lockdown, option1, option2)

    if lockdown_x is not None:
        for i,name in enumerate(param_names):
            if name in lockdown_x.keys():
                x[i] = lockdown_x[name]
        bounds = jnp.array([[0,1]]*7)
    params = x_to_params(x, pathogen, lockdown, option1, option2, fixed_params = (FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX, REC_UP, REC_SAME, IMPORT_STRENGTH), import_multiplier=import_multiplier)

    return params, param_names, bounds, tests, p_time_to_obs

def load_optimization_results(prefix, pathogen, seed, lockdown, option1, option2):
    if re.search(r'\d{6}',lockdown):
        lockdown_search = "FlexStepwise"
    else:
        lockdown_search = lockdown

    base_path = "Data/Processed/results"+str(seed)[:6]+"/"
    filename_pattern = "_"+pathogen+lockdown_search+option1+option2+str(seed)+".pickle"

    # Try both DE_opt and evosax_DE prefixes
    opt = None
    if prefix == "":
        for test_prefix in ["DE_opt", "evosax_DE", "scipy_DE", "evosax_DiffusionEvolution"]:
            filepath = base_path + test_prefix + filename_pattern
            try:
                with open(filepath, "rb") as f:
                    opt = pickle.load(f)
                print(f"Loaded: {test_prefix}{filename_pattern}")
                prefix = test_prefix
                break
            except FileNotFoundError:
                continue
    else:
        if "skip_resampling" in prefix:
            prefix = "evosax_DE"
        filepath = base_path + prefix + filename_pattern
        try:
            with open(filepath, "rb") as f:
                opt = pickle.load(f)
            print(f"Loaded: {prefix}{filename_pattern}")
        except FileNotFoundError:
            print(f"File not found: {prefix}{filename_pattern}")

    if opt is None:
        print(base_path+filename_pattern)
        print('File not found with any of the tested prefixes (DE_opt_, scipy_DE_, evosax_DE_, evosax_DiffusionEvolution_)')
        sys.exit()

    # Detect file type and extract results accordingly
    if "evosax" in prefix:
    # evosax_DE format
        x = opt["final_population"][np.argmin(opt["final_fitness"])]
        neg_log_likelihood = np.min(opt["final_fitness"])
        print("std of final fitness: "+str(jnp.std(opt["final_fitness"]))+"mean of final fitness: "+str(jnp.mean(opt["final_fitness"])) + "ratio: "+str(jnp.std(opt["final_fitness"])/jnp.abs(jnp.mean(opt["final_fitness"]))))
        if jnp.std(opt["final_fitness"]) <= 0.01 * jnp.abs(jnp.mean(opt["final_fitness"])):
            print("evosax converged according to scipy criteria")
        else:
            print("evosax did not converge according to scipy criteria")
    # scipy.optimize.differential_evolution format
    elif ("scipy_DE" in prefix) or ("DE_opt" in prefix):
        if opt.success:
            print("Optimization converged")
        else:
            print("Optimization did not converge")
            print(opt.message)
            print(opt.x)
            sys.exit()
        x = opt.x
        neg_log_likelihood = opt.fun
    return prefix,x,neg_log_likelihood

# unified x from DE, i.e. x is same length regardless of model options. assume option2=flexage, lockdown=FlexStepwise
def consistent_x_from_DE(pathogen, lockdown, option1, option2, seed, prefix="", NAG=7):
    _, x_DE, _ = load_optimization_results(prefix, pathogen, seed, lockdown, option1, option2)
    REC_UP, _, _, _ = pathogen_parameters(pathogen, import_multiplier=1e-9, skip_incidence=True)
    x_consistent = jnp.zeros(12 + 2*(lockdown=="Exponential" or "ExponentialODip" in lockdown or "ExponentialInOutODip" in lockdown) + 3*(lockdown=="Sigmoid" or lockdown=="ExponentialByAge") + 4*(lockdown=="RSV0415" or lockdown=="FlexStepwise") + NAG)
    x_consistent = x_consistent.at[0:2].set([REC_UP[0], REC_UP[1]]) # REC
    x_consistent = x_consistent.at[2:5].set(x_DE[0:3]) # BETA, SEASONALITY, OFFSET
    n = 3
    obs_age_start = 12
    if "wane" in option1:
        x_consistent = x_consistent.at[5:7].set(x_DE[3:5]) # WANE1, WANE2
        n += 2
    else:
        x_consistent = x_consistent.at[6].set(x_DE[3]) # WANE2
        n += 1
    if ("Influenza" in pathogen) and ("free" not in pathogen) and ('nr' not in option2):
        srel, pobsrel = constrained_immunity(x_DE[n],x_DE[n+1],x_DE[n+2])
        S_REL1 = srel[1]
        S_REL2 = srel[2]/srel[1]
        D_REL1 = pobsrel[1]
        D_REL2 = pobsrel[2]/pobsrel[1]
        x_consistent = x_consistent.at[7:11].set(jnp.array([S_REL1,S_REL2,D_REL1,D_REL2]))
        n += 3
    elif ("RSV" in pathogen) and ("nr" not in option2):
        x_consistent = x_consistent.at[7:9].set(x_DE[n:n+2]) # S_REL1, S_REL2
        x_consistent = x_consistent.at[9:11].set(jnp.array([0.46,0.31/0.46])) # D_REL1, D_REL2
        n += 2
    else:
        x_consistent = x_consistent.at[7:11].set(x_DE[n:n+4]) # S_REL1, S_REL2, D_REL1, D_REL2
        n += 4
    if "maxmimm" in option1:
        x_consistent = x_consistent.at[11].set(1) # maternal immunity
    elif "mimm" in option1:
        x_consistent = x_consistent.at[11].set(x_DE[n]) # maternal immunity
        n += 1
    if lockdown == "Exponential":
        x_consistent = x_consistent.at[12:14].set(x_DE[n:n+2]) # F1, R1
        n += 2
        obs_age_start = 14
    if lockdown == "ExponentialFixed":
        x_consistent = x_consistent.at[12:14].set([0.2,0.005]) # F1, R1
        obs_age_start = 12
    elif "ExponentialByAge" in lockdown:
        x_consistent = x_consistent.at[12:15].set(x_DE[n:n+3]) # F1, R1, R2
        n += 3
        obs_age_start = 15
    elif "Exponential" in lockdown:
        x_consistent = x_consistent.at[12:14].set(x_DE[n:n+2]) # F1, R1
        n += 2
        obs_age_start = 14
    elif lockdown == "Sigmoid":
        x_consistent = x_consistent.at[12:15].set(x_DE[n:n+3]) # F1, DT1, R1
        n += 3
        obs_age_start = 15
    elif lockdown == "RSV0415":
        x_consistent = x_consistent.at[12:16].set(x_DE[n:n+4]) # F1, F2, F3, F4
        n += 4
        obs_age_start = 16
    elif lockdown == "FlexStepwise": # this should only happen for the RSV fit
        x_consistent = x_consistent.at[12:16].set([0.7761238, 0.9229197, 0.796961, 0.9991904])
        n += 4
        obs_age_start = 16
    if "maxagep" in option2:
        match = re.search(r'maxagep(\d+)', option2)
        obs_age_max = int(match.group(1)) / (10 ** len(match.group(1)))
        OBS_AGE = jnp.zeros(NAG)
        if "RSV" in pathogen:
            OBS_AGE = OBS_AGE.at[0].set(obs_age_max)
            OBS_AGE = OBS_AGE.at[1:].set(obs_age_max * x_DE[n:n+NAG-1])
        else:
            OBS_AGE = OBS_AGE.at[:-1].set(obs_age_max * x_DE[n:n+NAG-1])
            OBS_AGE = OBS_AGE.at[-1].set(obs_age_max)
    else:
        OBS_AGE = x_DE[n:n+NAG]
    x_consistent = x_consistent.at[obs_age_start:obs_age_start+NAG].set(OBS_AGE) # AGE_OBS_1 to AGE_OBS_7
    return x_consistent

####### Generating interesting quantities from ODE results #######

def observations(result, POINTS, params, OBS_AGE, incidence=False,cap=False,N_C=2,time_conversion=30.44):

    """

    Generate observed cases or incidence from ODE results

    """

    NAG, N_S, BETA, contact, SEASONALITY, OFFSET, S_REL, I_REL, P_OBS = params["NAG"], params["N_S"], params["BETA"], params["contact"], params["SEASONALITY"], params["OFFSET"], params["S_REL"], params["I_REL"], params["P_OBS"]

    obs = np.zeros((len(POINTS),NAG))
    pop_size = np.sum(result,axis=0)
    for i_t,t in enumerate(POINTS):
        foi = BETA*np.dot(contact(t,SEASONALITY,OFFSET),np.sum((np.array([[result[jr+j,i_t] for j in range(NAG)] for jr in range(2*NAG,(2*N_S+2)*NAG,2*NAG)])*I_REL),axis=0))/pop_size[i_t]
        for i in range(N_S):
            if cap:
                class_foi = np.minimum(1,S_REL[i]*foi)
            else:
                class_foi = S_REL[i]*foi
            # S_REL*foi tells you what proportion of the population gets infected, the maximum is all of them
            # obs = obs.at[i_t,:].set(obs[i_t,:] + OBS_AGE*P_OBS[i]*class_foi*result[(N_C*i+1)*NAG:(N_C*i+2)*NAG,i_t])
            obs[i_t,:] += OBS_AGE*P_OBS[i]*class_foi*result[(N_C*i+1)*NAG:(N_C*i+2)*NAG,i_t]
    if incidence:
        obs = np.sum(obs,axis=1)/pop_size
    return(time_conversion*obs) 


def infections_by_age(result,params,N_C=2):
    """
    Generate infections attributable to each age group from ODE results
    """
    NAG, N_S, BETA, contact, SEASONALITY, OFFSET, S_REL, I_REL = params["NAG"], params["N_S"], params["BETA"], params["contact"], params["SEASONALITY"], params["OFFSET"], params["S_REL"], params["I_REL"]
    infs = np.zeros((len(result.t),NAG))
    for i_t,t in enumerate(result.t):
        infs[i_t,:] = np.sum(np.array([BETA*result.y[(N_C*i+2)*NAG:(N_C*i+3)*NAG,i_t]*I_REL[i]*np.dot(contact(t,SEASONALITY,OFFSET),np.sum([S_REL[j]*result.y[(N_C*j+1)*NAG:(N_C*j+2)*NAG,i_t] for j in range(N_S)],axis=0)) for i in range(N_S)]),axis=0)
    return(infs)

def age_of_first_infection(result,MEDIAN_AGE,NAG=7,sd=False):
    """
    Generate age of first infection from ODE results
    """
    ages = np.zeros(len(result.t))
    ages_sd = np.zeros(len(result.t))
    for i_t in range(len(result.t)):
        age_distribution = MEDIAN_AGE*result.y[2*NAG:3*NAG,i_t]/np.sum(result.y[2*NAG:3*NAG,i_t])
        ages[i_t] = np.mean(age_distribution)
        ages_sd[i_t] = np.std(age_distribution)
    if sd:
        return(ages,ages_sd)
    return(ages)

def susceptibility(solution,params,N_C=2,NAG=7,N_S=3):
    """
    Generate susceptibility by age group from ODE solutions
    """
    FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX,\
    BETA, WANE, S_REL, I_REL, P_OBS, OBS_AGE, RELATIVE_CONTACT, VAX_RATE, MATERNAL_IMMUNITY,\
    REC_UP, REC_SAME, IMPORT_STRENGTH = params
    sus = np.zeros((len(solution.ts),NAG))
    for i_t,t in enumerate(solution.ts):
        for i in range(N_S):
            sus[i_t,:] += S_REL[i]*solution.ys.T[1+N_C*i*NAG:1+(N_C*i+1)*NAG,i_t]
    return(sus)

def sum_age_to(expected_obs, max_month, AGE_GROUPS):
    results = []
    for i in range(len(AGE_GROUPS)):
        group = AGE_GROUPS[i]
        # Create a mask for columns in the age group range
        col_indices = jnp.arange(expected_obs.shape[1])
        mask = (col_indices >= group[0]) & (col_indices <= group[-1])
        # Use jnp.where to select columns instead of boolean indexing
        summed = jnp.sum(jnp.where(mask[None, :], expected_obs, 0), axis=1)
        result = jnp.where(group[-1] <= max_month, 
                          summed,
                          expected_obs[:,i-3+max_month])
        results.append(result)
    summed_obs = jnp.stack(results)
    return summed_obs.T

if __name__ == "__main__":
    extra_immunity = 0
    first_immunity = 0.1
    first_dis_inf_factor = 1
    S_REL, D_REL = constrained_immunity(extra_immunity,first_immunity,first_dis_inf_factor,MIN_EFF=0.6)
    print("S_REL: ", S_REL)
    print("D_REL: ", D_REL)
