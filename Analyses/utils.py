# import jax.numpy as np
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

@jax.jit
def constrained_immunity(extra_immunity,first_immunity,first_dis_inf_factor,DIS_INF_RATIO=0.674,MIN_EFF=0.39,CHILD_EFF_RATIO=1.54):
    """
    Convert free parameters between 0 and 1 to immunity parameters constrained by flu vaccine data
    """
    ESP = MIN_EFF + extra_immunity*(1/CHILD_EFF_RATIO - MIN_EFF)
    factor = ((1 - DIS_INF_RATIO) + jnp.sqrt(1 + DIS_INF_RATIO**2 + 2*DIS_INF_RATIO*(1-2*ESP)))/2
    S1 = ((1-CHILD_EFF_RATIO*ESP)/(1-ESP) + first_immunity*(1-(1-CHILD_EFF_RATIO*ESP)/(1-ESP)))**first_dis_inf_factor
    S2 = S1*(1-ESP)/factor
    D1 = ((1-CHILD_EFF_RATIO*ESP)/(1-ESP) + first_immunity*(1-(1-CHILD_EFF_RATIO*ESP)/(1-ESP)))**(1-first_dis_inf_factor)
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

def pathogen_parameters(pathogen, import_multiplier=1e-9, skip_incidence=False):
    ARRIVALS = jnp.asarray(np.genfromtxt('Data/Processed/arrivals_daily.csv', delimiter=','))
    if 'RSV' in pathogen:
        REC_UP = jnp.array([1/4.9,1/4.1,0.0])
        REC_SAME = jnp.array([0.0,0.0,1/4.1])
        IMPORT_STRENGTH = import_multiplier*ARRIVALS*jnp.asarray(np.genfromtxt('Data/Processed/RSV_positivity_daily.csv', delimiter=','))
        p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
        positives = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_RSV_positive_counts.csv', header=None).values)
        total_tests = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_RSV_total_counts.csv', header=None).values)
    elif 'InfluenzaA' in pathogen:
        REC_UP = jnp.array([1/3.0,1/3.0,0.0])
        REC_SAME = jnp.array([0.0,0.0,1/3.0])
        IMPORT_STRENGTH = import_multiplier*ARRIVALS*jnp.asarray(np.genfromtxt('Data/Processed/InfluenzaA_positivity_daily.csv', delimiter=','))
        p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
        positives = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_InfluenzaA_positive_counts.csv', header=None).values)
        total_tests = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_InfluenzaA_total_counts.csv', header=None).values)
    elif 'InfluenzaB' in pathogen:
        REC_UP = jnp.array([1/3.0,1/3.0,0.0])
        REC_SAME = jnp.array([0.0,0.0,1/3.0])
        IMPORT_STRENGTH = import_multiplier*ARRIVALS*jnp.asarray(np.genfromtxt('Data/Processed/InfluenzaB_positivity_daily.csv', delimiter=','))
        p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/Influenza_B_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
        positives = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_InfluenzaB_positive_counts.csv', header=None).values)
        total_tests = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_InfluenzaB_total_counts.csv', header=None).values)
    elif 'Parainfluenza3' in pathogen:
        REC_UP = jnp.array([1/3.0,1/3.0,0.0])
        REC_SAME = jnp.array([0.0,0.0,1/3.0])
        IMPORT_STRENGTH = import_multiplier*ARRIVALS*jnp.asarray(np.genfromtxt('Data/Processed/Parainfluenza3_positivity_daily.csv', delimiter=','))
        p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
        positives = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_Parainfluenza3_positive_counts.csv', header=None).values)
        total_tests = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_Parainfluenza3_total_counts.csv', header=None).values)
    elif 'Adenovirus' in pathogen:
        REC_UP = jnp.array([1/3.0,1/3.0,0.0])
        REC_SAME = jnp.array([0.0,0.0,1/3.0])
        IMPORT_STRENGTH = import_multiplier*ARRIVALS*jnp.asarray(np.genfromtxt('Data/Processed/Adenovirus_positivity_daily.csv', delimiter=','))
        p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
        positives = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_Adenovirus_positive_counts.csv', header=None).values)
        total_tests = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_Adenovirus_total_counts.csv', header=None).values)
    elif 'Metapneumovirus' in pathogen:
        REC_UP = np.array([1/4.9,1/4.1,0.0]) # based on RSV - Okiro 2010
        REC_SAME = np.array([0.0,0.0,1/4.1]) # based on RSV - Okiro 2010
        IMPORT_STRENGTH = import_multiplier*ARRIVALS*jnp.asarray(np.genfromtxt('Data/Processed/Metapneumovirus_positivity_daily.csv', delimiter=','))
        p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
        positives = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_Metapneumovirus_positive_counts.csv', header=None).values)
        total_tests = jnp.asarray(pd.read_csv(f'Data/Processed/KPSC_panel_Metapneumovirus_total_counts.csv', header=None).values)
    # elif "test" in pathogen:
    #     REC_UP = jnp.array([1/3.0,1/3.0,0.0])
    #     REC_SAME = jnp.array([0.0,0.0,1/3.0])
    #     IMPORT_STRENGTH = import_multiplier*ARRIVALS*jnp.asarray(np.genfromtxt('Data/Processed/Metapneumovirus_positivity_daily.csv', delimiter=','))
    #     p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
    #     if not skip_incidence:
    #         incidence = jnp.asarray(pd.read_csv("Data/Processed/KPSC_ARI_"+pathogen+"_incidence_age_daily.csv",index_col=0))
    # if skip_incidence:
    #     return REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs
    # join positives and total_tests into a single array with shape (time, age_group, 2)
    tests = jnp.stack((total_tests, positives), axis=-1)
    return REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, tests


# @partial(jax.jit, static_argnames=['pathogen','lockdown','option1','option2'])

# jax-safe importations for x_to_params
from Parameters.times_and_contacts import TT as defaultTT
from Parameters.times_and_contacts import FF as defaultFF
from new_vax import rsv_eff_vax_rate
def x_to_params(x, pathogen, lockdown, option1, option2, fixed_params = None, import_multiplier=1e-9, end_date='2025-05-01', print_params=False):
    if fixed_params is None:
        from Parameters.census_population import AGING_RATE
        CONTACT_MATRIX = jnp.asarray(pd.read_csv('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', header=None).values)
        BIRTH_RATE = jnp.asarray(np.genfromtxt('Data/Processed/birth_rate_daily.csv', delimiter=','))
        if pathogen == "sim":
            _, _, IMPORT_STRENGTH, _, _ = pathogen_parameters("test", import_multiplier=import_multiplier, skip_incidence=True)
        else:
            REC_UP, REC_SAME, IMPORT_STRENGTH, _, _ = pathogen_parameters(pathogen, import_multiplier=import_multiplier, skip_incidence=True)
    else:
        FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX = fixed_params[:4]
        REC_UP, REC_SAME, IMPORT_STRENGTH = fixed_params[-3:]

    N_S, NAG = 3, 7
    EPOCH = pd.to_datetime('1970-01-01')
    END = pd.to_datetime(end_date)
    FULL_PERIOD = pd.date_range(start=EPOCH, end=END, freq='D')
    FULL_POINTS = jnp.array(date_to_t(FULL_PERIOD))

    n = 0
    if pathogen == "sim":
        REC_UP = jnp.array([x[n],x[n+1],0.0])
        REC_SAME = jnp.array([0.0,0.0,x[n+1]])
        n += 2
    BETA = x[n]
    SEASONALITY = x[n+1]
    OFFSET = x[n+2]
    MATERNAL_IMMUNITY = jnp.zeros(FULL_POINTS.shape)
    n += 3
    if "wane" in option1:
        WANE = jnp.array([0.0,x[n],x[n+1]])
        n += 2
    else:
        WANE = jnp.array([0.0,0.0,x[n]])
        n += 1
    if ("Influenza" in pathogen) and ("free" not in pathogen) and (option2 != 'nr'):
        srel, pobsrel = constrained_immunity(x[n],x[n+1],x[n+2])
        S_REL = srel
        n += 3
    elif pathogen == 'RSV':
        S_REL = jnp.array([1,x[n],x[n]*x[n+1]])
        pobsrel = jnp.array([1,0.46,0.31]) # Henderson 1979
        n += 2
    else:
        S_REL = jnp.array([1,x[n],x[n]*x[n+1]])
        pobsrel = jnp.array([1,x[n+2],x[n+2]*x[n+3]])
        n += 4
    if 'flexage' not in option2:
        P_OBS = x[n]*pobsrel
        n += 1
    else:
        P_OBS = pobsrel
    if "RSV" in pathogen:
        from new_vax import rsv_maternal_immunity
        MATERNAL_IMMUNITY = rsv_maternal_immunity(FULL_POINTS)
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
            FF = [x[n],]
            TT = x[n+1]
            contact_factor = 1 + FF[0]*cm.MOBILITY_CHANGE_JAX
            MOBILITY_CONTACT = jnp.ones(len(FULL_POINTS))
            MOBILITY_CONTACT = MOBILITY_CONTACT.at[cm.MOBILITY_START:cm.MOBILITY_END+1].set(contact_factor)
            MOBILITY_CONTACT = MOBILITY_CONTACT.at[cm.MOBILITY_END+1:].set(TT)
            RELATIVE_CONTACT = MOBILITY_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
            n += 2
        elif lockdown == 'Mobility2':
            FF = [x[n],x[n+1]]
            TT = x[n+2]
            contact_factor = 1 + FF[0]*cm.MOBILITY_CHANGE_JAX + FF[1]*cm.MOBILITY_CHANGE_JAX**2
            MOBILITY_CONTACT = jnp.ones(len(FULL_POINTS))
            MOBILITY_CONTACT = MOBILITY_CONTACT.at[cm.MOBILITY_START:cm.MOBILITY_END+1].set(contact_factor)
            MOBILITY_CONTACT = MOBILITY_CONTACT.at[cm.MOBILITY_END+1:].set(TT)
            RELATIVE_CONTACT = MOBILITY_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
            n += 3
        elif lockdown == "Taube":
            contact_factor = 2.63692872 + 1.96488539*cm.MOBILITY_CHANGE_JAX + -0.03890451*cm.MOBILITY_CHANGE_JAX**2
            MOBILITY_CONTACT = jnp.ones(len(FULL_POINTS))
            MOBILITY_CONTACT = MOBILITY_CONTACT.at[cm.MOBILITY_START:cm.MOBILITY_END+1].set(contact_factor)
            RELATIVE_CONTACT = MOBILITY_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
    if re.search(r'\d{6}',lockdown):
        with open("Data/Processed/DE_cm_opt_"+str(lockdown)+".pickle","rb") as f:
            opt = pickle.load(f)
        x_lockdown = opt.x
        TT = jnp.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x_lockdown[0]*365,date_to_t('2020-03-19')+(x_lockdown[0]+x_lockdown[1])*365,date_to_t('2020-03-19')+(x_lockdown[0]+x_lockdown[1]+x_lockdown[2])*365])
        # Fs - element 2 must be bigger than element 1, element 3 must be smaller than element 2, element 4 must be bigger than element 2
        F1 = x_lockdown[3] # value between 0 and 1 (first lockdown)
        F2 = F1 + x_lockdown[4] - F1*x_lockdown[4] # value between x_lockdown[3] and 1 (inter-lockdown)
        F3 = F2*x_lockdown[5] # value less than F2 (second lockdown)
        F4 = F2 + x_lockdown[6] - F2*x_lockdown[6] # value between F2 and 1 (post-lockdown)
        FF = jnp.array([1,F1,F2,F3,F4])
        PIECEWISE_CONTACT = jax.vmap(lambda t: cm.piecewise(t, TT, FF, steepness=0.2))(FULL_POINTS)
        RELATIVE_CONTACT = PIECEWISE_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
        if pathogen != "sim":
            n += 7
    elif 'pathogen' in option1:
        RELATIVE_CONTACT = fixed_params[9]
    if ('flexage' in option2) & ('dynamic' not in option1):
        OBS_AGE = jnp.array([x[n],x[n+1],x[n+2],x[n+3],x[n+4],x[n+5],x[n+6]])
    elif 'dynamic' in option1:
        OBS_AGE = fixed_params[7]
    if ("Influenza" in pathogen):
        from new_vax import flu_eff_vax_rate
        protection_param = S_REL*P_OBS
        max_eff = (protection_param[-2]-protection_param[-1])/protection_param[-2]
        VAX_RATE = flu_eff_vax_rate(FULL_POINTS, max_eff)
    elif ("RSV" in pathogen):
        protection_param = S_REL*P_OBS
        max_eff0 = 1 - protection_param[-1]
        max_eff1 = (protection_param[-2]-protection_param[-1])/protection_param[-2]
        VAX_RATE = rsv_eff_vax_rate(FULL_POINTS, max_eff0, max_eff1)
    else:
        VAX_RATE = jnp.zeros((len(FULL_POINTS),NAG))

    params = (FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX,
                BETA, WANE, S_REL, P_OBS, OBS_AGE, RELATIVE_CONTACT, VAX_RATE, MATERNAL_IMMUNITY,
                REC_UP, REC_SAME, IMPORT_STRENGTH)
    
    if print_params:
        param_names = ["BETA","WANE","SEASONALITY","OFFSET","S_REL","P_OBS","OBS_AGE","FF","TT"]
        for i in range(len(param_names)):
            if param_names[i] == "TT":
                print("TT: " + [t_to_date(t).strftime('%Y-%m-%d') for t in eval(param_names[i])].__str__())
            else:
                print(param_names[i]+": "+eval(param_names[i]).__str__())
    
    return params

def parameters_names_bounds(pathogen, lockdown, option1, option2):
    bounds_dict = {"WANE2": [0,1e-2], "SEASONALITY": [0,1], "OFFSET": [0,1], "BETA": [0,1]}

    if "wane" in option1:
        bounds_dict["WANE1"] = [0,1e-2]
    if option1 == "nb":
        bounds_dict["OVERDISPERSION"] = [-5,10]
    elif ("mimm" in option1) & ("maxmimm" not in option1):
        bounds_dict["MATERNAL_IMMUNITY"] = [0,1]
    if ("Influenza" in pathogen) and ("free" not in pathogen) and (option2 != "nr"):
        bounds_dict["EXTRA_IMMUNITY"] = [0,1]
        bounds_dict["FIRST_IMMUNITY"] = [0.1,1]
        bounds_dict["FIRST_DIS_INF_FACTOR"] = [0,1]
    elif pathogen == "RSV":
        bounds_dict["S_REL1"] = bounds_dict["S_REL2"] = [0.1,1]
    else:
        bounds_dict["S_REL1"] = bounds_dict["S_REL2"] = bounds_dict["D_REL1"] = bounds_dict["D_REL2"] = [0.1,1]
    if "dynamic" not in option1:
        if option2 == "flexage":
            bounds_dict["AGE_OBS_1"] = bounds_dict["AGE_OBS_2"] = bounds_dict["AGE_OBS_3"] = bounds_dict["AGE_OBS_4"] = bounds_dict["AGE_OBS_5"] = bounds_dict["AGE_OBS_6"] = bounds_dict["AGE_OBS_7"] = [0,0.005]
        else:
            # Extract decimal value from option2 (e.g., "flexagep01" -> 0.01, "flexagep5" -> 0.5)
            match = re.search(r'flexagep(\d+)', option2)
            if match:
                upper_bound = int(match.group(1)) / (10 ** len(match.group(1)))
            else:
                upper_bound = 0.01
            bounds_dict["AGE_OBS_1"] = bounds_dict["AGE_OBS_2"] = bounds_dict["AGE_OBS_3"] = bounds_dict["AGE_OBS_4"] = bounds_dict["AGE_OBS_5"] = bounds_dict["AGE_OBS_6"] = bounds_dict["AGE_OBS_7"] = [0, upper_bound]
    if "pathogen" not in option1:
        if lockdown == "Mobility":
            bounds_dict["F1"] = [0,2]
        elif lockdown == "Mobility2":
            bounds_dict["F1"] = bounds_dict["F2"] = [0,2]
        else:
            bounds_dict["DT1"] = bounds_dict["DT2"] = bounds_dict["DT3"] = bounds_dict["F1"] = bounds_dict["F2"] = bounds_dict["F3"] = bounds_dict["F4"] = [0,1]

    # reorder bounds_dict to match order in x
    bounds_dict = {key: bounds_dict[key] for key in ["BETA","SEASONALITY","OFFSET","WANE1","WANE2","IMPORT_RATE","EXTRA_IMMUNITY","FIRST_IMMUNITY","FIRST_DIS_INF_FACTOR","S_REL1","S_REL2","D_REL1","D_REL2","P_OBS","MATERNAL_IMMUNITY","DT1","DT2","DT3","F1","F2","F3","F4","OVERDISPERSION","AGE_OBS_YOUNG","AGE_OBS_OLD","AGE_OBS_YOUNG_OLD","AGE_OBS_MATERNAL","AGE_OBS_1","AGE_OBS_2","AGE_OBS_3","AGE_OBS_4","AGE_OBS_5","AGE_OBS_6","AGE_OBS_7"]\
        if key in bounds_dict.keys()}
    bounds = jnp.array(list(bounds_dict.values()))
    param_names = list(bounds_dict.keys())
    
    return param_names, bounds

def parameters_from_DE(pathogen, lockdown, option1, option2, seed, lockdown_x=None, import_multiplier=1e-9):
    if re.match(r'\d{6}',lockdown):
        with open("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+"FlexStepwise"+option1+option2+str(seed)+".pickle","rb") as f:
            opt = pickle.load(f)
    else:
        try:
            with open("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","rb") as f:
                opt = pickle.load(f)
        except FileNotFoundError:
            print("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle")
            print('No file found')
            sys.exit()
    x = opt.x
    
    from Parameters.census_population import AGING_RATE
    REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, tests = pathogen_parameters(pathogen, import_multiplier=import_multiplier)
    CONTACT_MATRIX = jnp.asarray(pd.read_csv('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', header=None).values)
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

# unified x from DE, i.e. x is same length regardless of model options. assume option2=flexage, lockdown=FlexStepwise
def consistent_x_from_DE(pathogen, option1, seed, NAG=7):
    with open("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+"FlexStepwise"+option1+"flexage"+str(seed)+".pickle","rb") as f:
        opt = pickle.load(f)
    REC_UP, _, _, _ = pathogen_parameters(pathogen, import_multiplier=1e-9, skip_incidence=True)
    x_DE = opt.x
    x_consistent = jnp.zeros(19)
    x_consistent = x_consistent.at[0:2].set([REC_UP[0], REC_UP[1]]) # REC
    x_consistent = x_consistent.at[2:5].set(x_DE[0:3]) # BETA, SEASONALITY, OFFSET
    n = 3
    if "wane" in option1:
        x_consistent = x_consistent.at[5:7].set(x_DE[3:5]) # WANE1, WANE2
        n += 2
    else:
        x_consistent = x_consistent.at[6].set(x_DE[3]) # WANE2
        n += 1
    if ("Influenza" in pathogen) and ("free" not in pathogen):
        srel, pobsrel = constrained_immunity(x_DE[n],x_DE[n+1],x_DE[n+2])
        S_REL1 = srel[1]
        S_REL2 = srel[2]/srel[1]
        D_REL1 = pobsrel[1]
        D_REL2 = pobsrel[2]/pobsrel[1]
        x_consistent = x_consistent.at[7:11].set(jnp.array([S_REL1,S_REL2,D_REL1,D_REL2]))
        n += 3
    elif pathogen == "RSV":
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
    x_consistent = x_consistent.at[12:19].set(x_DE[-NAG:]) # AGE_OBS_1 to AGE_OBS_7
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

def susceptibility(solution,params,N_C=2):
    """
    Generate susceptibility by age group from ODE solutions
    """
    NAG, N_S = 7, 3
    FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX,\
    BETA, WANE, S_REL, P_OBS, OBS_AGE, RELATIVE_CONTACT, VAX_RATE, MATERNAL_IMMUNITY,\
    REC_UP, REC_SAME, IMPORT_STRENGTH = params
    sus = np.zeros((len(solution.ts),NAG))
    for i_t,t in enumerate(solution.ts):
        for i in range(N_S):
            sus[i_t,:] += S_REL[i]*solution.ys.T[1+N_C*i*NAG:1+(N_C*i+1)*NAG,i_t]
    return(sus)

if __name__ == "__main__":
    # measure length of incidence vector for rsv
    params = x_to_params(np.array([0.03,0.1,0.5,0.001,0.0001,1e-9,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.01,0.5,0.5,0.5,0.5,0.01,0.5]),'RSV','FlexStepwise','NA','flexage')
    print(len(params[0]))