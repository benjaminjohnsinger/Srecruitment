import numpy as np
import pandas as pd
import re
from numba import jit
import contact_model as cm

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

def constrained_immunity(extra_immunity,first_immunity,first_dis_inf_factor,DIS_INF_RATIO=0.674,MIN_EFF=0.39,CHILD_EFF_RATIO=1.54):
    """
    Convert free parameters between 0 and 1 to immunity parameters constrained by flu vaccine data
    """
    ESP = MIN_EFF + extra_immunity*(1/CHILD_EFF_RATIO - MIN_EFF)
    factor = ((1 - DIS_INF_RATIO) + np.sqrt(1 + DIS_INF_RATIO**2 + 2*DIS_INF_RATIO*(1-2*ESP)))/2
    S1 = ((1-CHILD_EFF_RATIO*ESP)/(1-ESP) + first_immunity*(1-(1-CHILD_EFF_RATIO*ESP)/(1-ESP)))**first_dis_inf_factor
    S2 = S1*(1-ESP)/factor
    D1 = ((1-CHILD_EFF_RATIO*ESP)/(1-ESP) + first_immunity*(1-(1-CHILD_EFF_RATIO*ESP)/(1-ESP)))**(1-first_dis_inf_factor)
    D2 = D1*factor
    return np.array([1,S1,S2]), np.array([1,D1,D2])

def scalars_to_params(scalar_values_dict, params, NAG=7, N_S=3, N_C=2):
    for name in scalar_values_dict.keys():
        if name in ["NAG","N_S","BIRTH_RATE","S_VAX","ACOV","BCOV","T_VAX","IMPORT_RATE","BETA","SEASONALITY","OFFSET"]:
            params[name] = scalar_values_dict[name]
        elif name == "WANE":
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
            CONTACT = np.genfromtxt('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', dtype=np.float64)
            @jit
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
        elif name == "WANE":
            scalar_dict[name] = param_dict[name][1]
        elif name in ["WANE","REC_UP","REC_SAME","P_OBS"]:
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

def pathogen_parameters(pathogen, lockdown=None, CONTACT=None):
    from Parameters.census_population import AGING_RATE
    from demography import birth_rate
    from vaccination import birth_vax, all_vax, flu_rate
    from mobility_and_import import arrivals
    if lockdown == 'Mobility':
        @jit
        def contact(t,seasonality,offset):
            return cm.google_prestige_work(t)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
    elif lockdown == 'X':
        @jit
        def contact(t,seasonality,offset):
            return CONTACT*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))
    elif lockdown == 'YoungEarly':
        @jit
        def contact(t,seasonality,offset):
            cont = cm.google_prestige_work(t)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
            if t > 18702: # 2021-03-16 where majority of schoools returned to in-person according to burbio
                cont[0:4] = (1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT[0:4]
            return cont
    else:
        Ts = np.array([date_to_t('1970-01-01'),
        date_to_t('2020-03-19'), # Newsom announces stay-at-home order
        date_to_t('2021-04-27'), # CDC amends mask guidance to allow vaccinated individuals to go maskless
        date_to_t('2021-12-15'), # CDC reinstates mask guidance
        date_to_t('2022-03-01')]) # End of mask mandate in California
        # date_to_t('2021-12-20')])
        Fs = np.array([1,0.2,1,0.2,1]) # 0.2 minimum relative contact rate between COMIX and POLYMOD
        @jit
        def contact(t,seasonality,offset):
            return cm.piecewise(t,Ts,Fs)*(1+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
    if pathogen == 'RSV':
        from Parameters.RSV import NAG, N_S, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, S_VAX, ACOV, BCOV, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET
        # Parameters for the ODE
        params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
        'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
        'contact': contact}
        p_time_to_obs = np.genfromtxt("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
        incidence = pd.read_csv("Data/Processed/KPSC_RSV_incidence_age_daily.csv",index_col=0)
    elif pathogen == 'InfluenzaA':
        from Parameters.InfluenzaA import NAG, N_S, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, S_VAX, BCOV, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET
        # Parameters for the ODE
        params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': flu_rate, 'BCOV': BCOV,
        'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
        'contact': contact}
        p_time_to_obs = np.genfromtxt("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
        incidence = pd.read_csv("Data/Processed/KPSC_Influenza_A_incidence_age_daily.csv",index_col=0)
    elif pathogen == 'InfluenzaB':
        from Parameters.InfluenzaB import NAG, N_S, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, S_VAX, BCOV, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET
        # Parameters for the ODE
        params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': flu_rate, 'BCOV': BCOV,
        'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
        'contact': contact}
        p_time_to_obs = np.genfromtxt("Data/Processed/Influenza_B_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
        incidence = pd.read_csv("Data/Processed/KPSC_Influenza_B_incidence_age_daily.csv",index_col=0)
    elif pathogen == 'Parainfluenza3':
        from Parameters.Parainfluenza3 import NAG, N_S, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, S_VAX, ACOV, BCOV, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET
        # Parameters for the ODE
        params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
        'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
        'contact': contact}
        p_time_to_obs = np.genfromtxt("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
        incidence = pd.read_csv("Data/Processed/KPSC_Parainfluenza3_incidence_age_daily.csv",index_col=0)
    elif pathogen == 'Adenovirus':
        from Parameters.Adenovirus import NAG, N_S, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, S_VAX, ACOV, BCOV, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET
        # Parameters for the ODE
        params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
        'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
        'contact': contact}
        p_time_to_obs = np.genfromtxt("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
        incidence = pd.read_csv("Data/Processed/KPSC_Adenovirus_incidence_age_daily.csv",index_col=0)
    elif pathogen == 'Metapneumovirus':
        from Parameters.Metapneumovirus import NAG, N_S, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, S_VAX, ACOV, BCOV, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET
        # Parameters for the ODE
        params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
        'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
        'contact': contact}
        p_time_to_obs = np.genfromtxt("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
        incidence = pd.read_csv("Data/Processed/KPSC_Metapneumovirus_incidence_age_daily.csv",index_col=0)
    return params, p_time_to_obs, incidence


####### Generating interesting quantities from ODE results #######

def observations(result,params,OBS_AGE,incidence=False,cap=False,N_C=2,time_conversion=30.44):
    """
    Generate observed cases or incidence from ODE results
    """
    NAG, N_S, BETA, contact, SEASONALITY, OFFSET, S_REL, I_REL, P_OBS = params["NAG"], params["N_S"], params["BETA"], params["contact"], params["SEASONALITY"], params["OFFSET"], params["S_REL"], params["I_REL"], params["P_OBS"]
    if type(result) == dict:
        ry = result["y"]
        rt = result["t"]
    else:
        ry = result.y
        rt = result.t
    obs = np.zeros((len(rt),NAG))
    pop_size = np.sum(ry,axis=0)
    for i_t,t in enumerate(rt):
        foi = BETA*np.dot(contact(t,SEASONALITY,OFFSET),np.sum((np.array([ry[(N_C*j+2)*NAG:(N_C*j+3)*NAG,i_t] for j in range(N_S)])*I_REL),axis=0))/pop_size[i_t]
        for i in range(N_S):
            if cap:
                class_foi = np.minimum(1,S_REL[i]*foi)
            else:
                class_foi = S_REL[i]*foi
            # S_REL*foi tells you what proportion of the population gets infected, the maximum is all of them
            obs[i_t,:] += OBS_AGE*P_OBS[i]*class_foi*ry[(N_C*i+1)*NAG:(N_C*i+2)*NAG,i_t]
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

def susceptibility(result,params,N_C=2):
    """
    Generate susceptibility by age group from ODE results
    """
    NAG, N_S, S_REL, S_AGE = params["NAG"], params["N_S"], params["S_REL"], params["S_AGE"]
    sus = np.zeros((len(result.t),NAG))
    for i_t,t in enumerate(result.t):
        for i in range(N_S):
            sus[i_t,:] += S_REL[i]*S_AGE*result.y[(N_C*i+1)*NAG:(N_C*i+2)*NAG,i_t]
    return(sus)