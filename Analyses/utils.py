import numpy as np
import pandas as pd
import re

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
    return (np.tanh(number)+1)/2

def p_to_real(probability):
    """
    Map (0,1) interval to real number
    """
    return np.arctanh(2*probability-1)

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

def scalars_to_params(scalar_values_dict, NAG=7, N_S=3, N_C=2):
    param_dict = {}
    for name in scalar_values_dict.keys():
        if name in ["NAG","N_S","BIRTH_RATE","S_VAX","ACOV","BCOV","T_VAX","IMPORT_RATE","BETA","SEASONALITY","OFFSET"]:
            param_dict[name] = scalar_values_dict[name]
        elif name in ["WANE","REC_UP","REC_SAME","P_OBS"]:
            param_dict[name] = scalar_values_dict[name]*np.ones(N_S)
        elif name in ["S_REL","I_REL","S_AGE","OBS_AGE"]:
            param_dict[name] = increment_to_vec(scalar_values_dict[name],N_S)
        elif name == "ACQUIRED_IMMUNITY":
            param_dict["S_REL"] = increment_to_vec(scalar_values_dict[name],N_S)
    if any([re.search(r"S_REL\d+",name) for name in scalar_values_dict.keys()]):
        param_dict["S_REL"] = np.concatenate(np.ones(1),np.array([scalar_values_dict["S_REL"+str(i)] for i in range(1,N_S)]))
    if any([re.search(r"ACQUIRED_IMMUNITY\d+",name) for name in scalar_values_dict.keys()]):]):
        param_dict["S_REL"] = np.concatenate(np.ones(1),np.array([1-scalar_values_dict["ACQUIRED_IMMUNITY"+str(i)] for i in range(1,N_S)]))
    return param_dict

def params_to_scalars(param_dict,scalar_names):
    scalar_dict = {}
    for name in scalar_names:
        if name in ["NAG","N_S","BIRTH_RATE","S_VAX","ACOV","BCOV","T_VAX","IMPORT_RATE","BETA","SEASONALITY","OFFSET"]:
            scalar_dict[name] = param_dict[name]
        elif name in ["WANE","REC_UP","REC_SAME","P_OBS"]:
            scalar_dict[name] = param_dict[name][0]
        elif name in ["S_REL","I_REL","S_AGE","OBS_AGE"]:
            scalar_dict[name] = 1-param_dict[name][1]
        elif name == "ACQUIRED_IMMUNITY":
            scalar_dict[name] = 1-param_dict["S_REL"][1]
    for i in range(1,param_dict["N_S"]):
        if any([re.search(r"S_REL\d+",name) for name in scalar_names]):
            scalar_dict["S_REL"+str(i)] = 1-param_dict["S_REL"][i]
        elif any([re.search(r"ACQUIRED_IMMUNITY\d+",name) for name in scalar_names]):
            scalar_dict["ACQUIRED_IMMUNITY"+str(i)] = 1-param_dict["S_REL"][i]
    return scalar_dict


####### Generating interesting quantities from ODE results #######

def observations(result,params,OBS_AGE,incidence=False,cap=False,N_C=2):
    """
    Generate observed cases or incidence from ODE results
    """
    NAG, N_S, AGING_RATE, births, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, arrivals, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact = params.values()
    obs = np.zeros((len(result.t),NAG))
    pop_size = np.sum(result.y,axis=0)
    for i_t,t in enumerate(result.t):
        foi = BETA*np.dot(contact(t,SEASONALITY,OFFSET),np.sum((np.array([result.y[(N_C*j+2)*NAG:(N_C*j+3)*NAG,i_t] for j in range(N_S)])*I_REL),axis=0))/pop_size[i_t]
        for i in range(N_S):
            if cap:
                class_foi = np.minimum(1,S_REL[i]*foi)
            else:
                class_foi = S_REL[i]*foi
            # S_REL*foi tells you what proportion of the population gets infected, the maximum is all of them
            obs[i_t,:] += OBS_AGE*P_OBS[i]*class_foi*result.y[(N_C*i+1)*NAG:(N_C*i+2)*NAG,i_t]
    if incidence:
        obs = np.sum(obs,axis=1)/pop_size
    return(30.44*obs)

def infections_by_age(result,params,N_C=2):
    """
    Generate infections attributable to each age group from ODE results
    """
    NAG, N_S, AGING_RATE, births, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, arrivals, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact = params.values()
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
    NAG, N_S, AGING_RATE, births, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, arrivals, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact = params.values()
    sus = np.zeros((len(result.t),NAG))
    for i_t,t in enumerate(result.t):
        for i in range(N_S):
            sus[i_t,:] += S_REL[i]*S_AGE*result.y[(N_C*i+1)*NAG:(N_C*i+2)*NAG,i_t]
    return(sus)