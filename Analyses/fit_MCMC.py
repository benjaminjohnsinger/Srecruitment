import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from SISn_ODEs import single_pathogen_deltas as sis_deltas
N_C = 2
from plotting import observations
from utils import *
import time
import types

def SIS_likelihood(data, params, POINTS, STATE0, OBS_AGE, age=False, incidence=False, start_t=date_to_t(pd.to_datetime('1970-01-01'))):
    result = sp.integrate.solve_ivp(sis_deltas,(start_t,POINTS[-1]),STATE0,args=params.values(),t_eval=POINTS,method='RK45')
    NAG = params["NAG"]
    N_S = params["N_S"]
    if incidence:
        if age:
            cases = data*np.array([np.sum(result.y[range(i,(N_S*N_C+1)*NAG,NAG),:],axis=0) for i in range(NAG)]).T
        else:
            cases = data*np.sum(result.y,axis=0)
    else:
        cases = data.copy()
    trajectory = observations(result,params,OBS_AGE,incidence=False)
    if not age:
        trajectory = np.sum(trajectory,axis=1)
    log_likelihood = 0
    if age:
        for i in range(len(POINTS)):
            for j in range(params["NAG"]):
                # If less than one case predicted but more than zero observed, round up predicted case count
                if trajectory[i,j] < 1 and cases[i,j] >= 1:
                    log_likelihood += sp.stats.poisson.logpmf(int(cases[i,j]),1)
                # otherwise round down both numbers
                else:
                    log_likelihood += sp.stats.poisson.logpmf(int(cases[i,j]),int(trajectory[i,j]))
    else:
        for i in range(len(POINTS)):
            log_likelihood += sp.stats.poisson.logpmf(int(cases[i]),int(trajectory[i]))
    return log_likelihood

def mcmc(data, init_params, POINTS, STATE0, OBS_AGE, likelihood, variables, priors_tanh, proposal_widths, n_iter, age=False, incidence=False):
    n_v = len(variables)
    acceptance = np.zeros((int(np.ceil(n_iter/n_v)),n_v))
    param_trajectory = np.zeros((n_iter+1,n_v))
    param_trajectory[0] = [params_to_scalars(init_params,variables)[variable] for variable in variables]
    # for i in range(n_v):
    #     if isinstance(init_params[variables[i]],np.ndarray):
    #         param_trajectory[0,i] = 1-init_params[variables[i]][1]
    #     else:
    #         param_trajectory[0,i] = init_params[variables[i]]
    current_params = init_params.copy()
    log_likelihood_priors = np.sum([priors_tanh[variable].logpdf(p_to_real(params_to_scalars(current_params,variables)[variable])) for variable in variables])
    log_likelihood_current = likelihood(data, current_params, POINTS, STATE0, OBS_AGE, age=age, incidence=incidence) + log_likelihood_priors

    for i in range(n_iter):
        if i % 100 == 0:
            print(i)
        proposal_params = current_params.copy()
        vary = variables[i%n_v]
        scalars = params_to_scalars(proposal_params,variables)
        scalars[vary] = real_to_p(p_to_real(scalars[vary]) + np.random.normal(0,proposal_widths[vary]))
        proposal_params = scalars_to_params(scalars,proposal_params)
        log_likelihood_priors = np.sum([priors_tanh[variable].logpdf(p_to_real(params_to_scalars(proposal_params,variables)[variable])) for variable in variables])
        log_likelihood_proposal = likelihood(data, proposal_params, POINTS, STATE0, OBS_AGE, age=age, incidence=incidence) + log_likelihood_priors
        log_likelihood_diff = log_likelihood_proposal - log_likelihood_current
        param_trajectory[i+1] = param_trajectory[i]
        if log_likelihood_diff > 0 or np.log(np.random.rand()) < log_likelihood_diff:
            current_params = proposal_params
            log_likelihood_current = log_likelihood_proposal
            acceptance[i//n_v,i%n_v] = 1
            param_trajectory[i+1,i%n_v] = scalars[vary]
    
    return param_trajectory, np.mean(acceptance,axis=0)