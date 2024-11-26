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

def mcmc(data, init_params, POINTS, STATE0, OBS_AGE, likelihood, variables, priors_logit, proposal_cov, n_iter, age=False, incidence=False):
    n_v = len(variables)
    acceptance = np.zeros(n_iter)
    param_trajectory = np.zeros((n_iter+1,n_v))
    param_trajectory[0] = list(params_to_scalars(init_params,variables).values())
    current_params = init_params.copy()
    log_likelihood_priors = np.sum([priors_logit[variable].logpdf(p_to_real(params_to_scalars(current_params,variables)[variable])) for variable in variables])
    log_likelihood_current = likelihood(data, current_params, POINTS, STATE0, OBS_AGE, age=age, incidence=incidence) + log_likelihood_priors
    start = time.time()
    for i in range(n_iter):
        if i % (n_iter//20) == 0:
            print(f"Progress: {i/n_iter*100:.0f}%, acceptance rate: {np.mean(acceptance[:i])*100:.4f}%, time elapsed: {time.time()-start:.2f}s")
        proposal_params = current_params.copy()
        scalars = np.array(list(params_to_scalars(proposal_params,variables).values()))
        scalars = real_to_p(p_to_real(scalars) + np.random.multivariate_normal(np.zeros(n_v),proposal_cov))
        scalar_dict = {variables[j]:scalars[j] for j in range(n_v)}
        proposal_params = scalars_to_params(scalar_dict,proposal_params)
        log_likelihood_priors = np.sum([priors_logit[variable].logpdf(p_to_real(params_to_scalars(proposal_params,variables)[variable])) for variable in variables])
        log_likelihood_proposal = likelihood(data, proposal_params, POINTS, STATE0, OBS_AGE, age=age, incidence=incidence) + log_likelihood_priors
        log_likelihood_diff = log_likelihood_proposal - log_likelihood_current
        param_trajectory[i+1] = param_trajectory[i]
        if log_likelihood_diff > 0 or np.log(np.random.rand()) < log_likelihood_diff:
            current_params = proposal_params
            log_likelihood_current = log_likelihood_proposal
            acceptance[i] = 1
            param_trajectory[i+1,:] = scalars
    
    return param_trajectory, np.mean(acceptance)