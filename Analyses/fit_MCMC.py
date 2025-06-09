import jax.numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from SISn_ODEs import single_pathogen_deltas as sis_deltas
N_C = 2
from utils import observations, real_to_p, p_to_real, scalars_to_params, age_detection, date_to_t
import time
import types

## POINTS must start (at least) len(p_time_to_obs) days before the first observation to avoid issues from np.roll behaviour
def SIS_likelihood(data, params, POINTS, STATE0, OBS_AGE, p_time_to_obs, age=True, incidence=True, start_t=date_to_t(pd.to_datetime('1970-01-01')), overdispersion=False, result=None):
    # run simulation
    if result is None:
        result = sp.integrate.solve_ivp(sis_deltas,(start_t,POINTS[-1]),STATE0,args=params.values(),t_eval=POINTS,method='RK45')
        ry = result.y
    else:
        if type(result) == dict:
            ry = result["y"]

    # convert into observed cases
    trajectory = observations(result,params,OBS_AGE,incidence=False,time_conversion=1)
    if not age:
        trajectory = np.sum(trajectory,axis=1)

    # format data into cases, rescaled appropriately by population age distribution
    if incidence:
        if age:
            NAG = params["NAG"]
            N_S = params["N_S"]
            cases = np.round(data*np.array([np.sum(ry[range(i,(N_S*N_C+1)*NAG,NAG),len(p_time_to_obs):],axis=0) for i in range(NAG)]).T)
        else:
            cases = data*np.sum(ry,axis=0)
    else:
        cases = data.copy()

    # the expected observations for a given date are the observations on each day i days prvious multiplied by the probability of detection i days after infection
    expected_obs = np.sum([np.roll(trajectory,i,axis=0)*p_time_to_obs[i] for i in range(len(p_time_to_obs))],axis=0)
    # cut off the first few days of the trajectory since they are not used in the likelihood (and the roll function is wrapping around)
    expected_obs = expected_obs[-len(cases):]

    # calculate the log likelihood
    if overdispersion:
        p = overdispersion/(overdispersion+expected_obs)
        likelihood = sp.stats.nbinom.logpmf(cases,overdispersion,p).sum()
    else:
        likelihood = sp.stats.poisson.logpmf(cases,expected_obs).sum()
    return likelihood


## OBS_AGE parameters must be last three in initial_scalars, POINTS must start (at least) len(p_time_to_obs) days before the first observation
def mcmc(data, init_params, POINTS, STATE0, OBS_AGE, likelihood, p_time_to_obs, variables, initial_scalars, log_priors, proposal_cov, n_iter, age=False, incidence=False,n_messages=20):
    n_v = len(variables)
    NAG = init_params["NAG"]
    acceptance = np.zeros(n_iter)
    param_trajectory = np.zeros((n_iter+1,n_v))
    param_trajectory[0] = initial_scalars
    likelihoods = np.zeros(n_iter)
    current_params = init_params.copy()
    log_likelihood_priors = log_priors(initial_scalars)
    log_likelihood_current = likelihood(data, current_params, POINTS, STATE0, OBS_AGE, p_time_to_obs, age=age, incidence=incidence) + log_likelihood_priors
    start = time.time()
    for i in range(n_iter):
        if i % (n_iter//n_messages) == 0:
            print(f"Progress: {i/n_iter*100:.0f}%, acceptance rate: {np.mean(acceptance[i-(n_iter//n_messages):i])*100:.4f}%, time elapsed: {time.time()-start:.2f}s")
        proposal_params = current_params.copy()
        scalars = param_trajectory[i]
        scalars = real_to_p(p_to_real(scalars) + np.random.multivariate_normal(np.zeros(n_v),proposal_cov))
        scalar_dict = {variables[j]:scalars[j] for j in range(n_v)}
        proposal_params = scalars_to_params(scalar_dict,proposal_params)
        if age:
            proposal_OBS_AGE = age_detection(NAG,*scalars[-3:])
        else:
            proposal_OBS_AGE = OBS_AGE
        log_likelihood_priors = log_priors(scalars)
        log_likelihood_proposal = likelihood(data, proposal_params, POINTS, STATE0, proposal_OBS_AGE, p_time_to_obs, age=age, incidence=incidence) + log_likelihood_priors
        log_likelihood_diff = log_likelihood_proposal - log_likelihood_current
        param_trajectory[i+1] = param_trajectory[i]
        if log_likelihood_diff > 0 or np.log(np.random.rand()) < log_likelihood_diff:
            current_params = proposal_params
            log_likelihood_current = log_likelihood_proposal
            acceptance[i] = 1
            param_trajectory[i+1,:] = scalars
        likelihoods[i] = log_likelihood_current
    
    return param_trajectory, np.mean(acceptance), likelihoods

## OBS_AGE parameters must be last three in initial_scalars, POINTS must start (at least) len(p_time_to_obs) days before the first observation
def mcmc_adaptive(data, init_params, POINTS, STATE0, OBS_AGE, likelihood, p_time_to_obs, variables, initial_scalars, log_priors, step_var, n_iter, age=False, incidence=False,n_messages=20):
    n_v = len(variables)
    NAG = init_params["NAG"]
    acceptance = np.zeros((n_iter,n_v))
    param_trajectory = np.zeros((n_iter+1,n_v))
    param_trajectory[0] = initial_scalars
    likelihoods = np.zeros(n_iter)
    current_params = init_params.copy()
    log_likelihood_priors = log_priors(initial_scalars)
    log_likelihood_current = likelihood(data, current_params, POINTS, STATE0, OBS_AGE, p_time_to_obs, age=age, incidence=incidence) + log_likelihood_priors
    start = time.time()
    for i in range(n_iter):
        if i % (n_iter//n_messages) == 0:
            # print(f"Progress: {i/n_iter*100:.0f}%, acceptance rate: {np.mean(acceptance[i-(n_iter//n_messages):i])*100:.4f}%, variance: {step_var}, time elapsed: {time.time()-start:.2f}s")
            # print each step_var value to 2 significant figures
            print(f"Progress: {i/n_iter*100:.0f}%, acceptance rate: {[f'{x:.4f}' for x in np.mean(acceptance[i-(n_iter//n_messages):i],axis=0)*100]}%, variance: {[f'{x:.2g}' for x in step_var]}, time elapsed: {time.time()-start:.2f}s")
        if i > 20:
            acceptance_rate = np.mean(acceptance[i-20:i],axis=0)
            update = (acceptance_rate - 0.23)/np.log(i+1)
            step_var = step_var*np.exp(update)
        random_variable_order = np.random.permutation(n_v)
        for i_v in random_variable_order:
            proposal_params = current_params.copy()
            scalars = param_trajectory[i]
            scalars[i_v] = real_to_p(p_to_real(scalars[i_v]) + np.random.normal(0,step_var[i_v]))
            scalar_dict = {variables[j]:scalars[j] for j in range(n_v)}
            proposal_params = scalars_to_params(scalar_dict,proposal_params)
            if age:
                proposal_OBS_AGE = age_detection(NAG,*scalars[-3:])
            else:
                proposal_OBS_AGE = OBS_AGE
            log_likelihood_priors = log_priors(scalars)
            log_likelihood_proposal = likelihood(data, proposal_params, POINTS, STATE0, proposal_OBS_AGE, p_time_to_obs, age=age, incidence=incidence) + log_likelihood_priors
            log_likelihood_diff = log_likelihood_proposal - log_likelihood_current
            param_trajectory[i+1] = param_trajectory[i]
            if log_likelihood_diff > 0 or np.log(np.random.rand()) < log_likelihood_diff:
                current_params = proposal_params
                log_likelihood_current = log_likelihood_proposal
                acceptance[i,i_v] = 1
                param_trajectory[i+1,:] = scalars
            likelihoods[i] = log_likelihood_current
    
    return param_trajectory, np.mean(acceptance,axis=0), likelihoods