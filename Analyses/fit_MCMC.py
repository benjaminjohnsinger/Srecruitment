import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from SISn_ODEs import single_pathogen_deltas as sis_deltas
from plotting import observations
from utils import *

def SIS_likelihood(data, params, POINTS, STATE0, OBS_AGE, start_t=date_to_t(pd.to_datetime('1970-01-01'))):
    result = sp.integrate.solve_ivp(sis_deltas,(start_t,POINTS[-1]),STATE0,args=(params,),t_eval=POINTS,method='RK45')
    trajectory = np.sum(observations(result,params,OBS_AGE,incidence=False),axis=1)
    log_likelihood = 0
    for i in range(len(POINTS)):
        log_likelihood += sp.stats.poisson.logpmf(int(data[i]),int(trajectory[i]))
    return log_likelihood

def mcmc(data, init_params, POINTS, STATE0, OBS_AGE, likelihood, vary_params, priors_tanh, proposal_widths, n_iter):
    n_v = len(vary_params)
    acceptance = np.zeros((int(np.ceil(n_iter/n_v)),n_v))
    param_trajectory = np.zeros((n_iter+1,n_v))
    for i in range(n_v):
        if isinstance(init_params[vary_params[i]],np.ndarray):
            param_trajectory[0,i] = 1-init_params[vary_params[i]][1]
        else:
            param_trajectory[0,i] = init_params[vary_params[i]]
    current_params = init_params.copy()
    log_likelihood_current = likelihood(data, current_params, POINTS, STATE0, OBS_AGE) + np.sum([priors_tanh[vary_params[i]].logpdf(p_to_real(to_increment(current_params[vary_params[i]]))) for i in range(n_v)])

    for i in range(n_iter):
        if i % 100 == 0:
            print(i)
        proposal_params = current_params.copy()
        vary_param = vary_params[i%n_v]
        if isinstance(current_params[vary_param],np.ndarray):
            prm_curr = to_increment(current_params[vary_param])
            prm_prop = real_to_p(p_to_real(prm_curr) + np.random.normal(0,proposal_widths[vary_param]))
            proposal_params[vary_param] = increment_to_vec(prm_prop,len(current_params[vary_param]))
        else:
            prm_prop = real_to_p(p_to_real(current_params[vary_param]) + np.random.normal(0,proposal_widths[vary_param]))
            proposal_params[vary_param] = prm_prop
        log_likelihood_proposal = likelihood(data, proposal_params, POINTS, STATE0, OBS_AGE) + np.sum([priors_tanh[vary_params[i]].logpdf(p_to_real(to_increment(proposal_params[vary_params[i]]))) for i in range(n_v)])
        log_likelihood_diff = log_likelihood_proposal - log_likelihood_current
        param_trajectory[i+1] = param_trajectory[i]
        if log_likelihood_diff > 0 or np.log(np.random.rand()) < log_likelihood_diff:
            current_params = proposal_params
            log_likelihood_current = log_likelihood_proposal
            acceptance[i//n_v,i%n_v] = 1
            param_trajectory[i+1,i%n_v] = prm_prop
    
    return param_trajectory, np.mean(acceptance,axis=0)

# ###### Copilot's guess at a MCMC fitting function ######
# def fit_MCMC(data, model, params, priors, n_iter, n_chains, n_burn, n_thin, save_file):
#     """
#     Fit a model to data using MCMC with Metropolis-Hastings algorithm.
    
#     Parameters
#     ----------
#     data : dict
#         Dictionary with data to fit the model. The keys are the names of the data and the values are the data.
#     model : function
#         Function that computes the log-likelihood of the model given the data and parameters.
#     params : dict
#         Dictionary with the initial values of the parameters to fit. The keys are the names of the parameters and the values are the initial values.
#     priors : dict
#         Dictionary with the prior distributions of the parameters. The keys are the names of the parameters and the values are the prior distributions.
#     n_iter : int
#         Number of iterations of the MCMC algorithm.
#     n_chains : int
#         Number of chains of the MCMC algorithm.
#     n_burn : int
#         Number of burn-in iterations of the MCMC algorithm.
#     n_thin : int
#         Thinning factor of the MCMC algorithm.
#     save_file : str
#         Name of the file to save the MCMC results.
#     """
#     # Initialize parameters
#     params = {key: np.array([params[key] for _ in range(n_chains)]) for key in params}
#     # Initialize log-likelihood
#     log_likelihood = {key: np.zeros(n_chains) for key in data}
#     # Initialize log-posterior
#     log_posterior = {key: np.zeros(n_chains) for key in data}
#     # Initialize acceptance rate
#     acceptance_rate = {key: 0 for key in data}
#     # Initialize chains
#     chains = {key: np.zeros((n_iter-n_burn)//n_thin, n_chains) for key in data}
#     # Initialize current values
#     current_values = {key: params[key].copy() for key in params}
#     # Initialize proposal values
#     proposal_values = {key: np.zeros(n_chains) for key in params}
    
#     # Run MCMC
#     for i in range(n_iter):
#         # Update proposal values
#         for key in params:
#             proposal_values[key] = priors[key].rvs(n_chains)
#         # Compute log-likelihood
#         for key in data
#             log_likelihood[key] = model(data[key], current_values)
#         # Compute log-prior
#         log_prior = {key: priors[key].logpdf(proposal_values[key]) - priors[key].logpdf(current_values[key]) for key in params}
#         # Compute log-posterior
#         for key in data:
#             log_posterior[key] = log_likelihood[key] + log_prior
#         # Compute acceptance probability
#         acceptance_probability = {key: np.exp(log_posterior[key] - log_posterior[key].max()) for key in data}
#         # Accept or reject proposal
#         for key in data:
#             accept = acceptance_probability[key] > np.random.rand(n_chains)
#             current_values[key][accept] = proposal_values[key][accept]
#             acceptance_rate[key] += accept.mean()
#         # Save chains
#         if i >= n_burn and i % n_thin == 0:
#             for key in data:
#                 chains[key][(i-n_burn)//n_thin] = current_values[key]

#     # Save results
#     with open(save_file, 'wb') as f:
#         pickle.dump(chains, f)
#         pickle.dump(acceptance_rate, f)
#         pickle.dump(log_likelihood, f)
#         pickle.dump(log_posterior, f)
#         pickle.dump(params, f)
#         pickle.dump(priors, f)
#         pickle.dump(data, f)
#         pickle.dump(model, f)
#         pickle.dump(n_iter, f)
#         pickle.dump(n_chains, f)
#         pickle.dump(n_burn, f)
#         pickle.dump(n_thin, f)
#     return chains, acceptance_rate, log_likelihood, log_posterior, params, priors, data, model, n_iter, n_chains, n_burn, n_thin
