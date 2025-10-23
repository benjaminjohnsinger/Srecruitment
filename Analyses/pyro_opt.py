## Script to fit SIS model to data using JAX and NumPyro
## BJS June 2025

import jax
import jax.numpy as jnp

import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle

from utils import *

## Parameters
# from vaccination import flu_rate
import contact_model as cm
from Parameters.census_population import AGING_RATE, CENSUS_AGE_POP
from Parameters.times_and_contacts import *

# import all using jnp
CONTACT_MATRIX = jnp.asarray(pd.read_csv('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', header=None).values)
BIRTH_RATE = jnp.asarray(np.genfromtxt('Data/Processed/birth_rate_daily.csv', delimiter=','))
ARRIVALS = jnp.asarray(np.genfromtxt('Data/Processed/arrivals_daily.csv', delimiter=','))
POSITIVITY = jnp.asarray(np.genfromtxt('Data/Processed/RSV_positivity_daily.csv', delimiter=','))


# example parameters for testing
NAG = 7
N_S = 3
DAYS = 19632
# BIRTH_RATE = jnp.ones(DAYS)*jnp.mean(BIRTH_RATE)
# ARRIVALS = jnp.ones(DAYS)*jnp.mean(ARRIVALS)
# POSITIVITY = jnp.ones(DAYS)*jnp.mean(POSITIVITY)
BETA = 0.43911709767941026
WANE = jnp.array([0.0, 0.0, 0.00328313])
S_REL = jnp.array([1.0, 0.17562081, 0.02200943])
P_OBS = jnp.array([1, 0.46, 0.31])
OBS_AGE = jnp.array([0.00118803,0.00136213,0.00342824,0.00018155,0.00010638,0.00036652,0.00394561])
REC_UP = jnp.array([1/4.9,1/4.1,0.0])
REC_SAME = jnp.array([0.0,0.0,1/4.1])
# IMPORT_STRENGTH = 0.01*ARRIVALS/jnp.max(ARRIVALS)*POSITIVITY/30.44
IMPORT_STRENGTH = 1e-9*ARRIVALS*POSITIVITY
VAX_RATE = jnp.zeros((DAYS,NAG))

SEASONALITY = 0.10139569171113522
OFFSET = 0.15655402567710153
TT = jnp.array([date_to_t('1970-01-01'), date_to_t('2020-03-19'), date_to_t('2021-03-18'), date_to_t('2021-09-14'), date_to_t('2022-02-23')])
FF = [1, 0.79711585, 0.94333315, 0.79516564, 0.95091717]
PIECEWISE_CONTACT = jnp.array([cm.piecewise(t, TT, FF, steepness=0.2) for t in range(DAYS)])
RELATIVE_CONTACT = PIECEWISE_CONTACT*(1.0 + SEASONALITY*jnp.cos(2*jnp.pi*((jnp.arange(DAYS)-274)/365 - OFFSET)))

MATERNAL_IMMUNITY = 0

STATE0_shaped = jnp.zeros((2*N_S+1,NAG))
STATE0_shaped = STATE0_shaped.at[0,:].set(CENSUS_AGE_POP-1)
STATE0_shaped = STATE0_shaped.at[1,:].set(1)
STATE0 = jnp.concatenate((jnp.array([0]), STATE0_shaped.flatten()))

params = (AGING_RATE, BIRTH_RATE, CONTACT_MATRIX,
            BETA, WANE, S_REL, P_OBS, OBS_AGE, RELATIVE_CONTACT, VAX_RATE, MATERNAL_IMMUNITY,
            REC_UP, REC_SAME, IMPORT_STRENGTH)

POINTS = jnp.arange(date_to_t('2015-10-01')-90,date_to_t('2025-05-01'))

p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
p_time_to_obs_flipped = jnp.flip(p_time_to_obs.flatten())
def obs_convolution(x):
    return jnp.convolve(x, p_time_to_obs_flipped, mode='same')

incidence = jnp.asarray(pd.read_csv("Data/Processed/KPSC_cleaned_RSV_incidence_age_daily.csv",index_col=0))
age_pops = jnp.asarray(pd.read_csv("Data/Processed/age_pops_daily.csv").values)
cases = jnp.round(incidence*age_pops[-len(incidence):])

## ODEs
from JAX_ODEs import deltas
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController

# define diffrax ODE term and solver
term = ODETerm(deltas)
solver = Dopri5()
saveat = SaveAt(ts=jnp.arange(date_to_t('2015-10-01')-90,date_to_t('2025-05-01')))
step_controller = PIDController(rtol=1e-5, atol=1e-5)

import equinox
@equinox.filter_jit
def trajectory(state0, params, term=term, solver=solver, saveat=saveat, step_controller=step_controller):
    solution = diffeqsolve(
                    term, solver,
                    t0=0, t1=date_to_t('2025-05-01')-1, dt0=None, stepsize_controller=step_controller,
                    saveat=saveat, y0=state0.flatten(), args=params, 
                    max_steps=100000,  
                    )
    # trajectory is np.diff over time of last NAG elements of solution
    cumulative_observations = solution.ys[:,-NAG:]  # shape (DAYS, NAG)
    observations = jnp.diff(cumulative_observations, axis=0)  # shape (DAYS-1, NAG)
    expected_obs = jax.vmap(obs_convolution, in_axes=1, out_axes=1)(observations)
    return expected_obs
    
bounds = jnp.array([[0,1e-2], [0,1], [0,1], [0,1], [0.1,1], [0.1, 1]])
logmeans = jnp.array([-5.620397,-2.313529,-1.8352774,-0.89054084,-2.5466263,-3.7229679 ])
test_params = x_to_params(jnp.exp(logmeans)+bounds[:,0], "RSV", "FlexStepwise", "dynamic", "flexage", fixed_params=params)
test_trajectory = trajectory(STATE0, test_params)[-len(cases):]
print(jnp.min(test_trajectory), jnp.max(test_trajectory), jnp.mean(test_trajectory))

## NumPyro model specifying parameters, priors, and likelihood
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import math
def bounded_exp(x, bound):
    return jax.nn.sigmoid(x - jnp.log(bound)) * bound

def model(obs_cases = None):
    bounds = jnp.array([[0,1e-2], [0,1], [0,1], [0,1], [0.1,1], [0.1, 1]])
    logmeans = jnp.array([-5.620397,-2.313529,-1.8352774,-0.89054084,-2.5466263,-3.7229679 ])
    logcov = jnp.array([[3.8271558e-02,4.4381559e-06,2.1638465e-03,-2.9892582e-04,-6.2326742e-03,-5.2306554e-03]
                        ,[4.4381559e-06,1.1341483e-02,-1.7394497e-04,1.1237896e-03,-3.4931437e-03,-4.7786776e-03]
                        ,[2.1638465e-03,-1.7394497e-04,1.0435041e-02,1.2063558e-03,-5.6188693e-04,-1.3124628e-02]
                        ,[-2.9892582e-04,1.1237896e-03,1.2063558e-03,7.6029692e-03,-1.4550927e-02,-3.3611790e-03]
                        ,[-6.2326738e-03,-3.4931444e-03,-5.6188717e-04,-1.4550924e-02,4.2553172e-02,-2.2161987e-03]
                        ,[-5.2306545e-03,-4.7786785e-03,-1.3124632e-02,-3.3611788e-03,-2.2161987e-03,1.6146423e+00]])
    multinorm = dist.MultivariateNormal(loc=logmeans, covariance_matrix=logcov)
    multinorm.support = dist.constraints.less_than(jnp.log(bounds[:,1] - bounds[:,0]))
    sample = numpyro.sample("params", multinorm)
    sample = jnp.exp(sample) + bounds[:,0]
    # sample = bounded_exp(sample, bounds[:,1]-bounds[:,0]) + bounds[:,0]

    # # # alternative: independent priors
    # param_means = jnp.array([0.00369014,0.0994705,0.16039424,0.41198775])
    # # param_stds = jnp.array([0.0006919,0.010534,0.01615894,0.03573003])
    # param_cov = jnp.array([
    #             [4.79203036e-07,6.11817303e-08,1.19563281e-06,-5.83700484e-07],
    #             [6.11817303e-08,1.11076320e-04,-1.89797453e-06,4.48372238e-05],
    #             [1.19563281e-06,-1.89797453e-06,2.61372765e-04,7.72046933e-05],
    #             [-5.83700484e-07,4.48372238e-05,7.72046933e-05,1.27791243e-03]
    #             ])
    # bounds = jnp.array([[0,1e-2], [0,1], [0,1], [0,1]])
    # # sample = numpyro.sample("params",
    # #     dist.TruncatedNormal(
    # #         low=jnp.array([0.0,0.0,0.0,0.0]),
    # #         high=jnp.array([1e-2,1.0,1.0,1.0]),
    # #         loc=param_means,
    # #         scale=param_stds,),
    # #         )
    # multinorm = dist.MultivariateNormal(loc=param_means, covariance_matrix=param_cov)
    # multinorm.support = dist.constraints.interval(bounds[:,0], bounds[:,1])
    # sample = numpyro.sample("params", multinorm)
    
    # wane = jnp.array([0.0, 0.0, sample[0]])
    # seasonality = sample[1]
    # offset = sample[2]
    # relative_contact = PIECEWISE_CONTACT * (1.0 + seasonality * jnp.cos(2 * jnp.pi * ((jnp.arange(DAYS) - 274) / 365 - offset)))
    # beta = sample[3]
    # srel = jnp.array([1.0, sample[4], sample[4]*sample[5]])
    sim_params = x_to_params(sample, 'RSV', 'FlexStepwise', "dynamic", 'flexage', fixed_params = params)

    # sim_params = (AGING_RATE, BIRTH_RATE, CONTACT_MATRIX,
    #         beta, wane, srel, P_OBS, OBS_AGE, relative_contact, VAX_RATE, MATERNAL_IMMUNITY,
    #         REC_UP, REC_SAME, IMPORT_STRENGTH)

    # run model
    expected_obs = trajectory(STATE0, sim_params)[-len(obs_cases):]  # get the expected observations for the last len(obs_cases) days
    # to prevent problems with Poisson likelihood, softplus with sharpness 10
    expected_obs = jnp.log(1 + jnp.exp(10 * expected_obs)) / 10

    # likelihood
    numpyro.sample("obs_cases", dist.Poisson(expected_obs), obs=obs_cases)



nuts_kernel = NUTS(
                model, 
                init_strategy = numpyro.infer.init_to_value(values={"params": 
                    jnp.array([-5.620397,-2.313529,-1.8352774,-0.89054084,-2.5466263,-3.7229679])}),
                max_tree_depth = 6,
                )
mcmc = MCMC(nuts_kernel, num_samples=4, num_warmup=4, progress_bar=True)
mcmc.run(jax.random.PRNGKey(250709), obs_cases=cases)

mcmc.print_summary()

# # save mcmc
# posterior_samples = mcmc.get_samples()
# # save posterior samples
# with open("Data/Processed/RSV_mcmc_samples_multilog2.pickle", "wb") as f:
#     pickle.dump(posterior_samples, f)
# # load posterior samples
# with open("Data/Processed/RSV_mcmc_samples_multilog2.pickle", "rb") as f:
#     posterior_samples = pickle.load(f)

# # wane, seasonality, offset, beta, srel1, srel2 = posterior_samples['params'].T

# def plot_histogram(samples, true_value, param_name, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots()
#     sns.histplot(samples, kde=True, stat='density', bins=30, ax=ax)
#     ax.set_title(f'Posterior Distribution of {param_name}')
#     ax.set_xlabel(param_name)
#     ax.set_ylabel('Density')
#     ax.axvline(x=jnp.mean(samples), color='red', linestyle='--', label='Mean')
#     ax.axvline(x=true_value, color='green', linestyle='--', label='Original')
#     ax.legend()
#     return ax

# # plot posterior samples
# sns.set_style("whitegrid")
# fig, ax = plt.subplots(3,2,figsize=(6.5, 6.5))
# lowbounds = jnp.array([0,0,0,0,0.1,0.1])
# transformed_samples = jnp.exp(posterior_samples['params']) + lowbounds
# param_names = ['wane', 'seasonality', 'offset', 'beta', 'srel1', 'srel2']
# true_values = [WANE[2], SEASONALITY, OFFSET, BETA, S_REL[1], S_REL[2]/S_REL[1]]
# for i in range(3):
#     for j in range(2):
#         idx = i*2 + j
#         plot_histogram(transformed_samples[:,idx], true_values[idx], param_names[idx], ax=ax[i,j])

# plt.tight_layout()

# plt.savefig("Figures/NumPyro_test_seasonality_beta_wane_multilog2.png")



# params, p_time_to_obs, incidence = pathogen_parameters("RSV", "FlexStepwise", CONTACT)

# # contacts1 = jnp.array([jnp.mean(jnp.sum(contact(t, 0.0001, 0), axis =1))/11.3 for t in POINTS])  # sum over age groups to get total contacts per day
# # contacts2 = jnp.array([jnp.mean(jnp.sum(contact(t, 0.0002, 0), axis =1))/11.3 for t in POINTS])  # sum over age groups to get total contacts per day
# # # stack on top of each other
# # contacts = jnp.stack([contacts1, contacts2], axis=1)  # shape (2, len(POINTS))
# # # print(np.mean(contacts))
# # # contacts = jnp.ones(len(POINTS))

# cs = jnp.array([np.mean(np.sum(contact(t,0.0001,0), axis=1))/11.3 for t in POINTS])
# ci = interpolate.interp1d(POINTS,cs)

# def SI(observed_data, time=POINTS, z_init=None, cs=cs):
#     """
#     Susceptible-Infectious (SI) model.

#     beta: infection rate (transmission rate).

#     S, I : arrays representing the number of susceptible and infected individuals
#            at each time step.
#     """

#     def dz_dt(z, t, cs=cs):
#         S = z[0]
#         I = z[1]

#         N = 762
#         # c = jnp.mean(jnp.sum(contact(t,0.0001,0)))/11.3  # Get the contact rate for the current time point
#         c = cs[t.astype(int)]  # Use precomputed contact rates
#         dS_dt = -beta * I * c * S / N
#         dI_dt =  beta * I * c *  S / N - rec * I

#         return jnp.stack([dS_dt, dI_dt])

#     if z_init is None:
#         z_init = jnp.array([[380.0, 380.0], [1.0, 1.0]])  # Initial conditions: S=763, I=1
    
#     # prior for beta
#     beta = numpyro.sample('beta',dist.Uniform(2.0, 5.0))
#     rec = numpyro.sample('rec', dist.Uniform(0.0, 1.0))

#     # integrate dz/dt, the result will have shape num_days x 2
#     # time = jnp.arange(len(observed_data))*1.0
#     z = numpyro.deterministic("z", odeint(dz_dt, z_init, time, rtol=1e-6, atol=1e-5, mxstep=1000))

#     expected_obs = numpyro.deterministic("expected_obs", jnp.sum(z[:, 1], axis=1))  # sum of infecteds
#     # clip exptected_obs at small number
#     expected_obs = jnp.clip(expected_obs, a_min=1e-6)
#     # likelihood
#     obs = numpyro.sample("observed", dist.Poisson(expected_obs), obs=observed_data)

# # observed data
# data = pd.read_csv("Data/Raw/influenza-england-1978-school.csv", index_col=0)
# observed_data = jnp.concatenate((jnp.array(data['in_bed']),jnp.zeros(len(POINTS)-len(data['in_bed']))))  # pad with zeros to match the length of POINTS
# # observed_data = jnp.array(data['in_bed'])  # use only the available data points

# # inference
# nuts_kernel = NUTS(SI)
# mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=1000, progress_bar=True)
# start_time = time.time()
# mcmc.run(jax.random.PRNGKey(0), observed_data)
# print(f"Time taken for MCMC: {time.time() - start_time:.2f} seconds")

# # extract posterior samples
# posterior_samples = mcmc.get_samples()
# samples = posterior_samples['expected_obs']
# # print(jnp.mean(posterior_samples["rec"]))
# plt.scatter(posterior_samples["beta"], posterior_samples["rec"], alpha=0.1)
# plt.show()
# # plt.figure()
# # plt.hist(samples["beta"])
# plt.axvline(x = 0.4, color = "r") # true beta value
# # plt.title("beta")
# # plt.figure()
# # plt.hist(samples["rec"])
# # plt.axvline(x = 0.2, color = "r") # true recma value
# # plt.title("rec")
# # plt.show()

# # calculate median and credible intervals
# lower_bound_infected = jnp.percentile(samples, 2.5, axis=0)
# upper_bound_infected = jnp.percentile(samples, 97.5, axis=0)
# mean_infected        = jnp.mean(samples, axis=0)

# # plot the observed data
# plt.figure(figsize=(10, 6))
# plt.scatter(jnp.arange(len(observed_data))*1.0, observed_data, color='blue', alpha=0.5, label='Infected')

# # plot median of the posterior distribution
# plt.plot(jnp.arange(len(observed_data))*1.0, mean_infected, color='red', label='Mean $I(t)$')

# # # plot uncertainty bounds
# # plt.fill_between(data['date'], lower_bound_infected, upper_bound_infected, color='pink', alpha=0.3, label='95% Credible Interval')

# plt.xlabel('Time')
# plt.ylabel('Number of Individuals')
# plt.title('SI Model Fit')
# plt.legend()
# plt.grid(True)
# plt.show()

# pathogen, seed, lockdown, option1, option2 = "RSV", 250516, "FlexStepwise", "setimport", "flexage"
# params, p_time_to_obs, incidence = pathogen_parameters("RSV", "FlexStepwise", CONTACT)

# def temp_deltas(state, t, scalar_variables, params=params):
#     NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, S_VAX, ACOV, BCOV, arrivals, regional_positivity, IMPORT_RATE, SEASONALITY, contact = params['NAG'], params['N_S'], params['AGING_RATE'], params['BIRTH_RATE'], params['WANE'], params['REC_UP'], params['REC_SAME'], params['S_REL'], params['S_AGE'], params['I_REL'], params['P_OBS'], params['birth_vax'], params['S_VAX'], params['ACOV'], params['BCOV'], params['arrivals'], params['regional_positivity'], params['IMPORT_RATE'], params['SEASONALITY'], params['contact']
#     BETA, OFFSET = (
#         scalar_variables[..., 0],
#         scalar_variables[..., 1],
#         )
#     # x = scalar_variables[..., 2:9]  # Extract the contact variables
#     # Ts = jnp.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[0]*365,date_to_t('2020-03-19')+(x[0]+x[1])*365,date_to_t('2020-03-19')+(x[0]+x[1]+x[2])*365])
#     # # Fs - element 2 must be bigger than element 1, element 3 must be smaller than element 2, element 4 must be bigger than element 2
#     # F1 = x[3] # value between 0 and 1 (first lockdown)
#     # F2 = F1 + x[4] - F1*x[4] # value between x[3] and 1 (inter-lockdown)
#     # F3 = F2*x[5] # value less than F2 (second lockdown)
#     # F4 = F2 + x[6] - F2*x[6] # value between F2 and 1 (post-lockdown)
#     # Fs = jnp.array([1,F1,F2,F3,F4])
#     # def contact(t,seasonality,offset,CONTACT=CONTACT,Ts=Ts,Fs=Fs):
#     #     return cm.piecewise(t,Ts,Fs)*(1+seasonality*jnp.cos(2*jnp.pi*((t-274)/365-offset)))*CONTACT
#     return sis_deltas(t, state, NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, S_VAX, ACOV, BCOV, arrivals, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact)
# import pickle
# with open("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","rb") as f:
#     opt = pickle.load(f)
# x = opt.x
# params["WANE"] = jnp.array([0.0,x[0],0.0])
# params["SEASONALITY"] = x[1]
# params["OFFSET"] = x[2]
# params["BETA"] = x[3]
# n = 4
# params["IMPORT_RATE"] = 0.01
# if (("Influenza" in pathogen) and (option2 != 'nr')) or (seed <= 250407):
#     srel, pobsrel = constrained_immunity(x[n],x[n+1],x[n+2])
#     params["S_REL"] = srel
#     n += 3
# elif pathogen == 'RSV':
#     params["S_REL"] = jnp.array([1,x[n],x[n]*x[n+1]])
#     pobsrel = jnp.array([1,0.46,0.31]) # Henderson 1979
#     n += 2
# else:
#     params["S_REL"] = jnp.array([1,x[n],x[n]*x[n+1]])
#     pobsrel = jnp.array([1,x[n+2],x[n+2]*x[n+3]])
#     n += 4
# params["P_OBS"] = pobsrel
# Ts = jnp.array([date_to_t('1970-01-01'),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[n]*365,date_to_t('2020-03-19')+(x[n]+x[n+1])*365,date_to_t('2020-03-19')+(x[n]+x[n+1]+x[n+2])*365])
# F1 = x[n+3] # value between 0 and 1 (first lockdown)
# F2 = F1 + x[n+4] - F1*x[n+4] # value between x[n+3] and 1 (inter-lockdown)
# F3 = F2*x[n+5] # value less than F2 (second lockdown)
# F4 = F2 + x[n+6] - F2*x[n+6] # value between F2 and 1 (post-lockdown)
# Fs = jnp.array([1,F1,F2,F3,F4])
# # @jit
# def contact(t,seasonality,offset):
#     return cm.piecewise(t,Ts,Fs)*(1+seasonality*jnp.cos(2*jnp.pi*((t-274)/365-offset)))*CONTACT
# params["contact"] = contact
# n += 7
# OBS_AGE = jnp.array([x[n],x[n+1],x[n+2],x[n+3],x[n+4],x[n+5],x[n+6]])
# print(params)

# def model(case_data, params=params, POINTS=POINTS, OBS_AGE=OBS_AGE, p_time_to_obs=p_time_to_obs):
#     N_S, NAG = params["N_S"], params["NAG"]

#     # ## Initial conditions
#     STATE0 = jnp.zeros((2*N_S+2)*NAG)
#     STATE0 = STATE0.at[NAG:2*NAG].set(CENSUS_AGE_POP-1) # Everyone is susceptible except
#     STATE0 = STATE0.at[2*NAG:3*NAG].set(1) # one individual in each age group that is infected.

#     def temp_deltas(state, t, params=params):
#         NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, S_VAX, ACOV, BCOV, arrivals, regional_positivity, IMPORT_RATE, SEASONALITY, OFFSET, contact = params['NAG'], params['N_S'], params['AGING_RATE'], params['BIRTH_RATE'], params['WANE'], params['REC_UP'], params['REC_SAME'], params['S_REL'], params['S_AGE'], params['I_REL'], params['P_OBS'], params['birth_vax'], params['S_VAX'], params['ACOV'], params['BCOV'], params['arrivals'], params['regional_positivity'], params['IMPORT_RATE'], params['SEASONALITY'], params['OFFSET'], params['contact']
#         # BETA, OFFSET = (
#         #     scalar_variables[..., 0],
#         #     scalar_variables[..., 1],
#         #     )
#     #     # x = scalar_variables[..., 2:9]  # Extract the contact variables
#     #     # Ts = jnp.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[0]*365,date_to_t('2020-03-19')+(x[0]+x[1])*365,date_to_t('2020-03-19')+(x[0]+x[1]+x[2])*365])
#     #     # # Fs - element 2 must be bigger than element 1, element 3 must be smaller than element 2, element 4 must be bigger than element 2
#     #     # F1 = x[3] # value between 0 and 1 (first lockdown)
#     #     # F2 = F1 + x[4] - F1*x[4] # value between x[3] and 1 (inter-lockdown)
#     #     # F3 = F2*x[5] # value less than F2 (second lockdown)
#     #     # F4 = F2 + x[6] - F2*x[6] # value between F2 and 1 (post-lockdown)
#     #     # Fs = jnp.array([1,F1,F2,F3,F4])
#     #     # def contact(t,seasonality,offset,CONTACT=CONTACT,Ts=Ts,Fs=Fs):
#     #     #     return cm.piecewise(t,Ts,Fs)*(1+seasonality*jnp.cos(2*jnp.pi*((t-274)/365-offset)))*CONTACT
#         return sis_deltas(t, state, NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, S_VAX, ACOV, BCOV, arrivals, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact)

#     # priors for beta and offset
#     BETA = numpyro.sample("BETA", dist.Normal(params["BETA"], 0.1))  # Use the optimized beta value
#     # OFFSET = numpyro.sample("OFFSET", dist.Uniform(0.0, 1.0))

#     # integrate the ODEs
#     z = numpyro.deterministic("z", odeint(temp_deltas, STATE0, POINTS))
#     # result = {'t': POINTS, 'y': z.T}  # Transpose to match the expected shape

#     # calculate observations
#     trajectory = observations(z,POINTS,params,OBS_AGE,incidence=False,time_conversion=1)
#     expected_obs = jnp.sum(jnp.array([jnp.roll(trajectory,i,axis=0)*p_time_to_obs[i] for i in range(len(p_time_to_obs))]),axis=0)
#     # # cut off the first few days of the trajectory since they are not used in the likelihood (and the roll function is wrapping around)
    
#     expected_obs = expected_obs[-len(case_data):]

#     # poission likelihood
#     obs = numpyro.sample("observed_cases", dist.Poisson(expected_obs), obs=case_data)

# # transform data into cases appropriate to simulated population size
# age_pops = jnp.array(pd.read_csv("Data/Processed/SISn_population_over_time_250610.csv", index_col=0).values)
# case_data = jnp.round(incidence*age_pops)[:,:2]


# # inference
# nuts_kernel = NUTS(model)
# mcmc = MCMC(nuts_kernel, num_warmup=10, num_samples=10, num_chains=1)
# mcmc.run(PRNGKey(0), case_data)

# # get posterior samples
# posterior_samples = mcmc.get_samples()

# plt.hist(posterior_samples["BETA"])
# plt.axvline(x = params["BETA"], color = "r") # true beta value
# plt.title("BETA")
# plt.savefig("Figures/BETA_histogram.png")