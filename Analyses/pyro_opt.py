## Script to fit SIS model to data using JAX and NumPyro
## BJS June 2025

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import matplotlib.pyplot as plt
import seaborn as sns
import time

from vaccination import birth_vax, all_vax, flu_rate, flu_eff_coverage
import contact_model as cm
from SISn_ODEs import single_pathogen_deltas as sis_deltas
from Parameters.census_population import *
from Parameters.times_and_contacts import *

from utils import *
from demography import *
from mobility_and_import import *


params, p_time_to_obs, incidence = pathogen_parameters("RSV", "FlexStepwise", CONTACT)

def SI(observed_data, time=POINTS, z_init=None):
    """
    Susceptible-Infectious (SI) model.

    beta: infection rate (transmission rate).

    S, I : arrays representing the number of susceptible and infected individuals
           at each time step.
    """

    def dz_dt(z, t):
        S = z[0]
        I = z[1]

        N = 762
        dS_dt = -beta * I * S / N
        dI_dt =  beta * I * S / N - rec * I

        return jnp.stack([dS_dt, dI_dt])

    if z_init is None:
        z_init = jnp.array([[380.0, 380.0], [1.0, 1.0]])  # Initial conditions: S=763, I=1
    
    # prior for beta
    beta = numpyro.sample('beta',dist.Uniform(2.0, 5.0))
    rec = numpyro.sample('rec', dist.Uniform(0.0, 1.0))

    # integrate dz/dt, the result will have shape num_days x 2
    # time = jnp.arange(len(observed_data))*1.0
    z = numpyro.deterministic("z", odeint(dz_dt, z_init, time, rtol=1e-6, atol=1e-5, mxstep=1000))

    expected_obs = numpyro.deterministic("expected_obs", jnp.sum(z[:, 1], axis=1))  # sum of infecteds
    # clip exptected_obs at small number
    expected_obs = jnp.clip(expected_obs, a_min=1e-6)
    # likelihood
    obs = numpyro.sample("observed", dist.Poisson(expected_obs), obs=observed_data)

# observed data
data = pd.read_csv("Data/Raw/influenza-england-1978-school.csv", index_col=0)
observed_data = jnp.concatenate((jnp.array(data['in_bed']),jnp.zeros(len(POINTS)-len(data['in_bed']))))  # pad with zeros to match the length of POINTS
# observed_data = jnp.array(data['in_bed'])  # use only the available data points

# inference
nuts_kernel = NUTS(SI)
mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=1000, progress_bar=True)
start_time = time.time()
mcmc.run(jax.random.PRNGKey(0), observed_data)
print(f"Time taken for MCMC: {time.time() - start_time:.2f} seconds")

# extract posterior samples
posterior_samples = mcmc.get_samples()
samples = posterior_samples['expected_obs']
# print(jnp.mean(posterior_samples["rec"]))
plt.scatter(posterior_samples["beta"], posterior_samples["rec"], alpha=0.1)
plt.show()
# plt.figure()
# plt.hist(samples["beta"])
# plt.axvline(x = 0.4, color = "r") # true beta value
# plt.title("beta")
# plt.figure()
# plt.hist(samples["rec"])
# plt.axvline(x = 0.2, color = "r") # true recma value
# plt.title("rec")
# plt.show()

# calculate median and credible intervals
lower_bound_infected = jnp.percentile(samples, 2.5, axis=0)
upper_bound_infected = jnp.percentile(samples, 97.5, axis=0)
mean_infected        = jnp.mean(samples, axis=0)

# plot the observed data
plt.figure(figsize=(10, 6))
plt.scatter(jnp.arange(len(observed_data))*1.0, observed_data, color='blue', alpha=0.5, label='Infected')

# plot median of the posterior distribution
plt.plot(jnp.arange(len(observed_data))*1.0, mean_infected, color='red', label='Mean $I(t)$')

# # plot uncertainty bounds
# plt.fill_between(data['date'], lower_bound_infected, upper_bound_infected, color='pink', alpha=0.3, label='95% Credible Interval')

plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SI Model Fit')
plt.legend()
plt.grid(True)
plt.show()

# params, p_time_to_obs, incidence = pathogen_parameters("RSV", "FlexStepwise", CONTACT)
# OBS_AGE = jnp.array([1, 0.75, 0.5, 0.25, 0.05, 0.15, 1])

# # def temp_deltas(state, t, scalar_variables, params=params):
# #     NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, S_VAX, ACOV, BCOV, arrivals, regional_positivity, IMPORT_RATE, SEASONALITY, contact = params['NAG'], params['N_S'], params['AGING_RATE'], params['BIRTH_RATE'], params['WANE'], params['REC_UP'], params['REC_SAME'], params['S_REL'], params['S_AGE'], params['I_REL'], params['P_OBS'], params['birth_vax'], params['S_VAX'], params['ACOV'], params['BCOV'], params['arrivals'], params['regional_positivity'], params['IMPORT_RATE'], params['SEASONALITY'], params['contact']
# #     BETA, OFFSET = (
# #         scalar_variables[..., 0],
# #         scalar_variables[..., 1],
# #         )
# #     # x = scalar_variables[..., 2:9]  # Extract the contact variables
# #     # Ts = jnp.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[0]*365,date_to_t('2020-03-19')+(x[0]+x[1])*365,date_to_t('2020-03-19')+(x[0]+x[1]+x[2])*365])
# #     # # Fs - element 2 must be bigger than element 1, element 3 must be smaller than element 2, element 4 must be bigger than element 2
# #     # F1 = x[3] # value between 0 and 1 (first lockdown)
# #     # F2 = F1 + x[4] - F1*x[4] # value between x[3] and 1 (inter-lockdown)
# #     # F3 = F2*x[5] # value less than F2 (second lockdown)
# #     # F4 = F2 + x[6] - F2*x[6] # value between F2 and 1 (post-lockdown)
# #     # Fs = jnp.array([1,F1,F2,F3,F4])
# #     # def contact(t,seasonality,offset,CONTACT=CONTACT,Ts=Ts,Fs=Fs):
# #     #     return cm.piecewise(t,Ts,Fs)*(1+seasonality*jnp.cos(2*jnp.pi*((t-274)/365-offset)))*CONTACT
# #     return sis_deltas(t, state, NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, S_VAX, ACOV, BCOV, arrivals, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact)

# def model(case_data, params=params, POINTS=POINTS, OBS_AGE=OBS_AGE, p_time_to_obs=p_time_to_obs):
#     N_S, NAG = params["N_S"], params["NAG"]

#     # ## Initial conditions
#     # STATE0 = jnp.zeros((2*N_S+2)*NAG)
#     # STATE0 = STATE0.at[NAG:2*NAG].set(CENSUS_AGE_POP-1) # Everyone is susceptible except
#     # STATE0 = STATE0.at[2*NAG:3*NAG].set(1) # one individual in each age group that is infected.

#     STATE0 = jnp.array([[2e6, 2e6], [1, 1]])

#     # def temp_deltas(state, t, params=params):
#     #     NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, S_VAX, ACOV, BCOV, arrivals, regional_positivity, IMPORT_RATE, SEASONALITY, OFFSET, contact = params['NAG'], params['N_S'], params['AGING_RATE'], params['BIRTH_RATE'], params['WANE'], params['REC_UP'], params['REC_SAME'], params['S_REL'], params['S_AGE'], params['I_REL'], params['P_OBS'], params['birth_vax'], params['S_VAX'], params['ACOV'], params['BCOV'], params['arrivals'], params['regional_positivity'], params['IMPORT_RATE'], params['SEASONALITY'], params['OFFSET'], params['contact']
#     #     # BETA, OFFSET = (
#     #     #     scalar_variables[..., 0],
#     #     #     scalar_variables[..., 1],
#     #     #     )
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
#     #     return sis_deltas(t, state, NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, S_VAX, ACOV, BCOV, arrivals, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact)

#     # very simple naiive sis model
#     def temp_deltas(state, t):
#         # Ujnpack the state vector
#         S = state[0]
#         I = state[1]
#         N = S + I
#         dS = -BETA * S * I / N  # Infection term
#         dI = BETA * S * I / N - REC * I
#         return jnp.stack([dS, dI])

#     # priors for beta and offset
#     BETA = numpyro.sample("BETA", dist.Uniform(0.0, 1.0))
#     REC = numpyro.sample("REC", dist.Uniform(0.0, 1.0))
#     # OFFSET = numpyro.sample("OFFSET", dist.Uniform(0.0, 1.0))

#     # integrate the ODEs
#     z = numpyro.deterministic("z", odeint(temp_deltas, STATE0, POINTS))
#     # result = {'t': POINTS, 'y': z.T}  # Transpose to match the expected shape

#     # calculate observations
#     # trajectory = observations(z,POINTS,params,OBS_AGE,incidence=False,time_conversion=1)
#     # expected_obs = jnp.sum(jnp.array([jnp.roll(trajectory,i,axis=0)*p_time_to_obs[i] for i in range(len(p_time_to_obs))]),axis=0)
#     # # cut off the first few days of the trajectory since they are not used in the likelihood (and the roll function is wrapping around)
#     # expected_obs = expected_obs[-len(case_data):]
#     expected_obs = jnp.sum(z[len(p_time_to_obs):,1] * jnp.array([0.5,0.2]), axis=0)  # sum of infecteds

#     # poission likelihood
#     obs = numpyro.sample("observed_cases", dist.Poisson(expected_obs), obs=case_data)

# # transform data into cases appropriate to simulated population size
# age_pops = jnp.array(pd.read_csv("Data/Processed/SISn_population_over_time_250610.csv", index_col=0).values)
# case_data = jnp.round(incidence*age_pops)[:,:2]


# # inference
# nuts_kernel = NUTS(model)
# mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000, num_chains=1)
# mcmc.run(PRNGKey(0), case_data)

# # get posterior samples
# posterior_samples = mcmc.get_samples()

# print(posterior_samples)