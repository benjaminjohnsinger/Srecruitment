## SIS model with n susceptibility classes, for a single pathogen
## BJS September 2024

# from autograd # import jax.numpy as jnp
# from autograd import grad, jacobian
# import jax.numpy as jnp
import numdifftools as nd
import scipy as sp
import pandas as pd
import time
import itertools as it
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
from matplotlib import cm as pltcm
import corner
import pickle
import colorsys
import datetime
from numba import jit
from sklearn import decomposition

from vaccination import birth_vax, birth_vax, flu_rate, flu_eff_coverage
import contact_model as cm
from SISn_ODEs import single_pathogen_deltas as sis_deltas
from Parameters.census_population import *
from Parameters.times_and_contacts import *

from utils import *
from demography import *
from mobility_and_import import *
from clustering import *
from sim_grid import *
from plotting import *
from fit_MCMC import *

# print(flu_rate(date_to_t(START),1-(np.arange(3)/3),np.ones(7),np.ones(7)))

# print all numpy array elements
np.set_printoptions(threshold=np.inf)

# # plot contact matrix (CONTACT) with age group labels
# from Parameters.times_and_contacts import *
# from Parameters.RSV import *
# fig, ax = plt.subplots(figsize=(6.5,6.5))
# ax.imshow(CONTACT, cmap='viridis', aspect='auto')
# cbar = plt.colorbar(ax.imshow(CONTACT, cmap='viridis', aspect='auto'), ax=ax)
# cbar.set_label('Contact rate')
# ax.set_xticks(jnp.arange(NAG), AGE_GROUP_NAMES, rotation=45)
# ax.set_yticks(jnp.arange(NAG), AGE_GROUP_NAMES)
# ax.set_xlabel('Age group')
# ax.set_ylabel('Age group')
# ax.set_title('Contact matrix')
# plt.tight_layout()
# plt.savefig('Figures/contact_matrix.png', dpi=300)

# # adjust plot text size
# plt.rcParams.update({'font.size':20})
# # text type is palatino
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Palatino']

# fig, ax = plt.subplots(3,2,figsize=(6.5,8.5))
# kpsc_positive_test_plot(ax[0,0],pathogen="RSV",AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, incidence=True, legend=True,aggregation="Month")
# kpsc_positive_test_plot(ax[1,0],pathogen="InfluenzaA",AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, incidence=True, legend=False,aggregation="Month")
# kpsc_positive_test_plot(ax[2,0],pathogen="Adenovirus",AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, incidence=True, legend=False,aggregation="Month")
# kpsc_positive_test_plot(ax[0,1],pathogen="Metapneumovirus",AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, incidence=True, legend=False,aggregation="Month")
# kpsc_positive_test_plot(ax[1,1],pathogen="InfluenzaB",AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, incidence=True, legend=False,aggregation="Month")
# kpsc_positive_test_plot(ax[2,1],pathogen="Parainfluenza3",AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, incidence=True, legend=False,aggregation="Month")
# for i in range(2):
#     for j in range(2):
#         ax[i,j].set_xlabel("")
# for i in range(3):
#     ax[i,1].set_ylabel("")
#     for j in range(2):
#         ax[i,j].set_xticklabels(["","2016","","2018","","2020","","2022","","2024"])

# ax[0,0].set_title("RSV")
# ax[1,0].set_title("Influenza A")
# ax[2,0].set_title("Adenovirus")
# ax[0,1].set_title("Metapneumovirus")
# ax[1,1].set_title("Influenza B")
# ax[2,1].set_title("Parainfluenza 3")

# ax[0,0].legend(ncol=2,title="Age group")

# plt.tight_layout()
# plt.savefig('Figures/test.png',dpi=300)


start_date = '2015-07-04'
end_date = '2023-09-30'
EPOCH = pd.to_datetime('1970-01-01')
START = pd.to_datetime(start_date) 
END = pd.to_datetime(end_date)
PERIOD = pd.date_range(start=START, end=END, freq='D')
POINTS = np.array(date_to_t(PERIOD))
# Ts = jnp.array([date_to_t('1970-01-01'), date_to_t('2020-03-19'), date_to_t('2020-12-05'), date_to_t('2020-12-10'), date_to_t('2021-08-12')])
# Fs = jnp.array([1,0.4001427,0.89507013,0.70283776,0.93153405])
# @jit
# def contact(t,seasonality,offset):
#     return cm.piecewise(t,Ts,Fs)*(1+seasonality*jnp.cos(2*jnp.pi*((t-274)/365-offset)))*CONTACT
from Parameters.RSV import *
# Parameters from differential evolution
# RSV_params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': jnp.array([0.        , 0.00136468, 0.        ]), 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': jnp.array([1.        , 0.10032712, 0.04941595]), 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': jnp.array([0.04535184, 0.02086185, 0.01405907]), 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
# 'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0.01, 'BETA': 0.5001441612672278, 'SEASONALITY': 0.0875344346283, 'OFFSET': 0.13221726766772357,
# 'contact': contact}
# params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
# 'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': 0, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
# 'contact': contact}
NAG = 7
N_S = 3
DAYS = 19632
BETA = 0.2
WANE = np.array([0.0,0.00395647,0])
S_REL = np.array([1.0,0.11615855, 0.02183006])
P_OBS = np.array([1,0.46,0.31])
OBS_AGE = np.array([1.51790104e-03, 2.86322367e-03, 4.93356844e-03, 1.87785669e-04, 
    4.49717437e-05, 3.69412228e-04, 3.33166543e-03])
REC_UP = np.array([1/4.9,1/4.1,0.0])
REC_SAME = np.array([0.0,0.0,1/4.1])
fig, ax = plt.subplots(1,4,figsize=(12,3))
ARRIVALS = np.genfromtxt('Data/Processed/arrivals_daily.csv', delimiter=',')
POSITIVITY = np.genfromtxt('Data/Processed/RSV_positivity_daily.csv', delimiter=',')
arrivals = lambda t: ARRIVALS[int(t)]/(30.44*np.max(ARRIVALS)) if t < len(ARRIVALS) else 0.0
regional_positivity = lambda t: POSITIVITY[int(t)] if t < len(POSITIVITY) else 0.0
dayrange = np.arange(date_to_t("2016-10-01"),date_to_t("2017-10-01"))
ax[0].plot([arrivals(day) for day in dayrange], label='Arrivals')
ax[0].set_title('Arrivals')
ax[1].plot([regional_positivity(day) for day in dayrange], label='Regional positivity')
ax[1].set_title('Regional positivity')
IMPORT_RATE = 0.01
CONTACT = np.asarray(pd.read_csv('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', header=None).values)
contact = lambda t, seasonality, offset: (1.0+seasonality*np.cos(2*np.pi*((t-274)/365-offset)))*CONTACT
BIRTH_RATE = np.genfromtxt('Data/Processed/birth_rate_daily.csv', delimiter=',')
birth_rate = lambda t: BIRTH_RATE[int(t)]
ax[2].plot([contact(day, 0.0896529807571384, 0.18329654609521207).sum() for day in dayrange], label='Contact')
ax[2].set_title('Contact')
ax[3].plot([birth_rate(day) for day in dayrange], label='Birth rate')
ax[3].set_title('Birth rate')
plt.tight_layout()
plt.savefig('Figures/old_model_rates_test.png', dpi=300)
params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': np.ones(7), 'I_REL': np.ones((3,1)), 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'S_VAX': 2, 'ACOV': ACOV, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': 0.0896529807571384, 'OFFSET': 0.18329654609521207,
'contact': contact}
# OBS_AGE = jnp.array([0.24594159,0.1857486,0.12555562,0.02660094,0.025,0.09684663,1])
# p_time_to_obs = np.asarray(pd.read_csv("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=','))
p_time_to_obs = np.zeros(90)
p_time_to_obs[-1] = 1.0
## Initial conditions
STATE0 = np.zeros((2*N_S+2)*NAG)
# STATE0 = STATE0.at[NAG:2*NAG].set(CENSUS_AGE_POP-1) # Everyone is susceptible except
STATE0[NAG:2*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
# STATE0 = STATE0.at[2*NAG:3*NAG].set(1) # one individual in each age group that is infected.
STATE0[2*NAG:3*NAG] = 1
# params = RSV_params.copy()
# start_date = '2015-10-01'
# end_date = '2021-10-01'
# EPOCH = pd.to_datetime('1970-01-01')
# START = pd.to_datetime(start_date) 
# END = pd.to_datetime(end_date)
# PERIOD = pd.date_range(start=START, end=END, freq='D')
# POINTS = jnp.array(date_to_t(PERIOD))
T_LOCKDOWN = date_to_t('2016-01-01')
LOCKDOWN_DURATION = 365
# Ts = jnp.array([date_to_t('1970-01-01'), T_LOCKDOWN, T_LOCKDOWN+LOCKDOWN_DURATION])
# Fs = jnp.array([1,0.4,1])
# @jit
# def contact(t,seasonality,offset,CONTACT=CONTACT):
#     return (1+seasonality*jnp.cos(2*jnp.pi*((t-274)/365-offset)))*CONTACT
# params["contact"] =  contact

# start = time.time()
# params, results = sim_grid(STATE0,params,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,
# grid_params=(("BETA",),("WANE",),("S_REL",)),
# N=7,factors=(1,1,1),grid_mode=("scale","scale","power_vec"))
# print(f"Simulation took {time.time()-start:.2f} seconds")
# # Save the results
# with open('Data/Processed/SIS_3D_power.pickle','wb') as f:
#     pickle.dump(results,f)
# with open('Data/Processed/SIS_3D_power_params.pickle','wb') as f:
#     pickle.dump(params,f)

# incidence = pd.read_csv("Data/Processed/KPSC_RSV_incidence_age_daily.csv",index_col=0)

# # NON-NEGATIVE log likelihood
# def likelihood(x):
#     sim_params = RSV_params.copy()
#     sim_params["WANE"] = jnp.array([0.0,x[0],0.0])
#     sim_params["SEASONALITY"] = x[1]
#     sim_params["OFFSET"] = x[2]
#     sim_params["BETA"] = x[3]
#     n = 4
#     sim_params["IMPORT_RATE"] = x[n]
#     n += 1
#     sim_params["S_REL"] = jnp.array([1,x[n],x[n]*x[n+1]])
#     pobsrel = jnp.array([1,x[n+2],x[n+2]*x[n+3]])
#     n += 4
#     sim_params["P_OBS"] = x[n]*pobsrel
#     n += 1
#     Ts = jnp.array([date_to_t(EPOCH),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[n]*365,date_to_t('2020-03-19')+(x[n]+x[n+1])*365,date_to_t('2020-03-19')+(x[n]+x[n+1]+x[n+2])*365])
#     # Fs - element 2 must be bigger than element 1, element 3 must be smaller than element 2, element 4 must be bigger than element 2
#     F1 = x[n+3] # value between 0 and 1 (first lockdown)
#     F2 = F1 + x[n+4] - F1*x[n+4] # value between x[n+3] and 1 (inter-lockdown)
#     F3 = F2*x[n+5] # value less than F2 (second lockdown)
#     F4 = F2 + x[n+6] - F2*x[n+6] # value between F2 and 1 (post-lockdown)
#     Fs = jnp.array([1,F1,F2,F3,F4])
#     # @jit
#     def contact(t,seasonality,offset):
#         return cm.piecewise(t,Ts,Fs)*(1+seasonality*jnp.cos(2*jnp.pi*((t-274)/365-offset)))*CONTACT
#     sim_params["contact"] = contact
#     n += 7
#     obs_age = age_detection(NAG,x[n],x[n+1],x[n+2])
#     lh = SIS_likelihood(incidence,sim_params,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)
#     # print("neg log likelihood: ",lh)
#     return lh
# x_init = jnp.array([0.00438131,0.09607721848972506,0.12504213724959057,0.47973501805559776,0.008897365162093825,0.10871732,0.0372482/0.10871732, 0.01272225/0.01273131, 0.00913129/(0.01273131*0.01272225), 0.01273131, 261/365, 5/365, 245/365, 0.4001427, 0.89507013, 0.70283776, 0.93153405, 1-0.68056091, 0.9479402-0.09321769, 0.5-((1-0.9479402)/2)])
# print(likelihood(x_init))
# # calculate the hessian - could try method="forward" for faster but less reliable results. May want to choose a smaller order, 1-2.
# hessian_f = nd.Hessian(likelihood,method="forward",order="1")
# hessian = hessian_f(x_init)
# print(hessian)
# fisher_info = jnp.linalg.inv(-hessian)
# prop_sigma = jnp.sqrt(jnp.diag(fisher_info))
# upper_bound = x_init + 1.96*prop_sigma
# lower_bound = x_init - 1.96*prop_sigma
# print("Upper bound: ", upper_bound)
# print("Lower bound: ", lower_bound)


# ## Period of simulation
# EPOCH = pd.to_datetime('1970-01-01')
# START = pd.to_datetime('2015-08-01')
# END = pd.to_datetime('2023-10-01')
# PERIOD = pd.date_range(start=START, end=END, freq='D')

# ## Contacts and force of infection
# IMPORT_RATE = 1e-12
# # Contact matrix for all contact types
# CONTACT = jnp.genfromtxt('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', dtype=jnp.float64)
# print(CONTACT.shape)
# print(NAG)
# print(OBS_AGE.shape)
# print(CENSUS_AGE_POP.shape)

# # Lockdown and other mobility changes
# T_LOCKDOWN = date_to_t('2020-03-01')
# LOCKDOWN_DURATION = 365
# LOCKDOWN_REDUCTION = 0.4
# Ts = jnp.array([date_to_t(EPOCH),T_LOCKDOWN,date_to_t('2021-05-01'),date_to_t('2021-12-01'),date_to_t('2022-03-01')])
# Fs = jnp.array([1,0.4,1,0.4,1])
# @jit
# def contact(t,seasonality,offset):
#     return cm.piecewise(t,Ts,Fs)*(1+seasonality*jnp.cos(2*jnp.pi*(t/365-offset)))*CONTACT

# # ## Integrate the system
# # # POINTS = jnp.concat((jnp.zeros(1),jnp.arange(T_LOCKDOWN-12*12,T_LOCKDOWN+LOCKDOWN_DURATION+12*12,0.2),jnp.ones(1)*PERIOD))
# # # POINTS = jnp.array([date_to_t('1960-01-01') + pd.DateOffset(months=x) for x in range(PERIOD)])
# T_VAX = date_to_t('2035-01-01')

# POINTS = jnp.array(date_to_t(PERIOD))


# from Parameters.times_and_contacts import *
# from Parameters.RSV_Lowensteyn import *
# p_time_to_obs = jnp.genfromtxt("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',',dtype=jnp.float64)


# ## Initial conditions
# STATE0 = jnp.zeros((2*N_S+2)*NAG)
# STATE0[NAG:2*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
# STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

# # Parameters for the ODE
# params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': ACOV, 'BCOV': BCOV,
# 'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
# 'contact': contact}
# params["ACOV"] = ACOV

# @jit
# # def contact(t,seasonality,offset):
# #     return cm.google_prestige_work(t)*(1+seasonality*jnp.cos(2*jnp.pi*((t-274)/365-offset)))*CONTACT
# # params["contact"] = contact

# incidence = jnp.array(pd.read_csv("Data/Processed/KPSC_RSV_incidence_age_daily.csv",index_col=0))

# def likelihood(x):
#     sim_params = params.copy()
#     sim_params["WANE"] = jnp.array([0.0,x[0],0.0])
#     sim_params["SEASONALITY"] = x[1]
#     sim_params["OFFSET"] = x[2]
#     sim_params["BETA"] = x[3]
#     sim_params["IMPORT_RATE"] = x[4]
#     sim_params["P_OBS"] = x[5]*jnp.array([1,0.4,0])
#     obs_age = age_detection(NAG,x[6],x[7],x[8])
#     return -SIS_likelihood(incidence,sim_params,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)

# # nelders-mead optimization
# opt = sp.optimize.minimize(likelihood,[1/365,0.13,(274/365)+3.65/(2*jnp.pi),0.046,0.01,0.072*0.45,0.4,0.05,0.95],method='Nelder-Mead')
# print(opt)
# print(opt.x)

# x = opt.x
# params["WANE"] = jnp.array([0.0,x[0],0.0])
# params["SEASONALITY"] = x[1]
# params["OFFSET"] = x[2]
# params["BETA"] = x[3]
# params["IMPORT_RATE"] = x[4]
# params["P_OBS"] = x[5]*jnp.array([1,0.4,0])
# OBS_AGE = age_detection(NAG,x[6],x[7],x[8])


# # # # # # # # # # # #### One-shot line plot #####
# set pyplot font to arial
plt.rcParams.update({'font.size':11})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


# # # # params['contact'] = lambda t,seasonality,offset: contact(t,seasonality,offset)*(1-flu_eff_coverage(t,S_REL))
# # # # params['BETA'] = 0
# from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController, SaveAt
# term = ODETerm(sis_deltas)
# solver = Dopri5()
# saveat = SaveAt(ts=POINTS)
# stepsize_controller = PIDController(rtol=1,atol=1)
# solution = diffeqsolve(term, solver, t0=date_to_t(EPOCH), t1=POINTS[-1], dt0=0.1, y0=STATE0, args=(NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, S_VAX, ACOV, BCOV, arrivals, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact)
# ,saveat=saveat,stepsize_controller=stepsize_controller, max_steps=None)
# print(solution)

# import jax
# from jax.experimental.ode import odeint
# from jax.random import PRNGKey
# import numpyro
# import numpyro.distributions as dist
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
# key = PRNGKey(250609)
# scalar_sample = numpyro.sample(
#     "scalar_variables",
#     dist.TruncatedNormal(
#         low = 0.0,
#         loc = jnp.array([params['BETA'], params['OFFSET']]),
#         scale = jnp.array([0.1, 0.1]),
#     ),
#     rng_key=key
# )
# start = time.time()
# solution = odeint(temp_deltas, STATE0, jnp.array(POINTS, dtype=jnp.float32), scalar_sample)
# print("result took ",time.time()-start," seconds")
# print(solution[-1])

# def temp_deltas(state, t, params=params):
#     NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, S_VAX, ACOV, BCOV, arrivals, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact = params['NAG'], params['N_S'], params['AGING_RATE'], params['BIRTH_RATE'], params['WANE'], params['REC_UP'], params['REC_SAME'], params['S_REL'], params['S_AGE'], params['I_REL'], params['P_OBS'], params['birth_vax'], params['S_VAX'], params['ACOV'], params['BCOV'], params['arrivals'], params['regional_positivity'], params['IMPORT_RATE'], params['BETA'], params['SEASONALITY'], params['OFFSET'], params['contact']
#     return sis_deltas(t, state, NAG, N_S, AGING_RATE, birth_rate, WANE, REC_UP, REC_SAME, S_REL, S_AGE, I_REL, P_OBS, birth_vax, S_VAX, ACOV, BCOV, arrivals, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact)
# sstart = time.time()
# jitted_observations = jax.jit(observations, static_argnames=('NAG', 'N_S', 'BETA', 'contact', 'SEASONALITY', 'OFFSET', 'incidence', 'time_conversion'))
# print("jitting took ",time.time()-sstart," seconds")
# start = time.time()
# solution = odeint(temp_deltas, STATE0, jnp.array(POINTS,dtype=jnp.float32))
# print(time.time() - start)
# print([jnp.sum(solution[jnp.arange(i,(N_S*N_C+1)*NAG,NAG),len(p_time_to_obs):],axis=0) for i in range(NAG)])
# print(time.time() - start)
# obs = observations(solution.T, POINTS, params["NAG"], params["N_S"], params["BETA"], params["contact"], params["SEASONALITY"], params["OFFSET"], params["S_REL"], params["I_REL"], params["P_OBS"], OBS_AGE,incidence=True,time_conversion=30.44)
# print(time.time() - start)
# # find population size over time
# age_pops = jnp.array([jnp.sum(solution[len(p_time_to_obs):,jnp.arange(i,(N_S*N_C+1)*NAG,NAG)],axis=1) for i in range(NAG)]).T
# plt.plot(age_pops)
# plt.show()

# save as csv
# age_pops_df = pd.DataFrame(age_pops, index=PERIOD[len(p_time_to_obs):], columns=AGE_GROUP_NAMES)
# age_pops_df.to_csv('Data/Processed/SISn_population_over_time_250610.csv')

# # print("result took ",time.time()-start," seconds")
# result = {'t': POINTS,'y': solution.T}  # Transpose to match the expected shape
# obs = observations(result,params,OBS_AGE,incidence=True,time_conversion=30.44)
# fig, ax = plt.subplots()
# mx = lockdown_incidence_plot(ax,STATE0,params,OBS_AGE,PERIOD,POINTS,date_to_t('2024-03-19'),365,result=result,obs=obs,label="Simulation",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs)
# plt.show()

# import jax
# sis_deltas = jax.jit(sis_deltas, static_argnames=('NAG', 'N_S', 'birth_rate', 'birth_vax', 'S_VAX', 'ACOV', 'BCOV', 'arrivals', 'regional_positivity', 'IMPORT_RATE', 'BETA', 'SEASONALITY', 'OFFSET', 'contact'))

# start = time.time()
result = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params.values(),t_eval=POINTS,method='RK45')
# print("result took ",time.time()-start," seconds")

# plt.plot(result.t,np.sum(result.y,axis=0))
# plt.show()
# # get total infectious compartment over time
# I = jnp.sum(jnp.array([result.y[(N_C*j+2)*NAG:(N_C*j+3)*NAG,:] for j in range(N_S)]),axis=0)
# age_pops = jnp.array([jnp.sum(result.y[jnp.arange(i_age,(2*N_S+1)*NAG,NAG),:],axis=0) for i_age in range(NAG)])
# @jit
# def proportion_infected(t,SP,ap,AR,rt=result.t):
#     """Calculate the proportion of the population that is infected at time t."""
#     idx = jnp.searchsorted(rt, t)
#     return I[:,idx]/age_pops[:,idx]

# @jit
# def contact2(t,seasonality,offset):
#     """Contact function where infected persons don't make contacts"""
#     eff_CONTACT = CONTACT.copy()
#     for i in range(NAG):
#         eff_CONTACT = eff_CONTACT.at[i,:].set(CONTACT[i,:] * (1 - 0.3 * proportion_infected(t, S_REL, S_VAX, ACOV)[i]))
#     return (1+seasonality*jnp.cos(2*jnp.pi*((t-274)/365-offset)))*eff_CONTACT
# @jit
# def contact3(t,seasonality,offset):
#     """Contact function where infected persons don't make contacts"""
#     eff_CONTACT = CONTACT.copy()
#     for i in range(NAG):
#         eff_CONTACT = eff_CONTACT.at[i,:].set(CONTACT[i,:] * (1 - 0.1 * proportion_infected(t, S_REL, S_VAX, ACOV)[i]))
#     return (1+seasonality*jnp.cos(2*jnp.pi*((t-274)/365-offset)))*eff_CONTACT

# params2 = params.copy()
# params2["OFFSET"] = 0.3
# start = time.time()
# result2 = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params2.values(),t_eval=POINTS,method='RK45')
# print("result2 took ",time.time()-start," seconds")
# params3 = params.copy()
# params3["OFFSET"] = 0.4
# start = time.time()
# result3 = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params3.values(),t_eval=POINTS,method='RK45')
# print("result3 took ",time.time()-start," seconds")

# params2_crossimmunity = params2.copy()
# params2_crossimmunity["ACOV"] = proportion_infected
# start = time.time()
# result2_crossimmunity = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params2_crossimmunity.values(),t_eval=POINTS,method='RK45')
# print("result2 took ",time.time()-start," seconds")

# params2_interference = params2.copy()
# params2_interference["contact"] = contact2
# start = time.time()
# result2_interference = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params2_interference.values(),t_eval=POINTS,method='RK45')
# print("result2 took ",time.time()-start," seconds")

# params3_interference = params3.copy()
# params3_interference["contact"] = contact3
# start = time.time()
# result3_interference = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params3_interference.values(),t_eval=POINTS,method='RK45')
# print("result3 took ",time.time()-start," seconds")

# params3_behaviour = params3.copy()
# params3_behaviour["contact"] = contact2
# start = time.time()
# result3_behaviour = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params3_behaviour.values(),t_eval=POINTS,method='RK45')
# print("result2 took ",time.time()-start," seconds")

# # # # # vaccination_proportion = pd.read_csv('Data/Processed/KPSC_vaccinated_proportion_ages_monthly.csv',index_col=0)
# # # # # hsv_colors = colormaps.hsv(-0.02+jnp.arange(7)/7)
# # # # # hsv_colors[3] = colormaps.hsv((3/7)+0.04)
# # # # # pop_size_by_age = jnp.array([jnp.sum(result.y[range(i_age,(N_C*N_S+1)*NAG,NAG),:],axis=0) for i_age in range(NAG)]).T
# # # # # for i in range(NAG): 
# # # # #     ax[0].plot(result.t,[flu_eff_coverage(t,S_REL)[i] for t in result.t],label=AGE_GROUP_NAMES[i],color=hsv_colors[i])
# # # # #     ax[1].plot(result.t,result.y[(2*(N_S-1)+1)*NAG+i,:].T/pop_size_by_age[:,i],label=AGE_GROUP_NAMES[i],color=hsv_colors[i])
# # # # # # ax[0].plot(result.t,[jnp.sum([flu_eff_coverage(t,S_REL)[i]*pop_size_by_age[:,i] for i in range(NAG)],axis=0)/jnp.sum(pop_size_by_age,axis=1) for t in result.t],label='Effective coverage',color="#648FFF")
# # # # # # ax[1].plot(result.t,jnp.sum(result.y[(2*(N_S-1)+1)*NAG:(2*(N_S-1)+2)*NAG,:],axis=0)/jnp.sum(pop_size_by_age,axis=1),color="#648FFF")
# # # # # plt.savefig('Figures/flu_vaccination_coverage_monthly_no_age_correction.png',dpi=300)
obs = observations(result.y,POINTS,params,OBS_AGE,incidence=False,time_conversion=30.44)
# print(obs)
fig, ax = plt.subplots()
mx = lockdown_incidence_plot(ax,STATE0,params,OBS_AGE,PERIOD,POINTS,date_to_t('2024-03-19'),365,result=result,obs=obs,label="Simulation",by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs)
plt.savefig('Figures/old_model_example.png',dpi=300)
# obs2 = observations(result2,params2,OBS_AGE,incidence=True,time_conversion=30.44)
# obs2_crossimmunity = observations(result2_crossimmunity,params2_crossimmunity,OBS_AGE,incidence=True,time_conversion=30.44)
# obs2_interference = observations(result2_interference,params2_interference,OBS_AGE,incidence=True,time_conversion=30.44)
# obs3_interference = observations(result3_interference,params3_interference,OBS_AGE,incidence=True,time_conversion=30.44)
# obs3 = observations(result3,params3,OBS_AGE,incidence=True,time_conversion=30.44)
# obs3_behaviour = observations(result3_behaviour,params3_behaviour,OBS_AGE,incidence=True,time_conversion=30.44)

# fig, ax = plt.subplots(2,2,figsize=(6.5,4.5),sharex=True,sharey=True)
# # panel 0 0 - no interaction
# mx = lockdown_incidence_plot(ax[0,0],STATE0,params,OBS_AGE,PERIOD,POINTS,date_to_t('2024-03-19'),365,result=result,obs=obs,label="Simulation",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs)
# mx2 = lockdown_incidence_plot(ax[0,0],STATE0,params2,OBS_AGE,PERIOD,POINTS,date_to_t('2024-03-19'),365,result=result2,obs=obs2,label="Simulation 2",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs,color='#DC267F')
# mx3 = lockdown_incidence_plot(ax[0,0],STATE0,params3,OBS_AGE,PERIOD,POINTS,date_to_t('2024-03-19'),365,result=result3,obs=obs3,label="Simulation 3",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs,color='#FFB000')
# # mx00 = jnp.max(jnp.array((mx, mx2, mx3)))
# # lockdown_incidence_format(ax[0,0],date_to_t('2024-03-19'),365,mx00,year_window=2)
# # panel 0 1 - cross-immunity
# mx = lockdown_incidence_plot(ax[0,1],STATE0,params,OBS_AGE,PERIOD,POINTS,date_to_t('2024-03-19'),365,result=result,obs=obs,label="Simulation",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs)
# mx2 = lockdown_incidence_plot(ax[0,1],STATE0,params2_crossimmunity,OBS_AGE,PERIOD,POINTS,date_to_t('2024-03-19'),365,result=result2_crossimmunity,obs=obs2_crossimmunity,label="Simulation 2",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs,color='#DC267F')
# mx3 = lockdown_incidence_plot(ax[0,1],STATE0,params3,OBS_AGE,PERIOD,POINTS,date_to_t('2024-03-19'),365,result=result3,obs=obs3,label="Simulation 3",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs,color='#FFB000')
# # mx01 = jnp.max(jnp.array((mx, mx2, mx3)))
# # lockdown_incidence_format(ax[0,1],date_to_t('2024-03-19'),365,mx01,year_window=2)
# # panel 1 0 - interference
# mx = lockdown_incidence_plot(ax[1,0],STATE0,params,OBS_AGE,PERIOD,POINTS,date_to_t('2024-03-19'),365,result=result,obs=obs,label="Simulation",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs)
# mx2 = lockdown_incidence_plot(ax[1,0],STATE0,params2_interference,OBS_AGE,PERIOD,POINTS,date_to_t('2024-03-19'),365,result=result2_interference,obs=obs2_interference,label="Simulation 2",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs,color='#DC267F')
# mx3 = lockdown_incidence_plot(ax[1,0],STATE0,params3_interference,OBS_AGE,PERIOD,POINTS,date_to_t('2024-03-19'),365,result=result3_interference,obs=obs3_interference,label="Simulation 3",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs,color='#FFB000')
# # mx10 = jnp.max(jnp.array((mx, mx2, mx3)))
# # lockdown_incidence_format(ax[1,0],date_to_t('2024-03-19'),365,mx10,year_window=2)
# # panel 1 1 - behaviour change
# mx = lockdown_incidence_plot(ax[1,1],STATE0,params,OBS_AGE,PERIOD,POINTS,date_to_t('2024-03-19'),365,result=result,obs=obs,label="Simulation",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs)
# mx2 = lockdown_incidence_plot(ax[1,1],STATE0,params2_interference,OBS_AGE,PERIOD,POINTS,date_to_t('2024-03-19'),365,result=result2_interference,obs=obs2_interference,label="Simulation 2",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs,color='#DC267F')
# mx3 = lockdown_incidence_plot(ax[1,1],STATE0,params3_behaviour,OBS_AGE,PERIOD,POINTS,date_to_t('2024-03-19'),365,result=result3_behaviour,obs=obs3_behaviour,label="Simulation 3",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,p_time_to_obs=p_time_to_obs,color='#FFB000')
# # lockdown_incidence_format(ax[1,1],date_to_t('2024-03-19'),365,mx10,year_window=2)
# # # ax.plot(POINTS,[10*cm.piecewise(pt,Ts,Fs) for pt in POINTS],label='Mobility',color='k')
# # # ax.plot(POINTS,[100*regional_positivity(pt) for pt in POINTS],label='Positivity',color='k',linestyle='--',alpha=0.5)
# # # # # for i in range(NAG):
# # # # #     ax.plot(POINTS,[jnp.sum(contact(t,params["SEASONALITY"],params["OFFSET"]),axis=1)[i] for t in POINTS],c=hsv_colors[i],linestyle='--',dashes=(1,0.5+i/NAG),label=AGE_GROUP_NAMES[i])

# ax[0,0].set_yticks([])
# ax[0,0].set_xticks([])


# # write a capital A in the top left corner of the first panel
# ax[0,0].text(0.01, 0.98, 'A', transform=ax[0,0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
# ax[0,1].text(0.01, 0.98, 'B', transform=ax[0,1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
# ax[1,0].text(0.01, 0.98, 'C', transform=ax[1,0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
# ax[1,1].text(0.01, 0.98, 'D', transform=ax[1,1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# plt.tight_layout()
# plt.savefig('Figures/interaction_demonstration_jaxtest.png',dpi=300)

# # params["WANE"] = jnp.array([0,1.195e-01,0])/365
# # params["P_OBS"] = 3.396e-02*jnp.array([1,0.46,0.31])
# # OBS_AGE = jnp.array([1,0.8229,0.6458,0.4687,0.2916,0.1375,0.8626])
# # params["SEASONALITY"] = 1.727e-01
# # params["OFFSET"] = 8.584e-01
# # params["BETA"] = 6.983e-02

# pms = [2.547e-02,1.656e-01
# ,8.209e-01,1.086e-01,1.238e-12,2.627e-02,3.167e-01,7.424e-01,5.960e-01]
# # mcmc value
# pms = [8.17814276e-05,1.51485052e-01,8.12003008e-01,1.14221137e-01
# ,1.53783669e-12,3.67742977e-02
# ,3.17488852e-01,7.53475678e-01,5.81731678e-01]
# # # print(likelihood(pms))


# result2 = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params.values(),t_eval=POINTS,method='RK45')
# obs2 = observations(result2,params,OBS_AGE,incidence=False,time_conversion=30.44)
# mx2 = lockdown_incidence_plot(ax[1],STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result2,obs=obs2,label="Fit",by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,color='#DC267F',p_time_to_obs=p_time_to_obs)
# lockdown_incidence_format(ax[1],T_LOCKDOWN,LOCKDOWN_DURATION,mx2,year_window=2)
# plt.tight_layout()
# plt.savefig('Figures/RSV_fit_test.png',dpi=300)

# ax[1].set_ylabel('Simulated incidence per 10k')
# ax[2].set_ylabel('Simulated incidence per 10k')
# ax[1].set_xlim(POINTS[89],POINTS[-1])
# ax[2].set_xlim(POINTS[89],POINTS[-1])
# # # # # # copy y limit from ax[1]
# # # ax[2].set_ylim(ax[1].get_ylim())

# ax[1].set_title('Hand-tuned simulation')
# ax[2].set_title('Maximum likelihood simulation')

# plt.tight_layout()

# plt.savefig('Figures/RSV_compare_fits_mcmc_trajectory_from_NM_test.png',dpi=300)

# params["BETA"] = 8.710e-02
# params["P_OBS"] = 1.535e-02*jnp.ones(N_S)
# params["SEASONALITY"] = 9.969e-02
# params["OFFSET"] = -1.369e-01
# params["WANE"] = 1/30*jnp.array([0.0,3.918e-01,0.0])/365
# OBS_AGE = age_detection(NAG,2.418,8.514e-01,4.772e-01)
# result2 = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params.values(),t_eval=POINTS,method='RK45')
# obs2 = observations(result2,params,OBS_AGE,incidence=False,time_conversion=1)

# # # # obs = jnp.sum(obs,axis=1)
# # # noisy_obs = jnp.random.poisson(obs)
# # # pop_size_by_age = jnp.array([jnp.sum(result.y[range(i_age,(2*N_S+1)*NAG,NAG),:],axis=0) for i_age in range(NAG)]).T
# # # noisy_incidence = noisy_obs/pop_size_by_age
# # # with open("Data/Processed/SIS_noisy_obs_daily_BETAp08_SEASONALITYp1_OFFSETp85_WANE10y_POBSp01_OBSAGEp2p85p1.pickle","wb") as f:
# # #     pickle.dump(noisy_incidence,f)
# # fig, ax = plt.subplots(2,1,figsize=(6.5,8.5))
# # mx = lockdown_incidence_plot(ax[0],STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,obs=obs,label="Simulation",by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000)
# mx = lockdown_incidence_plot(axes[1],STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,obs=obs,by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,color='#DC267F')
# # # plot monthly moving average of noisy incidence
# # moving_average = jnp.array([jnp.mean(noisy_incidence[i-30:i],axis=0)*10000 for i in range(30,len(noisy_incidence))])
# # # plot the age stratified incidence in shades of grey
# # # define grayscale color map
# hsv_colors = colormaps.hsv(-0.02+jnp.arange(7)/7)
# hsv_colors[3] = colormaps.hsv((3/7)+0.04)
# # for i in range(NAG):
# #     ax[0].plot(POINTS[30:],moving_average[:,i],color=hsv_colors[i],linestyle='--')
# #     ax[1].plot(POINTS[30:],moving_average[:,i],color=hsv_colors[i],linestyle='--')
# # # mx2 = lockdown_incidence_plot(ax,STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,obs=noisy_incidence,label="Simulation",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000,color='#DC267F')
# # lockdown_incidence_format(ax[0],T_LOCKDOWN,LOCKDOWN_DURATION,mx,year_window=2)
# lockdown_incidence_format(axes[1],T_LOCKDOWN,LOCKDOWN_DURATION,mx,year_window=2)
# # # ax.legend()
# # plt.show()
# # ax[0].set_title('Simulation with noisy observations')
# # ax[1].set_title('Fit to noisy observations')
# # ax[0].set_ylabel('Simulated incidence per 10k')
# # ax[1].set_ylabel('Simulated incidence per 10k')
# # plt.savefig('Figures/test.png',dpi=300)
# # # axes[1].legend()
# axes[1].set_ylabel('Simulated incidence per 10k')
# # plot google_prestige_work
# axes2 = axes[1].twinx()
# # print([cm.google_prestige_work(point) for point in POINTS])
# axes2.plot(POINTS,[cm.piecewise(point,Ts,Fs) for point in POINTS],color='black',linestyle='--',label='Google workplace mobility')
# axes[1].legend(axes[1].lines,['<1y','1-4y','5-17y','18-39y','40-64y','>=65y'],loc='upper left',title='Age group')
# # axes[1].vlines(18952,0,mx,linestyle=':',color='black')
# axes[1].set_xlim(POINTS[0],POINTS[-1])
# @jit
# def contact(t,seasonality,offset):
#     return cm.google_prestige_work(t)*(1+seasonality*jnp.cos(2*jnp.pi*(t/365-offset)))*CONTACT
# params['contact'] = contact
# result = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params.values(),t_eval=POINTS,method='RK45')
# obs = observations(result,params,OBS_AGE,incidence=False,time_conversion=30.44)
# mx = lockdown_incidence_plot(axes[2],STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,obs=obs,label="Simulation",by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=10000)
# lockdown_incidence_format(axes[2],T_LOCKDOWN,LOCKDOWN_DURATION,mx,year_window=2,title="Simulation with workplace mobility contact model")
# axes[2].set_ylabel('Simulated incidence per 10k')
# axes3 = axes[2].twinx()
# axes3.plot(POINTS,[cm.google_prestige_work(point) for point in POINTS],color='black',linestyle='--',label='Google workplace mobility')
# # axes[2].legend(axes[2].lines,['<1y','1-4y','5-17y','18-39y','40-64y','>=65y'],loc='upper left',title='Age group')   
# axes[2].set_xlim(POINTS[0],POINTS[-1])
# plt.tight_layout()
# # plt.savefig('Figures/8param_fit_test.png',dpi=300)
# plt.savefig('Figures/RSV_mobility_tests.png',dpi=300)

# # with open("Data/Processed/SIS_noisy_obs_BETAp15_SEASONALITYp06_OFFSETp1_WANE10y_ADRp1.pickle","rb") as f:
# #     incidence = pickle.load(f)

# # # # # # case_data = pd.read_csv('Data/Processed/KPSC_rsv_hosp_incidence_by_age.csv')
# # # # # # # case_data = pd.read_csv('Data/Processed/KPSC_flu_hosp.csv')['Count']
# # # # # # case_data = jnp.array(case_data)
# # # # # # # print(case_data)

# # # # # points_trimmed = jnp.array(date_to_t(PERIOD[PERIOD < '2020-01-01']))

# # params["WANE"] = jnp.array([0.0,pms[0],0.0])/365
# # OBS_AGE = age_detection(NAG,pms[1],pms[2],pms[3])
# # params["SEASONALITY"] = pms[4]
# # params["OFFSET"] = pms[5]
# # params["BETA"] = pms[6]
# # params["IMPORT_RATE"] = pms[7]
# # params["P_OBS"] =  pms[8]*jnp.array([1,0.46,0.31])


# jnp.random.seed(250226)
# variables = ["WANE","SEASONALITY","OFFSET","BETA","IMPORT_RATE","EXTRA_IMMUNITY","FIRST_IMMUNITY","FIRST_DIS_INF_FACTOR","P_OBS","OBS_AGE_YOUNG","OBS_AGE_OLD","OBS_AGE_YOUNG_OLD"]
# initial_scalars = jnp.array([x])
# log_priors_distribution = sp.stats.multivariate_normal([-1]*12,jnp.diag([1.5]*12))
# log_priors = lambda y : log_priors_distribution.logpdf(y)
# proposal_cov = jnp.diag([1e-4]*12)
# mcmc_trajectory, acceptance_rate, likelihoods = mcmc(incidence, params, POINTS, STATE0, OBS_AGE, SIS_likelihood, p_time_to_obs, variables, initial_scalars, log_priors, proposal_cov, 200, True, True)
# print(acceptance_rate)
# print(jnp.mean(mcmc_trajectory,axis=0))
# plt.plot(mcmc_trajectory)
# plt.savefig('Figures/mcmc_trajectory_from_FluA_DE.png',dpi=300)
# with open('Data/Processed/mcmc_trajectory_from_FluA_DE.pickle','wb') as f:
#     pickle.dump(mcmc_trajectory,f)
# with open('Data/Processed/mcmc_likelihoods_from_FluA_DE.pickle','wb') as f:
#     pickle.dump(likelihoods,f)

# with open('Data/Processed/mcmc_trajectory_from_FluA_DE.pickle','rb') as f:
#     mcmc_trajectory = pickle.load(f)
# principal_componets = decomposition.PCA(n_components=2)
# X = principal_componets.fit_transform(mcmc_trajectory)
# fig, ax = plt.subplots(1,1,figsize=(6.5,6.5))
# ax.plot(X[:,0],X[:,1],color='silver')
# # ax.scatter(X[:,0],X[:,1],c=likelihoods,cmap='viridis')
# plt.savefig('Figures/PCA_mcmc_trajectory_from_FluA_DE.png',dpi=300)


# # plt.show()
# with open('Data/Processed/mcmc_trajectory_multivariate.pickle','rb') as f:
#     mcmc_trajectory = pickle.load(f)
# # with open('Data/Processed/mcmc_trajectory_pre_lockdown.pickle','rb') as f:
# #     mcmc_trajectory_pre_lockdown = pickle.load(f)
# plt.plot(mcmc_trajectory)
# plt.show()

# print(jnp.mean(mcmc_trajectory,axis=0))

# burn_in = 1000

# # plt.plot(mcmc_trajectory)
# # plt.show()

# # figure = corner.corner(mcmc_trajectory[burn_in:],labels=['Transmissibility','Seasonality','Immunity'])
# # plt.savefig('Figures/SIS_MCMC_corner.png',dpi=300)

# fig = plt.figure(figsize=(13.3,7.5),layout='constrained')
# gs = GridSpec(2,3,figure=fig)
# trajectory_ax = fig.add_subplot(gs[0,:])
# heat_axes = jnp.array([fig.add_subplot(gs[1,i]) for i in range(3)])
# heat_axes[0].hist2d(mcmc_trajectory[burn_in:,0],mcmc_trajectory[burn_in:,1],bins=20)
# # outline the cell corresponding to the true values - 0.06 and 0.15 - in red
# heat_axes[0].plot([0.15,0.15,0.1501,0.1501,0.15],[0.05995,0.06005,0.06005,0.05995,0.05995],color='red')
# heat_axes[0].set_xlabel('Transmissibility')
# heat_axes[0].set_ylabel('Seasonality')
# heat_axes[1].hist2d(mcmc_trajectory[burn_in:,0],mcmc_trajectory[burn_in:,2],bins=20)
# heat_axes[1].plot([0.15,0.15,0.1501,0.1501,0.15],[0.39995,0.40005,0.40005,0.39995,0.39995],color='red')
# heat_axes[1].set_xlabel('Transmissibility')
# heat_axes[1].set_ylabel('Immunity')
# heat_axes[2].hist2d(mcmc_trajectory[burn_in:,1],mcmc_trajectory[burn_in:,2],bins=20)
# heat_axes[2].plot([0.05995,0.05995,0.06005,0.06005,0.05995],[0.39995,0.40005,0.40005,0.39995,0.39995],color='red')
# heat_axes[2].set_xlabel('Seasonality')
# heat_axes[2].set_ylabel('Immunity')
# trajectory_ax.plot(mcmc_trajectory[:,0],label='Transmissibility')
# trajectory_ax.plot(mcmc_trajectory[:,1],label='Seasonality')
# trajectory_ax.plot(mcmc_trajectory[:,2],label='Immunity')
# trajectory_ax.legend()
# trajectory_ax.set_xlabel('Iteration')
# trajectory_ax.set_ylabel('Parameter value')
# # plt.tight_layout()
# plt.savefig('Figures/example_MCMC_trajectory_multivariate.png',dpi=300)

# # ###### cumulative cases seasonal plot ######
# fig, ax = plt.subplots(2,6,figsize=(13.3,7.5),sharex=True)
# pathogens = ["RSV","Influenza_A","Influenza_B","Metapneumovirus","Adenovirus","Parainfluenza_3"]
# for i,pathogen in enumerate(pathogens):
#     season_plot(ax[0,i],pathogen,incidence=False,relative=False)
#     season_plot(ax[1,i],pathogen,incidence=False,relative=True)
#     ax[0,i].set_title(pathogen.replace("_"," "))
#     ax[1,i].set_ylim(0,1)
# ax[0,0].set_ylabel("Cases")
# ax[0,0].set_xticks(range(0,8),["2015/16","2016/17","2017/18","2018/19","2019/20","2020/21","2021/22","2022/23"])
# ax[1,0].set_ylabel("Age group share")
# # add space at bottom for legend
# fig.subplots_adjust(bottom=0.2)
# # add horizontal space between plots
# fig.subplots_adjust(hspace=0.4)
# # horizontal legend at bottom of figure for age groups
# fig.legend(["<3m","3–11m","1–4y","5–17y","18–39y","40–64y",">=65y"],loc='lower center',bbox_to_anchor=(0.5,0),ncol=7,frameon=False)

# # plt.tight_layout()
# plt.savefig("Figures/cumulative_seasons.png",dpi=300)

# # # ####### Computing observations ########
# with open('Data/Processed/SIS_3D_power.pickle','rb') as f:
#     results = pickle.load(f)
# with open('Data/Processed/SIS_3D_power_params.pickle','rb') as f:
#     param_dict = pickle.load(f)
# obses = {}
# for key,result in results.items():
#     if jnp.all(jnp.array(key[1:]) == 0):
#         print(key)
#     params = param_dict[key]
#     # params['contact'] = lambda t, seasonality, offset : contact(t,shape_static,seasonality,offset)
#     obs = observations(result,params,OBS_AGE,incidence=True)
#     obses[key] = obs
# with open('Data/Processed/SIS_3D_power_obs.pickle','wb') as f:
#     pickle.dump(obses,f)

# # # ######## Plotting clusters ########
# with open('Data/Processed/SIS_3D_little.pickle','rb') as f:
#     results = pickle.load(f)
# with open('Data/Processed/SIS_3D_little_obs.pickle','rb') as f:
#     obses = pickle.load(f)

# # #### plot age-based clusters
# ages = jnp.array([jnp.mean(age_of_first_infection(results[key],MEDIAN_AGE)[jnp.argmax(results[key].t>=T_LOCKDOWN-5*12):jnp.argmax(results[key].t>T_LOCKDOWN)]) for key in results.keys()])

# print(jnp.sum(ages<=3))
# ages_label = jnp.array([int(age>3) + int(age>12) + int(age>5*12) + int(age>18*12) + int(age>40*12) + int(age>65*12) for age in ages])
# n_age_clusters = max(ages_label)+1
# print(n_age_clusters)

# fig,axes = plt.subplots(n_age_clusters,4,figsize=(6.5,1.7*n_age_clusters),sharex='col',layout='constrained',squeeze=False)
# for row in range(n_age_clusters):
#     axes[row,1].sharey(axes[row,2])
#     axes[row,2].sharey(axes[row,3])
# cluster_plot(axes,results,obses,n_age_clusters,ages_label,None,color=False,line=True,N=15)
# fig.align_ylabels()
# plt.savefig('Figures/SIS_3D_2_age_clusters.png',dpi=500)

# # # # ## plot n clusters
# n_clusters = 3
# model = cluster_sims(results,obses,T_LOCKDOWN,n_clusters,width=5)
# fig, axes = plt.subplots(n_clusters,4,figsize=(6.5,1.2*n_clusters),sharex='col',layout='constrained',squeeze=False)
# for row in range(n_clusters):
#     axes[row,1].sharey(axes[row,2])
#     axes[row,2].sharey(axes[row,3])
# cluster_plot(axes,results,obses,model.n_clusters,model.labels_,model.cluster_centers_,color=True,line=False,
# base_values=[1/10,1/913,1/4],factors=[0.7,3,1],grid_mode=("scale","scale","based_vec"),
# N=20)
# fig.align_ylabels()
# plt.savefig('Figures/SIS_3D_little_'+str(n_clusters)+'clusters.png',dpi=500)

# ### plot all sims and select clusters
# model = cluster_sims(results,obses,T_LOCKDOWN,6)
# model1 = cluster_sims(results,obses,T_LOCKDOWN,1)
# ages = jnp.array([jnp.mean(age_of_first_infection(results[key],MEDIAN_AGE)[jnp.argmax(results[key].t>=T_LOCKDOWN-5*12):jnp.argmax(results[key].t>T_LOCKDOWN)]) for key in results.keys()])
# ages_sd = jnp.array([jnp.mean(age_of_first_infection(results[key],MEDIAN_AGE,sd=True)[1][jnp.argmax(results[key].t>=T_LOCKDOWN-5*12):jnp.argmax(results[key].t>T_LOCKDOWN)]) for key in results.keys()])

# def hsv_to_hex(h, s, v):
#     r, g, b = colorsys.hsv_to_rgb(h, s, v)
#     return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))

# age_colors = jnp.array([hsv_to_hex(mu/jnp.max(ages),1-sd/jnp.max(ages_sd), 0.9) for mu,sd in zip(ages,ages_sd)])

# fig, axes = plt.subplots(4,4,figsize=(6.5,1.7*4),sharex='col',layout='constrained',squeeze=False)
# for row in range(4):
#     axes[row,1].sharey(axes[row,2])
#     axes[row,2].sharey(axes[row,3])
# first_row = axes[0,:]
# first_row.shape = (1,4)
# # cluster_colors = jnp.array([["#648FFF", "#DC267F", "#785EF0", "#FFB000", "#FF832B", "#000000", "#FFD662", "#FF34FF", "#8B0000", "#00FF00"][i] for i in model.labels_])
# cluster_plot(first_row,results,obses,model1.n_clusters,model1.labels_,None,color=True,line=False,color_values_all=age_colors)
# cluster_plot(axes[1:,:],results,obses,model.n_clusters,model.labels_,model.cluster_centers_,color=True,clusters=[0,2,4],line=False,color_values_all=age_colors)
# fig.align_ylabels()
# plt.savefig('Figures/SIS_3D_clusters_1plus3of6_age_color.png',dpi=900)

# # ## Computing observations for a given cluster
# with open('Data/Processed/SIS_3D_based.pickle','rb') as f:
#     results = pickle.load(f)
# pick_cluster = 0
# # separate out results and observations from cluster pick and save
# cluster_pick_results = {}
# cluster_pick_obses = {}
# for key,result in results.items():
#     if model.labels_[list(results.keys()).index(key)] == pick_cluster:
#         cluster_pick_results[key] = result
#         cluster_pick_obses[key] = obses[key]
# with open('Data/Processed/SIS_3D_based_cluster'+str(pick_cluster+1)+'_results.pickle','wb') as f:
#     pickle.dump(cluster_pick_results,f)
# with open('Data/Processed/SIS_3D_based_cluster'+str(pick_cluster+1)+'_obses.pickle','wb') as f:
#     pickle.dump(cluster_pick_obses,f)

# ## Plot sub-clusters
# pick_cluster = 0
# # # ## load cluster and divide into more clusters
# with open('Data/Processed/SIS_3D_based_cluster'+str(pick_cluster+1)+'_results.pickle','rb') as f:
#     results = pickle.load(f)
# with open('Data/Processed/SIS_3D_based_cluster'+str(pick_cluster+1)+'_obses.pickle','rb') as f:
#     obses = pickle.load(f)

# n_clusters = 5
# model = cluster_sims(results,obses,T_LOCKDOWN,n_clusters)
# with open('Data/Processed/SIS_3D_based_'+str(n_clusters)+'clusters_of_cluster'+str(pick_cluster+1)+'of3.pickle','wb') as f:
#     pickle.dump(model,f)
# fig, axes = plt.subplots(n_clusters,4,figsize=(6.5,1.1*n_clusters),sharex='col',layout='constrained',squeeze=False)
# for row in range(n_clusters):
#     axes[row,1].sharey(axes[row,2])
#     axes[row,2].sharey(axes[row,3])
# cluster_plot(axes,results,obses,model.n_clusters,model.labels_,model.cluster_centers_,color=False,line=True,y_value="rebound peak incidence")
# fig.align_ylabels()
# plt.savefig('Figures/SIS_3D_based_'+str(n_clusters)+'clusters_of_cluster'+str(pick_cluster+1)+'of3_rebound_size.png',dpi=300)

# #### line plots with different parameter values, showing incidence and susceptibility #####
# # params['BETA'] = 70
# # params['SEASONALITY'] = 0.1737
# # params['OFFSET'] = 0.875
# # params['S_REL'] = jnp.array([1,0.8,0.6])
# fig, ax = plt.subplots(2,1,figsize=(13.3,7.5),sharey=True)
# kpsc_positive_test_plot(ax[0],pathogen="RSV", incidence=True, legend=False,aggregation="Month")
# mx = jnp.zeros(4)
# # four viridis colours
# colors = colormaps.viridis(jnp.linspace(0,1,4))
# # Plot the incidence
# for i in range(4):
#     p = [0.16,0.165,0.17,0.175][i]
#     params['BETA'] = p
#     start = time.time()
#     result = sp.integrate.solve_ivp(sis_deltas,(date_to_t(EPOCH),POINTS[-1]),STATE0,args=params.values(),t_eval=POINTS,method='RK45')
#     print(f"Simulation took {time.time()-start:.2f} seconds")
# #     ax[i//2,i%2].plot(result.t[1:],jnp.sum(result.y[NAG:2*NAG,:],axis=0)[1:],label='S1',color=colors[0])
# #     ax[i//2,i%2].plot(result.t[1:],jnp.sum(result.y[3*NAG:4*NAG,:],axis=0)[1:],label='S2',color=colors[1])
# #     ax[i//2,i%2].plot(result.t[1:],jnp.sum(result.y[5*NAG:6*NAG,:],axis=0)[1:],label='S3',color=colors[2])
# #     ax[i//2,i%2].set_title(f"Aquired immunity: {0.1*(i+2):.2f}")
# # ax[0,0].legend()
# # plt.show()
#     obs = observations(result,params,OBS_AGE,incidence=True,time_conversion=30.44)*10000    # plot each susceptible compartment
#     # pre_obs = obs[(result.t>T_LOCKDOWN-12*12) & (result.t<T_LOCKDOWN)]
#     # corr = jnp.correlate(pre_obs, pre_obs, mode='same')
#     # acorr = corr[len(pre_obs)//2 + 1:] / (pre_obs.var() * jnp.arange(len(pre_obs)-1, len(pre_obs)//2, -1))
#     # acorr = acorr + jnp.linspace(0.1, 0, len(acorr))
#     # lag = jnp.abs(acorr).argmax() + 1
#     # print(lag)
#     mx[i] = lockdown_incidence_plot(ax[1],STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,label=f'{p:.4f}',color=colors[i],relative=False,obs=obs)
#     # lockdown_susceptibility_plot(ax[1],STATE0,params,PERIOD,POINTS,T_LOCKDOWN,result=result,label=f'{p:.4f}',color=colors[i],relative=False)

# lockdown_incidence_format(ax[1],T_LOCKDOWN,LOCKDOWN_DURATION,max(mx),year_window=10)
# ax[1].set_xlim(POINTS[89],POINTS[-1])
# # three minor ticks between x ticks
# ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))
# # lockdown_susceptibility_format(ax[1],T_LOCKDOWN,LOCKDOWN_DURATION,ymax=None,year_window=10)
# # ax[1].set_ylabel("Observed incidence")
# # ax[0].set_ylim(0,1.1)
# ax[1].set_ylabel("Modelled incidence per 10k")
# # ax[1].set_ylim(1.5e7,2.2e7)
# ax[1].legend(title="Transm.")
# plt.tight_layout()
# plt.savefig('Figures/test.png',dpi=300)

# # ##### run multi-dimensional GRID SIMS #####
# start = time.time()
# param_dict, results = sim_grid(STATE0,params,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,
# grid_params=(("BETA",),("WANE",),("S_REL",)),
# N=2,factors=(0.7,3,1),grid_mode=("scale","scale","based_vec"))
# print(f"Simulation took {time.time()-start:.2f} seconds")
# # Save the results
# with open('Data/Processed/SIS_3D_little.pickle','wb') as f:
#     pickle.dump(results,f)
# with open('Data/Processed/SIS_3D_little_params.pickle','wb') as f:
#     pickle.dump(param_dict,f)

# ##### Plotting multi-dimensional grid sims #####
# # Load the results
# with open('Data/Processed/SIS_4D.pickle','rb') as f:
#     results = pickle.load(f)

# fig, axes = plt.subplots(2,3,figsize=(6.5,4.5),layout='constrained')
# start = time.time()
# im = grid_plot(axes,results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE,
# grid_params=(("BETA",),("WANE",),("S_REL",),("SEASONALITY",)),
# factors=(1,1,1,2),
# grid_mode=("scale","scale","fade_vec","scale"),
# label_mode=("mean","nz_mean","fade_vec","mean"),
# x_labels=("Transmission","Waning","Acquired immunity","Seasonality"),
# z_value="time to rebound",z_label="",
# fix=True,vmin=0,vmax=5)
# print(f"Plotting took {time.time()-start:.2f} seconds")
# # colorbar
# fig.colorbar(im, ax=axes, orientation='horizontal', label="Time to rebound (years)")
# # plt.tight_layout()
# # plt.show()
# plt.savefig('Figures/SIS_4D_rebound_time_to_rebound_fix.png',dpi=300)

##### Grid plots of oscillation size and peak incidence #####
# fig, ax = plt.subplots(2,1,figsize=(6.5,6.5))
# grid_plot(ax[0],results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE,
# grid_params=(("BETA",),("REC_UP","REC_SAME")),label_mode=("mean","nz_mean"),
# z_value="oscillation size",z_label="Oscillation size")
# grid_plot(ax[1],results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE,
# grid_params=(("BETA",),("REC_UP","REC_SAME")),label_mode=("mean","nz_mean"),
# z_value="peak incidence",z_label="Peak observed incidence")
# plt.show()

##### Grid plots of child infections, incidence, perodicity, time to rebound, susceptibility #####
# # # measure time to get results
# # start = time.time()
# # results = sim_grid(STATE0,params,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,N=10
# # ,grid_params=(("BETA","REC_UP","REC_SAME"),("S_REL",)),grid_mode=("scale","fade_vec"),factors=(1,1))
# # # save results
# # with open('Data/Processed/test_adult_SIS.pkl','wb') as f:
# #     pickle.dump(results,f)
# # end = time.time()
# # print('Time to get results:',end-start)
# # # load results
# with open('Data/Processed/test_adult_SIS.pkl','rb') as f:
#     results = pickle.load(f)
# fig, axes = plt.subplots(4,2,figsize=(6.5,8.5))
# z_values = [["child infections","under-five infections"],
# ["peak incidence","rebound peak incidence"],
# ["periodicity","time to rebound"],
# ["pre-lockdown susceptibility","post-lockdown susceptibility"]]
# titles = [["Child-caused\ninfections","Under-five-caused\ninfections"],
# ["Peak incidence","Rebound peak"],
# ["Periodicity","Time to rebound"],
# ["Pre-lockdown\nsusceptibility","Post-lockdown\nsusceptibility"]]
# cbar_labels = [["","Infections"],
# ["","Observed infections"],
# ["","Years"],
# ["","Susceptibility"]]
# for i in range(4):
#     for j in range(2):
#         grid_plot(axes[i,j],results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE=OBS_AGE,
#         grid_params=(("BETA","REC_UP"),("S_REL",)),grid_mode=("scale","fade_vec"),factors=(1,1),
#         label_mode=("diff_mean","fade_vec"),
#         z_value=z_values[i][j],z_label=cbar_labels[i][j])
#         axes[i,j].set_title(titles[i][j])
#         if i == 3:
#             axes[i,j].set_xlabel("Immunity")
#         if j == 0:
#             axes[i,j].set_ylabel("Growth rate")
# plt.tight_layout()
# plt.show()