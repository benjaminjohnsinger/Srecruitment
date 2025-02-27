import numpy as np
import scipy as sp
import pandas as pd 
import matplotlib.pyplot as plt
import corner
import pickle
from numba import jit
from sklearn import decomposition

from vaccination import birth_vax, birth_vax, all_vax, flu_rate
from Parameters.census_population import *

from utils import *
from demography import *
from mobility_and_import import *
from clustering import *
from sim_grid import *
from plotting import *
from fit_MCMC import *

from Parameters.times_and_contacts import *
from Parameters.InfluenzaA import *
p_time_to_obs = np.genfromtxt("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',',dtype=np.float64)
incidence = np.array(pd.read_csv("Data/Processed/KPSC_Influenza_A_incidence_age_daily.csv",index_col=0))

## Initial conditions
STATE0 = np.zeros((2*N_S+2)*NAG)
STATE0[NAG:2*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

# Parameters for the ODE
params = {'NAG': NAG, 'N_S': N_S, 'AGING_RATE': AGING_RATE, 'BIRTH_RATE': birth_rate, 'WANE': WANE, 'REC_UP': REC_UP, 'REC_SAME': REC_SAME, 'S_REL': S_REL, 'S_AGE': S_AGE, 'I_REL': I_REL, 'P_OBS': P_OBS, 'birth_vax': birth_vax, 'all_vax': all_vax, 'S_VAX': S_VAX, 'ACOV': flu_rate, 'BCOV': BCOV,
'arrivals': arrivals, 'regional_positivity': regional_positivity, 'IMPORT_RATE': IMPORT_RATE, 'BETA': BETA, 'SEASONALITY': SEASONALITY, 'OFFSET': OFFSET,
'contact': contact}


# flu A DE result
x = [7.204e-03,4.471e-01,1.319e-01,1.329e-02,
7.754e-10,3.510e-01,6.905e-01,4.581e-01,
3.762e-01,1.327e-01,8.848e-01,1.965e-01]
params["WANE"] = np.array([0.0,x[0],0.0])
params["SEASONALITY"] = x[1]
params["OFFSET"] = x[2]
params["BETA"] = x[3]
params["IMPORT_RATE"] = x[4]
srel, pobsrel = constrained_immunity(x[5],x[6],x[7])
params["S_REL"] = srel
params["P_OBS"] = x[8]*pobsrel
OBS_AGE = age_detection(NAG,x[9],x[10],x[11])

np.random.seed(250226)
variables = ["WANE","SEASONALITY","OFFSET","BETA","IMPORT_RATE","EXTRA_IMMUNITY",
             "FIRST_IMMUNITY","FIRST_DIS_INF_FACTOR","P_OBS","OBS_AGE_YOUNG","OBS_AGE_OLD","OBS_AGE_YOUNG_OLD"]
initial_scalars = np.array([7.20377389e-03,4.47077772e-01,1.31897392e-01,1.32852082e-02
,7.75403226e-10,3.51046785e-01,6.90508427e-01,4.58064738e-01
,3.76208662e-01,1.32704840e-01,8.84814311e-01,1.96535397e-01])
log_priors_distribution = sp.stats.multivariate_normal([-1]*12,np.diag([1.5]*12))
log_priors = lambda y : log_priors_distribution.logpdf(y)
proposal_cov = np.diag([2e-10]*12)
scalars = real_to_p(p_to_real(initial_scalars))
scalar_dict = {variables[j]:scalars[j] for j in range(12)}
init_params = scalars_to_params(scalar_dict,params)
mcmc_trajectory, acceptance_rate, likelihoods = mcmc(incidence, init_params, POINTS, STATE0, OBS_AGE, SIS_likelihood, p_time_to_obs, variables, initial_scalars, log_priors, proposal_cov, 10000, True, True, n_messages=100)
print(acceptance_rate)
plt.plot(mcmc_trajectory)
plt.savefig('Figures/mcmc_trajectory_from_FluA_DE.png',dpi=300)
with open('Data/Processed/mcmc_trajectory_from_FluA_DE.pickle','wb') as f:
    pickle.dump(mcmc_trajectory,f)
with open('Data/Processed/mcmc_likelihoods_from_FluA_DE.pickle','wb') as f:
    pickle.dump(likelihoods,f)

# with open('Data/Processed/mcmc_trajectory_from_FluA_DE.pickle','rb') as f:
#     mcmc_trajectory = pickle.load(f)
# with open('Data/Processed/mcmc_likelihoods_from_FluA_DE.pickle','rb') as f:
#     likelihoods = pickle.load(f)

print(mcmc_trajectory[-1])

principal_componets = decomposition.PCA(n_components=2)
X = principal_componets.fit_transform(mcmc_trajectory)
print(principal_componets.explained_variance_ratio_)
print(principal_componets.components_/mcmc_trajectory[-1])
fig, ax = plt.subplots(1,1,figsize=(6.5,6.5))
ax.plot(X[:,0],X[:,1],color='silver')
ax.scatter(X[:,0],X[:,1],c=np.concat((np.ones(1)*np.min(likelihoods),likelihoods)),cmap='viridis')
plt.savefig('Figures/PCA_mcmc_trajectory_from_FluA_DE.png',dpi=300)

print(np.mean(mcmc_trajectory[5000:],axis=0))

corner.corner(mcmc_trajectory[5000:],labels=variables,quantiles=[0.16,0.5,0.84],show_titles=True)
plt.savefig('Figures/corner_mcmc_trajectory_from_FluA_DE.png',dpi=300)