import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
import pandas as pd
from utils import date_to_t, t_to_date, calculate_population_size, susceptibility, load_optimization_results, x_to_params, sum_age_to, pathogen_parameters, load_mcmc_chain, parameters_names_bounds, constrained_immunity, calculate_R0_from_values, consistent_x_from_DE
import matplotlib.pyplot as plt
from scipy.stats import qmc

from likelihood import run_simulation

n_samples = 10000
np.random.seed(260724)

lockdown = "ExponentialODipp25"
option1 = "dedupsac"
pathogens = ["RSV","Metapneumovirus","Parainfluenza3","Adenovirus"]
option2 = "flexagep05"
option2s = ["maxagep028","fixage0maxagep006","maxagep004","maxagep003",]
seeds = [260612, 260622, 260612, 260612,]
# xs = []
# for pathogen, seed, option2 in zip(pathogens, seeds, option2s):
#     x = consistent_x_from_DE(pathogen, lockdown, option1, option2, seed, prefix="emcee_median")
#     xs = xs + [x,]
# xs = np.array(xs)
# true_xs = xs.copy()
# # xs[12:14] = np.mean(xs[12:14])
# minxs = np.min(xs, axis=0)
# maxxs = np.max(xs, axis=0)

# n_free_params = np.sum(minxs != maxxs)

# sampler = qmc.LatinHypercube(d=n_free_params)
# lhs_samples = sampler.random(n=n_samples)

# scaler = np.ones((n_samples,xs.shape[1]))
# scaler[:,minxs!=maxxs] = lhs_samples

# x_samples = minxs[None,:]*scaler + maxxs[None,:]*(1-scaler)


# PERIOD = pd.date_range(start=pd.to_datetime('2015-07-04'), end=pd.to_datetime('2020-01-01'), freq='MS')
# POINTS = np.array(date_to_t(PERIOD))
# from Parameters.census_population import CENSUS_AGE_POP_sac as CENSUS_AGE_POP, AGE_GROUP_NAMES_sac as AGE_GROUP_NAMES, AGE_GROUPS_sac as AGE_GROUPS
# N_S = 3
# NAG = 7
# STATE0 = jnp.zeros((2*N_S+1,NAG))
# STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
# STATE0 = STATE0.at[1,:].set(1)
# # # flatten initial state and add maternal immunity compartment
# STATE0 = STATE0.flatten()
# STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

# def simulator(x):
#     params = x_to_params(x, "sim", "Exponential", option1+"mimmwane", option2+"nr", NAG=NAG)
#     solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
#     values = solution.ys.T
#     trajectory = jnp.diff(values[-NAG:,:], axis=1).T
#     obs_summed_age = trajectory.sum(axis=1)
#     return obs_summed_age

# sim_map = jax.jit(jax.vmap(simulator))

# trajectories = sim_map(x_samples)
# mask_under2k = np.all(trajectories < 2000, axis=1)
# mask_dippers = np.min(trajectories, axis=1) < np.max(trajectories, axis=1)/10
# final_mask = mask_under2k & mask_dippers

# trajectories_dippers = trajectories[final_mask]
# x_dippers = x_samples[final_mask]

# print(len(trajectories_dippers))
# print(len(x_dippers))
# np.savetxt("Data/Processed/sampled_trajectories_nonflulike.csv",trajectories_dippers)
# np.savetxt("Data/Processed/sampled_parameters_nonflulike.csv", x_dippers)

trajectories_dippers = np.genfromtxt("Data/Processed/sampled_trajectories_fluonly.csv")
x_dippers = np.genfromtxt("Data/Processed/sampled_parameters_fluonly.csv")

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

tolerance = 9
clustering = AgglomerativeClustering(metric='chebyshev',
                                     distance_threshold=tolerance,
                                     n_clusters=None,
                                     linkage='complete')
cluster_labels = clustering.fit_predict(trajectories_dippers)


unique_clusters = set(cluster_labels)
print(len(unique_clusters), " clusters")
from scipy import stats
biggest_cluster = stats.mode(cluster_labels)
print("biggest cluster is ", biggest_cluster.mode, " with ", biggest_cluster.count, " trajectories")


# PERIOD = pd.date_range(start=pd.to_datetime('2015-07-04'), end=pd.to_datetime('2025-05-01'), freq='MS')
# POINTS = np.array(date_to_t(PERIOD))
# from Parameters.census_population import CENSUS_AGE_POP_sac as CENSUS_AGE_POP, AGE_GROUP_NAMES_sac as AGE_GROUP_NAMES, AGE_GROUPS_sac as AGE_GROUPS
# N_S = 3
# NAG = 7
# STATE0 = jnp.zeros((2*N_S+1,NAG))
# STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
# STATE0 = STATE0.at[1,:].set(1)
# # # flatten initial state and add maternal immunity compartment
# STATE0 = STATE0.flatten()
# STATE0 = jnp.concatenate((jnp.array([0]), STATE0))
# anchor_idx = 52
# from sim_grid import suppression_duration_single_series
# def suppression_duration_for_pathogen(x):
#     params = x_to_params(x, "sim", "Exponential", option1+"mimmwane", option2+"nr", NAG=NAG)
#     solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
#     values = solution.ys.T
#     trajectory = jnp.diff(values[-NAG:,:], axis=1).T
#     obs_summed_age = trajectory.sum(axis=1)
#     duration = suppression_duration_single_series(obs_summed_age, anchor_idx)
#     return jnp.where(jnp.isnan(duration), 0, duration)
# durfunc = jax.jit(jax.vmap(suppression_duration_for_pathogen))
# durations = durfunc(x_dippers)
# np.savetxt("Data/Processed/xdipperdurations_fluonly.csv", durations)
durations = np.genfromtxt("Data/Processed/xdipperdurations_fluonly.csv")

best_cluster_id = None
max_distance = -1.0
best_pair_indices = None

import time
start_time = time.time()
for i, cluster_id in enumerate(unique_clusters):
    # if i>0:
    #     print((len(unique_clusters)-i) * (time.time()-start_time)/i, end='\r')
    member_indices = np.where(cluster_labels == cluster_id)[0]
    
    # # We need at least 10 trajectories
    if len(member_indices) < 4:
        continue
        
    durations_in_cluster = durations[member_indices]
    
    # Measure parameter distance matrix within this cluster
    p_dist = pairwise_distances(durations_in_cluster.reshape(-1,1))
    current_max_dist = np.mean(p_dist)
    
    if current_max_dist > max_distance:
        max_distance = current_max_dist
        best_cluster_id = cluster_id
        
        # Get the index pair of the two most extreme parameter sets
        i, j = np.unravel_index(np.argmax(p_dist), p_dist.shape)
        best_pair_indices = (member_indices[i], member_indices[j])

print(best_cluster_id)

# 4. Extract results
idx1, idx2 = best_pair_indices
print(f"Max Parameter Distance Found: {max_distance:.4f}")
print(f"Parameter Set A: {x_dippers[idx1]}")
print(f"Parameter Set B: {x_dippers[idx2]}")

PERIOD = pd.date_range(start=pd.to_datetime('2015-07-04'), end=pd.to_datetime('2025-05-01'), freq='W')
POINTS = np.array(date_to_t(PERIOD))
from Parameters.census_population import CENSUS_AGE_POP_sac as CENSUS_AGE_POP, AGE_GROUP_NAMES_sac as AGE_GROUP_NAMES, AGE_GROUPS_sac as AGE_GROUPS
N_S = 3
NAG = 7
STATE0 = jnp.zeros((2*N_S+1,NAG))
STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
STATE0 = STATE0.at[1,:].set(1)
# # flatten initial state and add maternal immunity compartment
STATE0 = STATE0.flatten()
STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

def simulator(x, m):
    params = x_to_params(x, "sim", "Exponential", option1+"mimmwane", option2+"nr", NAG=NAG)
    x[-7:] = x[-7:]*m
    solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
    values = solution.ys.T
    trajectory = jnp.diff(values[-NAG:,:], axis=1).T
    obs_summed_age = trajectory.sum(axis=1)
    return obs_summed_age

print(np.sum(cluster_labels == best_cluster_id))
# mult = [1,2,1.25,1]
cluster_idxs = np.where(cluster_labels == best_cluster_id)[0]
best_cluster_trajectories = [simulator(x_dippers[cluster_idxs[i]],1) for i in range(len(cluster_idxs))]

np.savetxt("Data/Processed/clustered_trajectories_durations_fluonly_scaled.csv",best_cluster_trajectories)


best_cluster_trajectories = np.genfromtxt("Data/Processed/clustered_trajectories_durations_fluonly_scaled.csv")
print(len(best_cluster_trajectories))
colors = ["#648FFF","#DC267F", "#FFB000" , "#785EF0","#FF832B", "#004D40", ]*3
print(np.max(best_cluster_trajectories,axis=1))
best_cluster_trajectories = [best_cluster_trajectories[i] for i in [1,2,3]]

np.savetxt("Data/Processed/favourite_fluonly_clustered_trajectories_scaled2.csv", best_cluster_trajectories)

from sim_grid import suppression_duration_single_series
for color, traj in zip(colors,best_cluster_trajectories):
    duration = str(int(suppression_duration_single_series(traj,anchor_idx=260)/4.35))
    plt.plot(traj,alpha=1,color=color, label=duration+" months")
plt.legend(title="Suppression duration", frameon=False)
plt.savefig("Figures/cluster_trajectories_duration_fluonly.png",dpi=500)
