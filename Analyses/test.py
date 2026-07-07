## Brainstorming for susceptible recruitment project
## Code to explore how susceptibles recruitment affects outbreak dynamics
## BJS August 2024

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy as sp
import pandas as pd
import emcee
# import itertools as it
from plotting import lockdown_incidence_plot, lockdown_incidence_format, kpsc_positive_test_plot
# from math import comb
from utils import *
from Parameters.census_population import AGE_GROUP_NAMES, CENSUS_AGE_POP
from Parameters.times_and_contacts import PERIOD
# import pickle
# from scipy.optimize import curve_fit
import time
import corner

import seaborn as sns
from matplotlib import cm as colormaps
hsv_colors = colormaps.hsv(-0.02+np.arange(7)/7)
hsv_colors[3] = colormaps.hsv((3/7)+0.04)

from fit_MCMC import SIS_likelihood, run_simulation
from fit_opt import get_likelihood
import time
import pickle
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({'font.size':8})
# text type is palatino
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino']

# # Adenovirus
# S_REL = [1, 0.41513026, 0.40689045]
# P_OBS = [1, 0.29316723, 0.16823587]
# OBS_AGE = [0.00081195, 0.00244854, 0.00187553, 0.00020819, 0.00031958, 0.00069796, 0.003]
# # find proportional difference between each age group and prior one for OBS_AGE
# AGE_REL = [1] + [OBS_AGE[i]/OBS_AGE[i-1] for i in range(1, len(OBS_AGE))]
# print(AGE_REL)
# # Influenza A
# S_REL = [1.0, 0.81661355, 0.5758953]
# P_OBS = [1.0, 0.98442453, 0.7888398]
# OBS_AGE = [0.00729031, 0.00509014, 0.00213986, 0.0005347, 0.00109613, 0.00390308, 0.035]
# AGE_REL = [1] + [OBS_AGE[i]/OBS_AGE[i-1] for i in range(1, len(OBS_AGE))]
# print(AGE_REL)
# # RSV
# S_REL = [1.0, 0.43337923, 0.04520457]
# P_OBS = [1.0, 0.46, 0.31]
# OBS_AGE = [0.028, 0.00756089, 0.00427777, 0.00039543, 0.0002941, 0.00129138, 0.01125933]
# AGE_REL = [1] + [OBS_AGE[i]/OBS_AGE[i-1] for i in range(1, len(OBS_AGE))]
# print(AGE_REL)
# # Metapneumovirus
# S_REL = [1.0, 0.8700617, 0.7241823]
# P_OBS = [1.0, 0.17723344, 0.0211956]
# OBS_AGE = [0.00115598, 0.0030468, 0.00358958, 0.00050364, 0.00060943, 0.00208401, 0.015]
# AGE_REL = [1] + [OBS_AGE[i]/OBS_AGE[i-1] for i in range(1, len(OBS_AGE))]
# print(AGE_REL)
# # Parainfluenza 3
# S_REL = [1.0, 0.55912185, 0.29086292]
# P_OBS = [1.0, 0.9597223, 0.72615325]
# OBS_AGE = [0.0030517078, 0.0021154704, 0.00080672855, 0.000054024615, 0.000097420758, 0.00041733991, 0.0040000002]
# AGE_REL = [1] + [OBS_AGE[i]/OBS_AGE[i-1] for i in range(1, len(OBS_AGE))]
# print(AGE_REL)

# ### plot percent positivity
# fig, ax = plt.subplots(2, 3, figsize = (6.5,4), sharex=True, sharey=True)
# for pi, pathogen in enumerate(["RSV", "Metapneumovirus", "Parainfluenza3", "InfluenzaB", "Adenovirus", "InfluenzaA"]):
#     pp = np.genfromtxt(f"Data/Processed/{pathogen}_positivity_daily.csv", delimiter=',')
#     # this is a 1d array by day from 1970-01-01. Plot against dates from 2015-10-01 to 2025-05-01
#     start_date = pd.to_datetime('2015-10-01')
#     end_date = pd.to_datetime('2025-05-01')
#     start_idx = (start_date - pd.to_datetime('1970-01-01')).days
#     dates = pd.date_range(start=start_date, end=end_date, freq='D')
#     ax[pi//3, pi%3].plot(dates, pp[start_idx:start_idx+len(dates)], label="Data", color='k')
#     ax[pi//3, pi%3].set_title(pathogen)
# plt.show()

### plot mcmc output
# lockdown = "ExponentialODipp25"
# pathogens = ["RSV","Metapneumovirus","Parainfluenza3","Adenovirus","InfluenzaA","InfluenzaB",]
# flunet_pathogens = ["RSV","Metapneumovirus","Parainfluenza","Adenovirus","InfluenzaA","InfluenzaB",]
# colors = ["#DC267F", "#FFB000", "#FF832B", "#648FFF", "#785EF0", "#004D40",]
option1 = "dedupsac"
# option2s = ["maxagep028","fixage0maxagep006","maxagep004","maxagep003","maxagep035","maxagep035",]
# seeds = [260612, 260622, 260612, 260612, 260612, 260612,]
lockdown = "ExponentialODipp25"
pathogens = [ "Adenovirus",]
option2s = [ "maxagep003", ]
seeds = [260612,]
colors = ["#FF832B",]
def get_srel1_from_constrained_immunity(extra_immunity, first_immunity, first_dis_inf_factor):
    srel, _ = constrained_immunity(extra_immunity, first_immunity, first_dis_inf_factor)
    return srel[1]
# fig, ax = plxt.subplots()
r0_samples_by_pathogen = {}
immunity_samples_by_pathogen = {}
for pathogen, color, option2, seed in zip(pathogens, colors, option2s, seeds):
    start_time = time.time()
    prefix = ""
    NAG = 7
    N_S = 3
    from Parameters.census_population import AGE_GROUPS_sac as AGE_GROUPS, AGE_GROUP_NAMES_sac, CENSUS_AGE_POP_sac as CENSUS_AGE_POP
    param_names, bounds = parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)

    mcmc_filepath = f"Outputs/mcmc_samples_big_{prefix}{pathogen}_{lockdown}_{option1}_{option2}_{seed}_refined.csv"
    # mcmc_filepath = "Outputs/mcmc_samples_big_RSV_ExponentialODipLinear_dedupsac_maxagep028_260531_refined_studio.csv"
    mcmc_samples = np.genfromtxt(mcmc_filepath, delimiter=',', skip_header=0)
    loading_time = time.time()
    print(f"Loading samples time for {pathogen}: {time.time() - start_time:.2f} seconds")
    n_params = bounds.shape[0]
    n_walkers = 500
    n_iterations = len(mcmc_samples) // n_walkers
    n_params = len(param_names)

    # # # cut off burn-in
    # if pathogen in ["RSV", "Metapneumovirus", "Parainfluenza3", "Adenovirus"]:
    n_iterations -= 2000
    mcmc_samples = mcmc_samples[2000*n_walkers:, :]

    # # find index of S_REL1 in pram_names
    # thinning = 100 if "Influenza" not in pathogen else 1000
    # thinned_samples = mcmc_samples[::thinning, :]
    # gamma = 4.9 if ("RSV" in pathogen) or "Metapneumovirus" in pathogen else 3.0
    # r0_samples = thinned_samples[:,0]*15.24*gamma
    # if "S_REL1" in param_names:
    #     j = param_names.index("S_REL1")
    #     srel1_samples = thinned_samples[:,j]
    # else:
    #     eidx = param_names.index("EXTRA_IMMUNITY")
    #     fidx = param_names.index("FIRST_IMMUNITY")
    #     fdifdx = param_names.index("FIRST_DIS_INF_FACTOR")
    #     srel1_samples = jax.jit(jax.vmap(get_srel1_from_constrained_immunity))(thinned_samples[:,eidx], thinned_samples[:,fidx], thinned_samples[:,fdifdx])
    # thinning_time = time.time()
    # print(f"Thinning samples time for {pathogen}: {time.time() - loading_time:.2f} seconds")
    # median_r0 = np.median(r0_samples)
    # median_immunity = 1-np.median(srel1_samples)
    # print(f"Median R0 for {pathogen}: {median_r0:.2f}, Median first immunity for {pathogen}: {median_immunity:.2f}")
    # r0_samples_by_pathogen[pathogen] = r0_samples
    # immunity_samples_by_pathogen[pathogen] = 1-srel1_samples
    # # if pathogen == "RSV":
    # #     mcmc_samples = mcmc_samples[-17000*64:, :]
    # mcmc_samples = mcmc_samples.reshape((-1, n_walkers, n_params))


    # # load logprob
    # logprob_filepath = f"Outputs/mcmc_log_prob_big_{prefix}{pathogen}_{lockdown}_{option1}_{option2}_{seed}_refined.csv"
    # log_prob_samples = np.genfromtxt(logprob_filepath, delimiter=',', skip_header=0)

    # if pathogen in ["RSV", "Metapneumovirus", "Parainfluenza3", "Adenovirus"]:
    #     log_prob_samples = log_prob_samples[10000:]

    # # if pathogen == "RSV":
    # #     log_prob_samples = log_prob_samples[-17000:]
    # print(log_prob_samples.shape)
    # # print parameters at minimum nll
    # log_prob_2d = np.atleast_2d(log_prob_samples)
    # best_idx = np.unravel_index(np.argmax(log_prob_2d), log_prob_2d.shape)
    # mcmc_samples_2d = mcmc_samples.reshape(-1, n_params)
    # flat_best_idx = best_idx[0] * log_prob_2d.shape[1] + best_idx[1]
    # best_params = mcmc_samples_2d[flat_best_idx]
    # best_log_prob = float(log_prob_2d[best_idx])
    # best_neg_log_likelihood = -best_log_prob
    # if "Influenza" in pathogen:
    #     best_srel1 = get_srel1_from_constrained_immunity(best_params[eidx], best_params[fidx], best_params[fdifdx])
    # else:
    #     best_srel1 = best_params[j]
    # print(f"Best R0 found by MCMC: {15.24*gamma*best_params[0]:.4f}, best immunity found by MCMC: {1-best_srel1:.4f}, with negative log-probability: {best_neg_log_likelihood:.4f}")
    

    # mcmc_samples_2d = mcmc_samples.reshape(-1, n_params)
    # median_params = np.median(mcmc_samples_2d, axis=0)
    # log_prob_of_median = get_likelihood(pathogen, lockdown, option1, option2, 1e-9, NAG=NAG, CENSUS_AGE_POP=CENSUS_AGE_POP, AGE_GROUPS=AGE_GROUPS)[0](median_params)
    # # # --- AD-HOC EMCEE EXPORT BLOCK (safe to delete when no longer needed) ---
    # WRITE_EMCEE_ADHOC_EXPORT = True
    # if WRITE_EMCEE_ADHOC_EXPORT:
    #     results_dir = f"Data/Processed/results{str(seed)[:6]}"
    #     os.makedirs(results_dir, exist_ok=True)
    #     emcee_results_file = f"{results_dir}/emcee_median_{prefix}{pathogen}{lockdown}{option1}{option2}{seed}.pickle"
    #     with open(emcee_results_file, "wb") as f:
    #         pickle.dump(
    #             {
    #                 "final_population": np.asarray([median_params]),
    #                 "final_fitness": np.asarray([log_prob_of_median]),
    #             },
    #             f,
    #         )
    #     print(f"Wrote ad-hoc emcee export: {emcee_results_file}")
    # # --- END AD-HOC EMCEE EXPORT BLOCK ---

    
#     # fig, trax = plt.subplots(4,4, figsize=(10,6))
#     # for param_idx in range(n_params):
#     #     for walker_idx in range(n_walkers):
#     #         # if best_idx[1] == walker_idx:
#     #         #     trax[param_idx//4, param_idx%4].plot(mcmc_samples[:,walker_idx,param_idx], alpha=1, zorder=10, color='k', label="Best fit walker")
#     #         # else:
#     #         trax[param_idx//4, param_idx%4].plot(mcmc_samples[:,walker_idx,param_idx], alpha=0.4)
#     #     trax[param_idx//4, param_idx%4].set_title(param_names[param_idx])
#     #     # trax[param_idx//4, param_idx%4].axvline(x=best_idx[0], color='k', linestyle='-', label="Best fit")
#     # plt.tight_layout()
#     # plt.savefig(f"Figures/mcmc_traces_big_{prefix}{pathogen}_{lockdown}_{option1}_{option2}_{seed}_all_walkers_test.png", dpi=300, bbox_inches='tight')
#     # plt.close(fig)


    # # Reshape it back to 3D to separate the walkers properly
    chain_3d = mcmc_samples.reshape(n_iterations, n_walkers, n_params)

    # calculate ensemble-aware autocorrelation time and ESS
    try:
        autocorr_times = emcee.autocorr.integrated_time(chain_3d, quiet=True)
    except emcee.autocorr.AutocorrError as error:
        autocorr_times = np.asarray(error.tau)
        print("Warning: chain may be too short for reliable autocorrelation estimates.")

# #     autocorr_times = np.asarray(autocorr_times, dtype=float)
# #     effective_sample_sizes = (n_iterations * n_walkers) / autocorr_times
# #     print("Autocorrelation times:\n", autocorr_times)
# #     print("Effective sample sizes:\n", effective_sample_sizes)

# #     maximum_autocorr_time = int(np.ceil(np.max(autocorr_times)))

#     # # chain_3d_pruned = chain_3d[500:][::maximum_autocorr_time]
#     median_value_by_parameter = np.median(chain_3d, axis=(0,1))
#     lower_value = np.percentile(chain_3d, 2.5, axis=(0,1))
#     upper_value = np.percentile(chain_3d, 97.5, axis=(0,1))
#     # # print(median_value_by_parameter)
#     # # print(lower_value)
#     # # print(upper_value)

#     # # print best parameter set with credible intervals
#     # print("Best parameter set found by MCMC, with negative log-likelihood:", best_neg_log_likelihood)
#     for i in range(n_params):
#         print(f"median {param_names[i]}: {median_value_by_parameter[i]:.4f} ({lower_value[i]:.4f}–{upper_value[i]:.4f})")
#         # print(f"optimal {param_names[i]}: {best_params[i]:.4f}")

    import corner
    fig = corner.corner(chain_3d.reshape(-1, n_params), labels=param_names, show_titles=True, title_fmt=".4f", title_kwargs={"fontsize": 8})
    plt.savefig(f"Figures/mcmc_corner_big_{prefix}{pathogen}_{lockdown}_{option1}_{option2}_{seed}_cutoff2000.pdf", bbox_inches='tight')
    plt.close(fig)
    
# #     # sns.kdeplot(ax=ax, x=r0_samples, y=1-srel1_samples, color=color, label=pathogen, fill=True)
# #     # # ax.scatter(15.24*gamma*best_params[0], 1-best_srel1, marker='x', color='k', label="MAP estimate")
# #     # ax.scatter(median_r0, median_immunity, marker='o', color='k', label=f"Median parameter values")
# #     # print(f"Plotting time for {pathogen}: {time.time() - thinning_time:.2f} seconds")

# # # build custom legend with colored squares
# # handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=8) for color in colors]
# # labels = pathogens.copy()
# # # add to legend a black x for MAP estimate
# # handles.append(plt.Line2D([0], [0], marker='x', color='k', markersize=8, linestyle='None'))
# # labels.append("MAP estimate")
# # ax.legend(handles, labels, frameon=False, fontsize=8, loc='lower right')
# # ax.set_xscale('log')
# # ax.set_xticks([1,2,3,4,5,6,7,8,9,10,20])
# # ax.set_xticklabels([1,2,3,4,5,6,7,8,9,10,20])
# # ax.set_xlabel("Basic reproduction number (log scale)")
# # ax.set_ylabel("Immunity from first infection")
# # plt.savefig(f"Figures/mcmc_r0_vs_srel1_kde_thinned.png", dpi=300, bbox_inches='tight')

# # line_sample_start = time.time()
# # # # bootstrap line of best fit by pulling one sample from each pathogen to get confidence intervals on the slope and intercept
# # # slope_samples = []
# # # intercept_samples = []
# # # y_samples = []
# # # x = np.linspace(1, 20, 1000)
# # # for i in range(1000):
# # #     r0_sample = []
# # #     srel1_sample = []
# # #     for pathogen in pathogens:
# # #         sample_idx = np.random.choice(len(r0_samples_by_pathogen[pathogen]))
# # #         r0_sample.append(r0_samples_by_pathogen[pathogen][sample_idx])
# # #         srel1_sample.append(immunity_samples_by_pathogen[pathogen][sample_idx])
# # #     r0_sample = np.log(np.log(np.array(r0_sample)))
# # #     srel1_sample = np.array(srel1_sample)
# # #     slope, intercept, r_value, p_value, std_err = sp.stats.linregress(r0_sample, srel1_sample)
# # #     slope_samples = np.append(slope_samples, slope)
# # #     intercept_samples = np.append(intercept_samples, intercept) 
# # #     y_samples = np.append(y_samples, slope * np.log(np.log(x)) + intercept)

# # from scipy import odr
# # def linear_func(B, x):
# #     return B[0] * x + B[1]

# # linear_model = odr.Model(linear_func)

# # n_iterations = 2000  # Bumped up slightly for smoother confidence bands
# # slope_samples = np.zeros(n_iterations)
# # intercept_samples = np.zeros(n_iterations)

# # # Create a dense x-grid for plotting
# # x_grid = np.linspace(1.05, 20, 10000)
# # x_grid_transformed = np.log(np.log(x_grid))

# # # We will store the evaluated y-values for each sampled line here
# # y_lines = np.zeros((n_iterations, len(x_grid)))

# # for i in range(n_iterations):
# #     r0_draws = []
# #     srel1_draws = []
    
# #     # Draw exactly one (R0, Immunity) pair from each pathogen's joint posterior
# #     for pathogen in pathogens:
# #         sample_idx = np.random.choice(len(r0_samples_by_pathogen[pathogen]))
# #         r0_draws.append(r0_samples_by_pathogen[pathogen][sample_idx])
# #         srel1_draws.append(immunity_samples_by_pathogen[pathogen][sample_idx])
        
# #     # Transform x-data exactly as before
# #     x_data = np.log(np.log(np.array(r0_draws)))
# #     y_data = np.array(srel1_draws)
    
# #     # 2. Run Orthogonal Distance Regression
# #     # We don't need to pass weights (wd, we) here because the Monte Carlo 
# #     # sampling process itself naturally weights the parameter space!
# #     data = odr.Data(x_data, y_data)
    
# #     # Provide a rough initial guess for the solver (slope=1.0, intercept=0.0)
# #     myodr = odr.ODR(data, linear_model, beta0=[1.0, 0.0])
# #     output = myodr.run()
    
# #     # Extract optimized slope and intercept
# #     slope_samples[i] = output.beta[0]
# #     intercept_samples[i] = output.beta[1]
    
# #     # Evaluate the line on our plotting grid
# #     y_lines[i, :] = slope_samples[i] * x_grid_transformed + intercept_samples[i]

# # # plot line of best fit with confidence interval from bootstrap
# # y_median = np.median(y_lines, axis=0)
# # y_lower = np.percentile(y_lines, 2.5, axis=0)
# # y_upper = np.percentile(y_lines, 97.5, axis=0)
# # # Filter out x values where the predictions go completely out of bounds (below 0 or above 1)
# # valid_mask = (y_median >= 0) & (y_median <= 1)
# # x_plot = x_grid[valid_mask]
# # y_median_plot = y_median[valid_mask]
# # # Also ensure bounds are valid for the fill_between
# # valid_mask_fill = (y_upper >= 0) & (y_upper <= 1)
# # x_fill = x_grid[valid_mask_fill]
# # y_lower_fill = y_lower[valid_mask_fill]
# # y_upper_fill = y_upper[valid_mask_fill]
# # # clip y_lower above 0
# # y_lower_fill = np.clip(y_lower_fill, 0, None)
# # ax.plot(x_plot, y_median_plot, color='k', linestyle='-', label="ODR Best fit")
# # ax.plot(x_fill, y_lower_fill, color='k', linestyle='--', label="95% CI")
# # ax.plot(x_fill, y_upper_fill, color='k', linestyle='--')

# # plt.savefig(f"Figures/mcmc_r0_vs_srel1_kde_thinned_w_odr_bootstrap_new.png", dpi=300, bbox_inches='tight')
# # print(f"Line of best fit plotting time: {time.time() - line_sample_start:.2f} seconds")



# pathogen = "RSV"
# option2 = "maxagep028"
# filepath = f"Data/Processed/results260505/evosax_DE_{pathogen}ExponentialODipEqualdedupsplit{option2}260505.pickle"
# with open(filepath, "rb") as f:
#     results = pickle.load(f)
# print(f"Loaded results from disk: {results.keys()}")
# metrics_log = results["metrics_log"]
# print(metrics_log.keys())


# # # ##### exploratory figures to understand age distribution of infections in the model
# # pathogen = "InfluenzaA"
# # seed = 260527
# # lockdown = "ExponentialODipLinear"
# # option1 = "dedupsplit"
# # option2 = "maxagep03"
# # NAG = 8
# # N_S = 3
# # from Parameters.census_population import AGE_GROUP_NAMES_split, CENSUS_AGE_POP_split as CENSUS_AGE_POP

# p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/RSV_incubation_admittance_distribution.csv", delimiter=',', header=None).values)
# PERIOD = pd.date_range(start=pd.to_datetime('2015-10-01'), end=pd.to_datetime('2025-10-01'), freq='D')
# POINTS = np.array(date_to_t(PERIOD))
# ## Initial conditions
# STATE0 = jnp.zeros((2*N_S+1,NAG))
# STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
# STATE0 = STATE0.at[1,:].set(1)
# # # flatten initial state and add maternal immunity compartment
# STATE0 = STATE0.flatten()
# STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

# fig, ax = plt.subplots(2, 4, figsize=(3,3))
# for _ in range(100):
#     # random x from chain
#     x = chain_3d[np.random.randint(500, n_iterations), np.random.randint(0, n_walkers), :]
#     params = x_to_params(x, pathogen, lockdown, option1, option2, NAG=NAG)
#     age_obs_2 = x[8]
#     # # red to blue gradient for age_obs2 from 0.25 to 0.325
#     # color = plt.cm.RdBu((age_obs_2 - 0.25) / (0.325 - 0.25))
#     color = ['r', 'b'][(age_obs_2 > 0.285).astype(int)]
#     for age_group in range(NAG):
#         mx = lockdown_incidence_plot(ax[age_group//4, age_group%4],STATE0,params,POINTS,date_to_t('2020-03-19'), alpha=0.3, color=color, label="Simulation",by_age=True,select_age_group=age_group,AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=[1,7,30.44][[None,"W","MS"].index(None)]*10000,p_time_to_obs=p_time_to_obs,NAG=NAG,)
# plt.tight_layout()
# plt.savefig(f"Figures/mcmc_traces_big_{pathogen}_{lockdown}_{option1}_{option2}_{seed}_incidence_colored_by_age_obs_2.png", dpi=300, bbox_inches='tight')
# plt.close(fig)
# # fig, ax = plt.subplots(7,7, sharex="col", sharey="row")
# # fig,ax = plt.subplots(3,3, sharex=True,sharey=True)
# # matrix = np.zeros((3,6,8))
# same_inf = np.zeros((6,NAG))
# diff_inf = np.zeros((6,NAG))
# # pre_pandemic_cog = np.zeros(6)
# # data_cog = np.zeros(6)
# # beta = np.zeros(6)
# # seasonality = np.zeros(6)
# # phase = np.zeros(6)
# # wane = np.zeros(6)
# # srel1 = np.zeros(6)
# # srel2 = np.zeros(6)
# pathogens = ["RSV",  "Metapneumovirus", "Parainfluenza3", "InfluenzaB", "Adenovirus", "InfluenzaA"]
# opt2s = ["maxagep028", "maxagep0085", "maxagep007", "maxagep04", "maxagep004", "maxagep04"]
# colors = ["#DC267F", "#FFB000",  "#FF832B", "#648FFF",  "#785EF0","k"]
# for i, (pathogen, option2, color) in enumerate(zip(pathogens,opt2s, colors)):
#     # x = consistent_x_from_DE(pathogen, lockdown, option1, option2, seed, NAG=NAG)
#     # params = x_to_params(x, "sim", lockdown, option1+"mimmwane", option2+"nr", NAG=NAG)
#     _, x, _ = load_optimization_results("", pathogen, seed, lockdown, option1, option2)
#     params = x_to_params(x, pathogen, lockdown, option1, option2, NAG=NAG)
#     solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
#     # find total observed infections each season in each age group
#     values = solution.ys.T
#     shaped_values = values[1:, :].reshape((1+2*N_S, NAG, -1))
#     infectious = shaped_values[1:2*N_S:2, :, :]
#     susceptible = shaped_values[0:2*N_S:2, :, :]
#     all_infectious = infectious.sum(axis=0)
#     population_size = calculate_population_size(values, NAG=NAG)
#     relative_contact = params[10]
#     foi_matrix = params[4] * params[3][:, :, None] * all_infectious[None, :, :] / jnp.sum(population_size, axis=1)[None, None, :]
#     infections_matrix = params[6][:, None, None, None] * foi_matrix[None, :, :, :] * susceptible[:, :, None, :]
    
#     age_infections_matrix = infections_matrix.sum(axis=0)[:1553].mean(axis=-1)
#     print(age_infections_matrix)
#     same_inf[i,:] = np.diag(age_infections_matrix)/np.sum(age_infections_matrix, axis=1)
#     for age_infected in range(NAG):
#         diff_inf[i,age_infected] = np.sum([age_infections_matrix[age_infected, age_infector] for age_infector in range(NAG) if age_infector != age_infected])
#     diff_inf[i, :] /= np.sum(age_infections_matrix, axis=1)

#     # outcome = worker((x, lockdown, POINTS, STATE0, p_time_to_obs, option1, option2, NAG))

#     # matrix[0,i] = outcome[5, :5, :-1].mean(axis=0)/(365)
#     # matrix[1,i] = outcome[4, :5, :-1].mean(axis=0)/(outcome[7, 0, :-1]*365)
#     # matrix[2,i] = outcome[6, :5, :-1].mean(axis=0)/(outcome[7, 0, :-1]*365)
#     # pre_pandemic_cog[i] = (jnp.sum(outcome[1, :5, :-1]*outcome[7, 0, :-1], axis=1)/jnp.sum(outcome[7, 0, :-1])).mean()
#     # peak_times = extract_target_value_from_data(pathogen, "peak_times", NAG=8)
#     # print(peak_times)
#     # data_cog[i] = (jnp.sum(peak_times[:5, :-1]*outcome[7, 0, :-1], axis=1)/jnp.sum(outcome[7, 0, :-1])).mean()
#     # beta[i] = x[2]
#     # seasonality[i] = x[3]
#     # phase[i] = x[4]*365
#     # wane[i] = x[6]*365
#     # srel1[i] = x[7]
#     # srel2[i] = x[8]

#     # for i in range(7):
#     #     for j in range(1,8):
#     #         if j>i:
#     #             ax[i,j-1].scatter(hospitalizors[j], hospitalizors[i], color=color, label=pathogen)

# # fig, ax = plt.subplots(2,2, sharex="col")
# # ax[0,0].scatter(phase, pre_pandemic_cog, color=colors)
# # ax[0,0].plot(phase, phase, 'k-', alpha=0.5, label="y=x")
# # # plot dotted lines from pathogens down to y=x
# # for i in range(len(phase)):
# #     ax[0,0].plot([phase[i], phase[i]], [pre_pandemic_cog[i], phase[i]], 'k--', alpha=0.5)
# # ax[0,0].set_ylabel("Simulated peak day of season")
# # ax[0,1].scatter((beta*(1-srel2)*wane), pre_pandemic_cog - phase, color=colors)
# # z1 = np.polyfit((beta*(1-srel2)*wane), pre_pandemic_cog - phase, 1)
# # p1 = np.poly1d(z1)
# # x1_line = np.linspace((beta*(1-srel2)*wane).min(), (beta*(1-srel2)*wane).max(), 100)
# # ax[0,1].plot(x1_line, p1(x1_line), 'k-', alpha=0.5, label="line of best fit")
# # # ax[0,1].set_xlabel("Beta")
# # ax[0,1].set_ylabel("Simulated peak - Phase (days)")
# # y_err = 1.96 * np.std(pre_pandemic_cog - phase - p1((beta*(1-srel2)*wane))) / np.sqrt(len(pre_pandemic_cog))
# # ax[0,1].fill_between(x1_line, p1(x1_line) - y_err, p1(x1_line) + y_err, color='gray', alpha=0.2)
# # ax[1,0].scatter(phase, data_cog, color=colors)
# # ax[1,0].plot(phase, phase, 'k-', alpha=0.5, label="y=x")
# # ax[1,0].set_xlabel("Phase (days)")
# # ax[1,0].set_ylabel("Data peak day of season")
# # for i in range(len(phase)):
# #     ax[1,0].plot([phase[i], phase[i]], [data_cog[i], phase[i]], 'k--', alpha=0.5)
# # ax[1,1].scatter((beta*(1-srel2)*wane), data_cog - phase, color=colors)
# # z4 = np.polyfit((beta*(1-srel2)*wane), data_cog - phase, 1)
# # p4 = np.poly1d(z4)
# # x4_line = np.linspace((beta*(1-srel2)*wane).min(), (beta*(1-srel2)*wane).max(), 100)
# # ax[1,1].plot(x4_line, p4(x4_line), 'k-', alpha=0.5, label="line of best fit")
# # ax[1,1].set_xlabel("Beta + Acquired immunity")
# # ax[1,1].set_ylabel("Data peak - Phase (days)")
# # # add uncertainty shaded region to ax[1,1] based on 95% confidence intervals of polyfit
# # y_err = 1.96 * np.std(data_cog - phase - p4((beta*(1-srel2)*wane))) / np.sqrt(len(data_cog))
# # ax[1,1].fill_between(x4_line, p4(x4_line) - y_err, p4(x4_line) + y_err, color='gray', alpha=0.2)
# # # legend of pathogen colors along bottom
# # for axis in ax.flatten():
# #     axis.legend(loc='upper left', frameon=False, fontsize=8)
# # handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) for color in colors]
# # labels = pathogens
# # fig.legend(handles, labels, frameon=False, fontsize=8, loc='lower center', ncol=len(pathogens), bbox_to_anchor=(0.5, -0.05))
# # plt.tight_layout()
# # plt.savefig(f"Figures/COG_vs_phase_beta_seasonality_srel2_wane_times.png", dpi=300, bbox_inches='tight')

# fig, ax = plt.subplots(2,4,figsize=(6.5,4),sharex=True,sharey=True)
# for ai in range(NAG):
#     ax[ai//4,ai%4].plot([0,0.7],[1,0.3],color='silver',zorder=0)
#     for pi in range(len(pathogens)):
#         ax[ai//4,ai%4].scatter(same_inf[pi,ai], diff_inf[pi,ai], color=colors[pi], label=pathogens[pi])
#     ax[ai//4,ai%4].set_title(AGE_GROUP_NAMES_split[ai])
# handles, labels = ax[0,0].get_legend_handles_labels()
# # overall xlabel and y label
# fig.text(0.5, 0.04, "Proportion of infections from same age group", ha='center', va='center')
# fig.text(0.06, 0.5, "Proportion of infections from different age groups", ha='center', va='center', rotation='vertical')
# fig.legend(handles, labels, frameon=False, fontsize=8, loc='lower center', ncol=len(pathogens), bbox_to_anchor=(0.5, -0.15))
# plt.savefig(f"Figures/same_inf_vs_diff_inf_by_age_group_irel05028.png", dpi=300, bbox_inches='tight')

# # # Age group control variables for plotting
# # age_group_x_idx = 0
# # age_group_x_name = AGE_GROUP_NAMES_split[age_group_x_idx]
# # age_group_x_name = "<1y"
# # age_group_y_idx = -1
# # age_group_y_name = AGE_GROUP_NAMES_split[age_group_y_idx]

# # fig, ax = plt.subplots(1,3, figsize=(6.5,2.5))
# # for i in range(len(pathogens)):
# #     ax[0].scatter(matrix[0,i,0:2].sum(), matrix[0,i,age_group_y_idx], color=colors[i], label=pathogens[i])
# #     ax[1].scatter(matrix[1,i,0:2].sum(), matrix[1,i,age_group_y_idx], color=colors[i], label=pathogens[i])
# #     ax[2].scatter(matrix[2,i,0:2].sum(), matrix[2,i,age_group_y_idx], color=colors[i], label=pathogens[i])
# # ax[0].set_xlabel(age_group_x_name)
# # ax[0].set_ylabel(age_group_y_name)
# # ax[0].set_title("Force of infection")
# # ax[1].set_xlabel(age_group_x_name)
# # ax[1].set_title("Infectors")
# # ax[2].set_xlabel(age_group_x_name)
# # ax[2].set_title("Hospitalizors")
# # handles, labels = ax[0].get_legend_handles_labels()
# # fig.legend(handles, labels, frameon=False, fontsize=8, loc='lower center', ncol=len(pathogens), bbox_to_anchor=(0.5, -0.15))
# # plt.tight_layout()
# # plt.savefig(f"Figures/force_of_infection_infectors_hospitalizors_scatter_{age_group_x_name}_vs_{age_group_y_name}.png", dpi=300, bbox_inches='tight')

# # # print matrix[2] in tab-separated format to two significant figures, not in scientific notation
# # header = "," + ",".join(pathogens)
# # print(header)
# # for i, age_group in enumerate(AGE_GROUP_NAMES_split):
# #     row = [age_group]
# #     for j in range(len(pathogens)):
# #         val = matrix[2, j, i]
# #         # Format to 2 significant figures
# #         if val == 0:
# #             row.append("0")
# #         else:
# #             row.append(f"{val:.2g}")
# #     print(",".join(row))

# # # ax.legend()
# # matrix1 = matrix/np.sum(matrix, axis=1)[:,None,:]
# # matrix2 = matrix/np.sum(matrix, axis=2)[:,:,None]
# # for j,m in enumerate([matrix,matrix1,matrix2]):
# #     ax[j,0].imshow(np.log(m[0]))
# #     ax[j,1].imshow(np.log(m[1]))
# #     ax[j,2].imshow(np.log(m[2]))
# # ax[0,0].set_title("Force of infection")
# # ax[0,1].set_title("Infectors")
# # ax[0,2].set_title("Hospitalizors")
# # ax[0,0].set_yticks(range(6))
# # ax[0,0].set_yticklabels(pathogens)
# # ax[0,0].set_xticks(range(NAG))
# # ax[0,0].set_xticklabels(AGE_GROUP_NAMES_split)
# # plt.show()


# ######
# from Parameters.census_population import AGE_GROUPS_split as AGE_GROUPS
# expected_obs = jnp.ones((10,65))
# expected_obs = expected_obs.at[:,0:3].set(0.5)
# expected_obs = expected_obs.at[:,3:12].set(0.3)
# expected_obs = expected_obs.at[:,12:60].set(0.1)
# expected_obs = expected_obs.at[:,60].set(0.9)
# expected_obs = expected_obs.at[:,61].set(0.7)
# expected_obs = expected_obs.at[:,62].set(0.4)
# expected_obs = expected_obs.at[:,63].set(0.6)
# expected_obs = expected_obs.at[:,64].set(0.8)

# print(sum_age_to(expected_obs, 12*5, AGE_GROUPS))

######### interval between first and second birth in months
# # percent with no second birth, within 24 months, 25-36 months, 37-48 months, and 49+ months
# percent = np.array([29.5, 18.7, 17.3, 10.7, 23.9])/100
# # renormalize to look at kids with at least one older sibling (i.e. get rid of no second birth)
# percent = percent[1:]/np.sum(percent[1:])

# weights = np.array([0.11007698, 0.33023093, 1.79616209, 0.63537594 ])
# weights = weights/np.sum(weights)

# print(np.sum(percent[0:3]), percent[-1])
# print(np.sum(weights[0:3]), weights[-1])


####### plot difference between aging at an exponential rate vs aging fully realistically
# compartments = jnp.zeros(4)
# time_in_age_group = jnp.array([3/12*365,9/12*365,4*365,80*365])
# age_rate = 1/time_in_age_group
# birth_rate = 1/(80*365)
# # run simulation of birth and aging until steady state
# def birth_and_aging(t, compartments):
#     aging = age_rate * compartments
#     dcompartments_dt = jnp.array([birth_rate, aging[0], aging[1], aging[2]]) - aging
#     return dcompartments_dt
# from scipy.integrate import solve_ivp
# t_eval = np.linspace(0, 80*365*5, 10000)
# start_time = time.time()
# sol = solve_ivp(birth_and_aging, [0, 80*365*5], compartments, t_eval=t_eval)
# print('done in', time.time() - start_time, 'seconds')

# # try with more compartments (and sum compartments to same age groups)
# compartments2 = jnp.zeros(4*5+1)
# time_in_age_group2 = jnp.array([1/4*365]*4*5 + [80*365])
# age_rate2 = 1/time_in_age_group2
# # run simulation of birth and aging until steady state
# def birth_and_aging2(t, compartments2):
#     aging = age_rate2 * compartments2
#     dcompartments_dt = jnp.zeros_like(compartments2)
#     dcompartments_dt = dcompartments_dt.at[0].add(birth_rate)
#     dcompartments_dt = dcompartments_dt.at[1:].add(aging[:-1])
#     dcompartments_dt = dcompartments_dt - aging
#     return dcompartments_dt
# from scipy.integrate import solve_ivp
# t_eval = np.linspace(0, 80*365*5, 10000)
# start_time = time.time()
# sol2 = solve_ivp(birth_and_aging2, [0, 80*365*5], compartments2, t_eval=t_eval)
# print('done in', time.time() - start_time, 'seconds')
# values = sol2.y.T
# # sum into same age groups as first simulation
# age_group_compartments2 = jnp.array([jnp.sum(values[:,0:1], axis=1),
#                                      jnp.sum(values[:,1:4], axis=1),
#                                     jnp.sum(values[:,4:-1], axis=1),
#                                      jnp.sum(values[:,-1:], axis=1)])

# # try with more compartments (and sum compartments to same age groups)
# compartments3 = jnp.zeros(12*5+1)
# time_in_age_group3 = jnp.array([1/12*365]*12*5 + [80*365])
# age_rate3 = 1/time_in_age_group3
# # run simulation of birth and aging until steady state
# def birth_and_aging3(t, compartments3):
#     aging = age_rate3 * compartments3
#     dcompartments_dt = jnp.zeros_like(compartments3)
#     dcompartments_dt = dcompartments_dt.at[0].add(birth_rate)
#     dcompartments_dt = dcompartments_dt.at[1:].add(aging[:-1])
#     dcompartments_dt = dcompartments_dt - aging
#     return dcompartments_dt
# from scipy.integrate import solve_ivp
# t_eval = np.linspace(0, 80*365*5, 10000)
# start_time = time.time()
# sol3 = solve_ivp(birth_and_aging3, [0, 80*365*5], compartments3, t_eval=t_eval)
# print('done in', time.time() - start_time, 'seconds')
# values = sol3.y.T
# # sum into same age groups as first simulation
# age_group_compartments3 = jnp.array([jnp.sum(values[:,0:3], axis=1),
#                                      jnp.sum(values[:,3:12], axis=1),
#                                     jnp.sum(values[:,12:-1], axis=1),
#                                      jnp.sum(values[:,-1:], axis=1)])

# # run alternative simulation where people spend exactly the time in each age group (i.e. non-markov)
# sim_days = 80*365*5
# daily_compartments = jnp.zeros((sim_days,81*365))
# start_time = time.time()
# def simulate_birth_and_aging(sim_days, birth_rate):
#     def body_fn(day, daily_compartments):
#         daily_compartments = daily_compartments.at[day, 1:].set(daily_compartments[day-1, :-1])
#         daily_compartments = daily_compartments.at[day, 0].set(birth_rate)
#         return daily_compartments
    
#     daily_compartments = jnp.zeros((sim_days, 81*365))
#     daily_compartments = jax.lax.fori_loop(1, sim_days, body_fn, daily_compartments)
#     return daily_compartments

# start_time = time.time()
# daily_compartments = jax.jit(simulate_birth_and_aging, static_argnums=(0,))(sim_days, birth_rate)
# print('done in', time.time() - start_time, 'seconds')
# # sum up age groups to get same age groups as in the first simulation
# age_group_compartments = jnp.array([jnp.sum(daily_compartments[:,0:int(3/12*365)], axis=1),
#                                     jnp.sum(daily_compartments[:,int(3/12*365):int(365)], axis=1),
#                                     jnp.sum(daily_compartments[:,int(365):int(5*365)], axis=1),
#                                     jnp.sum(daily_compartments[:,int(5*365):], axis=1)])

# # save sol and daily_compartments
# with open("Outputs/birth_and_aging_comparison.pkl", "wb") as f:
#     pickle.dump((sol, age_group_compartments), f)

# # Define colors for age groups and line styles for simulation types
# age_group_colors = ['#648FFF', '#DC267F', '#FFB000', '#785EF0']  # Blue, Pink, Orange, Purple
# age_group_labels = ["0-3m", "3-11m", "1-4y", "5y+"]
# linestyles = ['-', ':', '--', '-.']
# simulation_labels = ['Markov (3 compartments)', 'Markov (4*5 compartments)', 'Markov (12*5 compartments)', 'Non-Markov (exact)']

# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot exponential model
# for i, (color, label) in enumerate(zip(age_group_colors, age_group_labels)):
#     ax.plot(sol.t/365, sol.y[i], color=color, linestyle='-', linewidth=2, label=f'{label} - {simulation_labels[0]}')

# # Plot Markov model
# for i, (color, label) in enumerate(zip(age_group_colors, age_group_labels)):
#     ax.plot(sol2.t/365, age_group_compartments2[i], color=color, linestyle=':', linewidth=2, label=f'{label} - {simulation_labels[1]}')

# # Plot Markov model (12*5 compartments)
# for i, (color, label) in enumerate(zip(age_group_colors, age_group_labels)):
#     ax.plot(sol3.t/365, age_group_compartments3[i], color=color, linestyle='--', linewidth=2, label=f'{label} - {simulation_labels[2]}')

# # Plot non-Markov model
# for i, (color, label) in enumerate(zip(age_group_colors, age_group_labels)):
#     ax.plot(np.arange(sim_days)/365, age_group_compartments[i], color=color, linestyle='-.', linewidth=2, label=f'{label} - {simulation_labels[3]}')

# ax.set_xlabel('Time (years)')
# ax.set_ylabel('Compartment value')
# ax.legend(frameon=False, fontsize=9, ncol=4)
# plt.tight_layout()
# plt.show()




# labels = ["RSV", "Metapneumovirus", "InfluenzaA", "InfluenzaB", "Adenovirus", "Parainfluenza3"]
# x = np.array([0.05055, 0.01256, 0.5846, 0.1926, 0.9883, 0.6028])
# y = np.array([0.2908, 0.01664, 0.6762, 0.5275, 0.9916, 0.9820])
# # # line of best fit with constraint that coefficients sum to 1
# # # fit: y = a*x + b where a + b = 1, so b = 1 - a
# # # minimize: sum((y - (a*x + (1-a)))^2)
# # def objective(a):
# #     y_pred = a * x + (1 - a)
# #     return np.sum((y - y_pred) ** 2)

# # a_opt = sp.optimize.minimize_scalar(objective, bounds=(0, 1), method='bounded').x
# # b_opt = 1 - a_opt
# a_opt, b_opt = 0.75, 0.25
# print("Line of best fit (constrained): y = {:.3f}x + {:.3f}".format(a_opt, b_opt))
# plt.plot(x, a_opt*x + b_opt, color='r', label='Line of Best Fit (Constrained)')
# plt.scatter(x, y)
# for i, label in enumerate(labels):
#     plt.annotate(label, (x[i], y[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
# plt.show()

#### latin hypercube sampling of parameter space and likelihood evaluation
# # BETA follows a lognormal distribution with median 0.15
# def BETA_distribution(key):
#     return jax.random.normal(key, shape=()) * 0.5 + np.log(0.2)

# def SEASONALITY_distribution(key):
#     return jax.random.normal(key, shape=()) * 0.5 + np.log(0.1)

# def OFFSET_distribution(key):
#     return jax.random.normal(key, shape=()) * 0.1 + 0.15

# def WANE2_distribution(key):
#     return jax.random.normal(key, shape=()) * 0.5 + np.log(0.005)

# def REL_distribution(key):
#     return jax.random.uniform(key, shape=(), minval=0.1, maxval=1.0)

# def F1_distribution(key):
#     return jax.random.uniform(key, shape=(), minval=0.0, maxval=1.0)

# def R1_distribution(key):
#     return jax.random.uniform(key, shape=(), minval=0.002, maxval=0.01)

# def OBS_distribution(key):
#     return jax.random.uniform(key, shape=(), minval=0.0, maxval=1.0)

# def quantile_to_params(quantiles, pathogen="RSV"):
#     """Convert quantiles [0,1] to parameter values using inverse CDF"""
#     BETA = jnp.exp(jsp.special.erfinv(2*quantiles[0] - 1) * 0.5 + np.log(0.2))
#     SEASONALITY = jnp.exp(jsp.special.erfinv(2*quantiles[1] - 1) * 0.5 + np.log(0.1))
#     OFFSET = jsp.special.erfinv(2*quantiles[2] - 1) * 0.1 + 0.15
#     WANE2 = jnp.exp(jsp.special.erfinv(2*quantiles[3] - 1) * 0.5 + np.log(0.005))
#     S_REL1 = 0.1 + quantiles[4] * 0.9
#     S_REL2 = 0.1 + quantiles[5] * 0.9
#     n = 5
#     if pathogen != "RSV":
#         D_REL1 = 0.1 + quantiles[n] * 0.9
#         D_REL2 = 0.1 + quantiles[n+1] * 0.9
#     F1 = quantiles[n]
#     R1 = 0.002 + quantiles[n+1] * 0.008
#     AGE_OBS = jnp.array([quantiles[n+2+i] for i in range(7)])
#     if pathogen != "RSV":
#         return jnp.concatenate([jnp.array([BETA, SEASONALITY, OFFSET, WANE2, S_REL1, S_REL2, D_REL1, D_REL2, F1, R1]), AGE_OBS])
#     return jnp.concatenate([jnp.array([BETA, SEASONALITY, OFFSET, WANE2, S_REL1, S_REL2, F1, R1]), AGE_OBS])

# from fit_opt import get_likelihood
# key = jax.random.PRNGKey(0)

# from Parameters.census_population import CENSUS_AGE_POP_split as CENSUS_AGE_POP
# from fit_opt import latin_hypercube_sample

# pathogens = ["RSV", "Metapneumovirus", "InfluenzaA", "InfluenzaB", "Adenovirus", "Parainfluenza3"]

# chunk_size = 1000
# total_samples = 1_000_000

# for p_idx, pathogen in enumerate(pathogens):
#     print(f"\n=== {pathogen} ({p_idx+1}/{len(pathogens)}) ===")

#     likelihood, _ = get_likelihood(
#         pathogen,
#         "Exponential",
#         "split",
#         "nrdaycare5maxagep028",
#         1e-9,
#         normalize=False,
#         CENSUS_AGE_POP=CENSUS_AGE_POP,
#         NAG=8,
#     )

#     # RSV has 15 params in quantile_to_params, non-RSV has 17
#     n_params = 15 if pathogen == "RSV" else 17

#     print(f"Generating {total_samples} LHS samples for {pathogen}...")
#     key, subkey = jax.random.split(key)
#     lhs_samples = latin_hypercube_sample(subkey, total_samples, n_params)

#     vmap_params = jax.jit(jax.vmap(lambda q: quantile_to_params(q, pathogen=pathogen)))
#     vmap_likelihood = jax.jit(jax.vmap(likelihood))

#     parameter_samples = []
#     likelihoods = []

#     start_time = time.time()
#     n_chunks = (total_samples + chunk_size - 1) // chunk_size

#     for i in range(0, total_samples, chunk_size):
#         chunk_id = i // chunk_size + 1
#         chunk_end = min(i + chunk_size, total_samples)

#         elapsed = time.time() - start_time
#         done_chunks = max(1, chunk_id - 1)
#         eta_min = ((n_chunks - chunk_id) * (elapsed / done_chunks)) / 60

#         print(f"[{pathogen}] chunk {chunk_id}/{n_chunks}, ETA: {eta_min:.2f} min")

#         chunk_lhs = lhs_samples[i:chunk_end]
#         chunk_params = vmap_params(chunk_lhs)
#         chunk_likelihoods = vmap_likelihood(chunk_params)

#         parameter_samples.append(chunk_params)
#         likelihoods.append(chunk_likelihoods)

#     parameter_samples = jnp.concatenate(parameter_samples, axis=0)
#     likelihoods = jnp.concatenate(likelihoods, axis=0)

#     with open(f"Outputs/{pathogen}_parameter_samples_lhs.pkl", "wb") as f:
#         pickle.dump((parameter_samples, likelihoods), f)

#     min_idx = jnp.nanargmin(likelihoods)
#     print(f"[{pathogen}] Minimum likelihood: {likelihoods[min_idx]}")
#     print(f"[{pathogen}] Corresponding parameters: {parameter_samples[min_idx]}")

#     # plot PMF of each parameter based on inverse-likelihood weights
#     n_plot_params = int(parameter_samples.shape[1])
#     parameter_names = [f"param_{i}" for i in range(n_plot_params)]

#     fig, axes = plt.subplots(5, 4, figsize=(16, 10))
#     axes = axes.flatten()

#     weights = 1.0 / (1.0 + np.array(likelihoods))
#     weights = weights / np.sum(weights)

#     for i, ax in enumerate(axes):
#         if i < n_plot_params:
#             ax.hist(
#                 np.array(parameter_samples[:, i]),
#                 bins=50,
#                 weights=weights,
#                 alpha=0.7,
#                 color="steelblue",
#                 edgecolor="black",
#             )
#             ax.set_title(parameter_names[i])
#             ax.set_xlabel(parameter_names[i])
#             ax.set_ylabel("Probability Mass")
#         else:
#             ax.axis("off")

#     plt.tight_layout()
#     plt.savefig(f"Figures/{pathogen}_parameter_pmf_from_likelihood_lhs.png", dpi=300)
#     plt.close(fig)



###### spectrum analysis
# N = date_to_t("2020-03-19")-date_to_t('2015-10-01')
# # N = date_to_t("2025-05-01")-date_to_t('2015-10-01')

# fig, axes = plt.subplots(3, 2, figsize=(13.3, 7.5), sharex=True)
# pathogens = ["InfluenzaA", "InfluenzaB", "RSV", "Metapneumovirus",  "Parainfluenza3", "Adenovirus"]

# for ax, pathogen in zip(axes.flatten(), pathogens):
#     data = pd.read_csv(f"Data/Processed/KPSC_panel_proportion_positive_ARI_nonCOVID_{pathogen}_incidence_age_daily.csv", index_col=0, parse_dates=True)
#     for i, age_group in enumerate(AGE_GROUP_NAMES):
#         age_data = data[age_group].values
#         ft = sp.fft.fft(age_data[:N])
#         ft = ft / np.mean(np.abs(ft))  # Normalize the FFT
#         xf = sp.fft.fftfreq(N, d=1/365)[:N//2]
#         # plot the power spectrum
#         ax.plot(xf,2.0/N * np.abs(ft[:N//2])**2, color=hsv_colors[i], label=age_group)
#     age_summed_data = data[AGE_GROUP_NAMES].sum(axis=1)
#     ft = sp.fft.fft(age_summed_data[:N])
#     ft = ft / np.mean(np.abs(ft))  # Normalize the FFT
#     xf = sp.fft.fftfreq(N, d=1/365)[:N//2]
#     ax.plot(xf,2.0/N * np.abs(ft[:N//2])**2, color='k', label='All ages')
#     ax.set_title(pathogen)
#     ax.set_xlabel("Frequency")
#     ax.set_ylabel("Power")
#     ax.set_xlim(0,5)
# axes[0,0].legend(frameon=False)
# fig.suptitle("Power spectrum of observed incidence by age group", fontsize=16)
# plt.tight_layout()
# plt.savefig("Figures/power_spectrum_of_observed_incidence_age_normalized_pre2020.png", dpi=300)


# ##### plotting likelihood functions

# n_p = 10
# n = 30
# H = 70
# N = 8e5

# inc = 3e-5

# print(np.round((n_p / n) * H))


# def poisson_probability(n_p, n, H, inc):
#     k = np.round((n_p / n) * H)
#     l = inc * N
#     return jsp.stats.poisson.pmf(k, l)

# def binomial_probability(n_p, n, H, inc):
#     p = inc * N / H
#     return jsp.stats.binom.pmf(n_p, n, p)

# # four-panel comparison: vary n_p, n, H, and l
# fig, axes = plt.subplots(2, 2, figsize=(8, 7))
# ax1, ax2, ax3, ax4 = axes.flatten()

# # Panel 1: vary n_p
# n_p_values = np.arange(0, 49)
# poisson_probs = [poisson_probability(npi, n, H, inc) for npi in n_p_values]
# binomial_probs = [binomial_probability(npi, n, H, inc) for npi in n_p_values]
# poisson_probs = np.array(poisson_probs) / np.sum(poisson_probs)  # Normalize to sum to 1
# binomial_probs = np.array(binomial_probs) / np.sum(binomial_probs)  # Normalize to sum to 1
# ax1.plot(n_p_values, poisson_probs, label="Poisson", color="blue")
# ax1.plot(n_p_values, binomial_probs, label="Binomial", color="orange")
# ax1.axvline(n_p_values[np.argmax(poisson_probs)], color="blue", linestyle="--", alpha=0.7)
# ax1.axvline(n_p_values[np.argmax(binomial_probs)], color="orange", linestyle="--", alpha=0.7)
# ax1.set_title("Vary n_p")
# ax1.set_xlabel("n_p")
# ax1.set_ylabel("Probability")
# ax1.legend(frameon=False)

# # Panel 2: vary n # n_fixed_values = np.arange(20, H, 5)
# poisson_n = [poisson_probability(n_p, ni, H, inc) for ni in n_fixed_values]
# binomial_n = [binomial_probability(n_p, ni, H, inc) for ni in n_fixed_values]
# poisson_n = np.array(poisson_n) / np.sum(poisson_n)  # Normalize to sum to 1
# binomial_n = np.array(binomial_n) / np.sum(binomial_n)  # Normalize to sum to 1
# ax2.plot(n_fixed_values, poisson_n, color="blue")
# ax2.plot(n_fixed_values, binomial_n, color="orange")
# ax2.axvline(n_fixed_values[np.argmax(poisson_n)], color="blue", linestyle="--", alpha=0.7)
# ax2.axvline(n_fixed_values[np.argmax(binomial_n)], color="orange", linestyle="--", alpha=0.7)
# ax2.set_title(f"Vary n")
# ax2.set_xlabel("n")
# ax2.set_ylabel("Probability")

# # Panel 3: vary H # H_values = np.arange(50, 150, 10)
# poisson_H = [poisson_probability(n_p, n, Hi, inc) for Hi in H_values]
# binomial_H = [binomial_probability(n_p, n, Hi, inc) for Hi in H_values]
# poisson_H = np.array(poisson_H) / np.sum(poisson_H)  # Normalize to sum to 1
# binomial_H = np.array(binomial_H) / np.sum(binomial_H)  # Normalize to sum to 1
# ax3.plot(H_values, poisson_H, color="blue")
# ax3.plot(H_values, binomial_H, color="orange")
# ax3.axvline(H_values[np.argmax(poisson_H)], color="blue", linestyle="--", alpha=0.7)
# ax3.axvline(H_values[np.argmax(binomial_H)], color="orange", linestyle="--", alpha=0.7)
# ax3.set_title(f"Vary H")
# ax3.set_xlabel("H")
# ax3.set_ylabel("Probability")

# # Panel 4: vary l # inc_values = np.logspace(-6.5, np.log10(H/N), 120)
# poisson_l = [poisson_probability(n_p, n, H, inci) for inci in inc_values]
# binomial_l = [binomial_probability(n_p, n, H, inci) for inci in inc_values]
# poisson_l = np.array(poisson_l) / np.sum(poisson_l)  # Normalize to sum to 1
# binomial_l = np.array(binomial_l) / np.sum(binomial_l)  # Normalize to sum to 1
# ax4.plot(inc_values, poisson_l, color="blue")
# ax4.plot(inc_values, binomial_l, color="orange")
# ax4.axvline(inc_values[np.argmax(poisson_l)], color="blue", linestyle="--", alpha=0.7)
# ax4.axvline(inc_values[np.argmax(binomial_l)], color="orange", linestyle="--", alpha=0.7)
# ax4.set_xscale("log")
# ax4.set_title(f"Vary inc")
# ax4.set_xlabel("inc")
# ax4.set_ylabel("Probability")

# # give fixed values in overall title
# fig.suptitle(f"Poisson vs Binomial Probability Comparison\n(n_p={n_p}, n={n}, H={H}, inc={inc})", fontsize=16)

# plt.tight_layout()
# plt.savefig("Figures/poisson_binomial_four_panel_comparison_high_incidence.png", dpi=300)

# #### plotting contact funcitons
# EPOCH = pd.to_datetime('1970-01-01')
# END = pd.to_datetime("2025-05-01")
# FULL_PERIOD = pd.date_range(start=EPOCH, end=END, freq='D')
# FULL_POINTS = jnp.array(date_to_t(FULL_PERIOD))
# data_start = date_to_t('2015-10-01')

# from contact_model import exponential_recovery_byage
# x = [0.109, 0.001, 0.1]
# n=0
# FF = [1,x[n]]
# TT = [date_to_t(EPOCH), date_to_t('2020-03-19')]
# RR = jnp.array([[x[n+1],x[n+2],]],)
# EXPONENTIAL_CONTACT = exponential_recovery_byage(FULL_POINTS, TT, FF, RR, age_partition=5)

# for age_group in AGE_GROUP_NAMES:
#     plt.plot(FULL_POINTS, EXPONENTIAL_CONTACT[:,AGE_GROUP_NAMES.index(age_group)], label=age_group, color=hsv_colors[AGE_GROUP_NAMES.index(age_group)])
# plt.legend()
# plt.xlim(date_to_t('2020-01-01'), date_to_t('2025-05-01'))
# plt.xlabel("Time # plt.ylabel("Relative contact rate")
# plt.title("Exponential recovery contact function by age group")
# plt.show()

# # x=[0.8,0.6,0.5,0.02,0.02]
# # n=0
# # FF = [1, x[n],x[n+1]]
# # TT = [date_to_t(EPOCH), date_to_t('2020-03-19'), date_to_t('2020-03-19')+x[n+2]*365]
# # RR = [x[n+3],x[n+4]]

# x=[0.8,1,0.005]
# n=0
# FF = [1, x[n]]
# TT1 = [date_to_t(EPOCH), date_to_t('2020-03-19'), date_to_t('2020-03-19')+x[n+1]*365]
# TT2 = [date_to_t(EPOCH), date_to_t('2020-03-19'), date_to_t('2020-03-19')+0.01*365]
# TT3 = [date_to_t(EPOCH), date_to_t('2020-03-19'), date_to_t('2020-03-19')+2*365]
# RR = [x[n+2],]

# def exponential_recovery(t, ts, fs, rs, steepness=0.2):
#     """Sudden tanh reduction at ts[i] with exponential recovery to fs[0]"""
#     result = fs[0]
#     for i in range(1, len(ts)):
#         # Sudden tanh reduction at ts[i]
#         reduction = 0.5 * #         # Exponential recovery back to fs[0]
#         recovery = jnp.exp(-rs[i-1] * jnp.maximum(0, t - ts[i]))
#         # Combine: reduce to fs[i], then recover toward fs[0]
#         transition = fs[i] + #         result = result * #     return result
# def sigmoidal_recovery(t, ts, fs, rs, steepness=0.2):
#     """Sudden tanh reduction at ts[i] with sigmoidal recovery to fs[0]"""
#     result = fs[0]
#     # Sudden tanh reduction at ts[i]
#     reduction = #     # Sigmoidal recovery back to fs[0]
#     recovery = #     # Combine: reduce to fs[i], then recover toward fs[0]
#     transition = fs[1] + #     result = result * #     return result
# # MOBILITY_CONTACT = exponential_recovery(FULL_POINTS, TT, FF, RR)
# SIGMOIDAL_CONTACT = sigmoidal_recovery(FULL_POINTS, TT2, FF, [0.002,])
# SIGMOIDAL_CONTACT2 = sigmoidal_recovery(FULL_POINTS, TT2, FF, [0.01,])
# SIGMOIDAL_CONTACT3 = sigmoidal_recovery(FULL_POINTS, TT2, FF, [0.005,])
# # plt.plot(FULL_PERIOD[data_start:], MOBILITY_CONTACT[data_start:], color='k')
# plt.plot(FULL_PERIOD[data_start:], SIGMOIDAL_CONTACT[data_start:], color='r')
# plt.plot(FULL_PERIOD[data_start:], SIGMOIDAL_CONTACT2[data_start:], color='g')
# plt.plot(FULL_PERIOD[data_start:], SIGMOIDAL_CONTACT3[data_start:], color='b')
# plt.plot()
# plt.show()

# # # ### plotting functions for optimization results
# daily_hospitalization_rates_pd = pd.read_csv('Data/Processed/KPSC_ARI_nonCOVID_hospitalization_rates_by_day_age_group_split.csv',index_col=0,parse_dates=True)
# N = jnp.prod(jnp.asarray(daily_hospitalization_rates_pd.shape))

# for pathogen, maxage in [("Metapneumovirus", 2)]:
#     # plot for just one result
#     filepath = f"Data/Processed/results260416/evosax_DE_{pathogen}ExponentialInOutmaxmimmsplitdaycarep5maxagep0{maxage}260416.pickle"
#     with open(filepath, "rb") as f:
#         results = pickle.load(f)
#     print(results)
#     metrics_log = results["metrics_log"]
#     generations = metrics_log["generation_counter"]
#     best_solution = jnp.asarray(metrics_log["best_solution_in_generation"])
#     n_params = best_solution.shape[1]

#     param_names, bounds = parameters_names_bounds(pathogen, "ExponentialInOut", "maxmimmsplit", "daycarep5maxagep0"+str(maxage),NAG=8)
#     print(param_names)
#     # Create subplots
#     n_cols = 5
#     n_rows = int(np.ceil(n_params / n_cols))
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2*n_rows))
#     axes = axes.flatten()

#     # Plot each parameter trajectory
#     for i in range(n_params):
#         axes[i].plot(generations, best_solution[:, i], color='k', alpha=0.7)
#         axes[i].set_xlabel('')
#         axes[i].set_ylabel('')
#         axes[i].set_title(f'{param_names[i]} Trajectory')
#         axes[i].set_ylim(bounds[i])
#         axes[i].grid(True, alpha=0.3)

#     # Hide unused subplots
#     for i in range(n_params, len(axes)):
#         axes[i].set_visible(False)

#     plt.tight_layout()
#     plt.savefig("Figures/"+filepath.split("/")[-1].replace(".pickle", "_parameter_trajectories.png"), dpi=300)

#     # plot range of population fitness over genrations
#     for pathogen, maxage in [("Metapneumovirus", 2)]:
#         filepath = f"Data/Processed/results260416/evosax_DE_{pathogen}ExponentialInOutmaxmimmsplitdaycarep5maxagep0{maxage}260416.pickle"
#         with open(filepath, "rb") as f:
#             results = pickle.load(f)
#         print(results)
#         metrics_log = results["metrics_log"]
#         generations = metrics_log["generation_counter"]
#         best_fitness = jnp.asarray(metrics_log["best_fitness_in_generation"]) * N

#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.plot(generations, best_fitness, label="Best Fitness", color='k')
#         ax.set_title(f"{pathogen} Fitness over Generations")
#         ax.set_xlabel("Generation")
#         ax.set_ylabel("Negative Log Likelihood")
#         ax.grid(True, alpha=0.3)
#         ax.legend()
        
#         # Add inset zooming in on last 500 generations
#         axins = inset_axes(ax, width="40%", height="40%", loc="upper right")
#         last_500_idx = max(0, len(generations) - 500)
#         axins.plot(generations[last_500_idx:], best_fitness[last_500_idx:], color='k', linewidth=1.5)
#         axins.set_title("Last 500 Gen", fontsize=8)
#         axins.grid(True, alpha=0.3)
#         axins.tick_params(labelsize=8)
        
#         plt.tight_layout()
#         plt.savefig("Figures/"+filepath.split("/")[-1].replace(".pickle", "_fitness_over_generations.png"), dpi=300)
#         # plot rolling standard deviation of fitness over last 100 generations
#         rolling_std = pd.Series(best_fitness).rolling(window=100).std()
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.plot(generations, rolling_std, label="Rolling Std Dev", color='r')
#         ax.set_title(f"{pathogen} Log Likelihood Rolling Std Dev over Generations")
#         ax.set_xlabel("Generation")
#         ax.set_ylabel("Rolling Std Dev")
#         ax.grid(True, alpha=0.3)
        
#         # Add inset zooming in on last 500 generations
#         axins = inset_axes(ax, width="40%", height="40%", loc="upper right")
#         last_500_idx = max(0, len(generations) - 500)
#         axins.plot(generations[last_500_idx:], rolling_std[last_500_idx:], color='r', linewidth=1.5)
#         axins.set_title("Last 500 Gen", fontsize=8)
#         axins.grid(True, alpha=0.3)
#         axins.tick_params(labelsize=8)
        
#         ax.legend()
#         plt.tight_layout()
#         plt.savefig("Figures/"+filepath.split("/")[-1].replace(".pickle", "_rolling_std_dev.png"), dpi=300)
#         plt.close()


# # plot for a bunch of results
# plt.figure(figsize=(10, 5))

# # Iterate over different seeds
# # seed_suffixes = ['24', '242', '243', '244', '245', '246']
# # populations = ["20*20", "20*20", "20*20", "10*20", "10*20", "10*20", "20*20", "100*20", "100*20"]
# populations = ["200*20", "200*20", "200*20", "200*20", "200*20", "200*20"]
# pathogens = ["RSV", "Metapneumovirus", "InfluenzaA", "Parainfluenza3", "Adenovirus", "InfluenzaB"]
# population_colors = {
#     "10*20": "#648FFF", 
#     "20*20": "#DC267F",
#     "100*20": "#FF832B", 
#     "200*20": "#FFB000"
# }
# colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#000000"]

# for i, seed_suffix in enumerate(pathogens):
#     pathogen = pathogens[i]
#     filepath = f"Data/Processed/results260324/evosax_DE_{pathogen}SigmoidNAflexagep01260324.pickle"
#     try:
#         with open(filepath, "rb") as f:
#             results = pickle.load(f)
#         metrics_log = results["metrics_log"]
#         generations = metrics_log["generation_counter"]
#         best_fitness = jnp.asarray(metrics_log["best_fitness"]) * N
#         # color = population_colors[populations[i]]
#         plt.plot(generations, best_fitness, label=pathogen, alpha=0.7, color=colors[i])
#     except FileNotFoundError:
#         print(f"File not found for {pathogen} seed 260324")

# # for i, seed_suffix in enumerate(seed_suffixes):
# #     filepath = f"Data/Processed/results260303/evosax_DE_RSVFlexStepwiseincidence_dataflexagep012603{seed_suffix}.pickle"
# #     try:
# #         with open(filepath, "rb") as f:
# #             results = pickle.load(f)
# #         metrics_log = results["metrics_log"]
# #         generations = metrics_log["generation_counter"]
# #         best_fitness = jnp.asarray(metrics_log["best_fitness"]) * N
# #         color = population_colors[populations[i]]
# #         plt.plot(generations, best_fitness, label=f"Pop: {populations[i]}, inc.", marker=".", markersize=3, alpha=0.7, color=color)
# #     except FileNotFoundError:
# #         print(f"File not found for seed 2603{seed_suffix}")

# # for i, seed_suffix in enumerate(seed_suffixes):
# #     filepath = f"Data/Processed/results260303/evosax_DE_RSVFlexStepwisesmoothedincidence_dataflexagep012603{seed_suffix}.pickle"
# #     try:
# #         with open(filepath, "rb") as f:
# #             results = pickle.load(f)
# #         metrics_log = results["metrics_log"]
# #         generations = metrics_log["generation_counter"]
# #         best_fitness = jnp.asarray(metrics_log["best_fitness"]) * N
# #         color = population_colors[populations[i]]
# #         plt.plot(generations, best_fitness, label=f"Pop: {populations[i]}, sm. inc.", marker="+", markersize=3, alpha=0.7, color=color)
# #     except FileNotFoundError:
# #         print(f"File not found for seed 2603{seed_suffix}")

# # filepath = f"Data/Processed/results260309/evosax_DE_RSVFlexStepwiseNAflexagep01260309.pickle"
# # with open(filepath, "rb") as f:
# #     results = pickle.load(f)
# # metrics_log = results["metrics_log"]
# # generations = metrics_log["generation_counter"]
# # best_fitness = jnp.asarray(metrics_log["best_fitness"]) * N
# # color = population_colors["200*20"]
# # plt.plot(generations, best_fitness, label=f"Pop: {"200*20"}", marker="o", markersize=3, alpha=0.7, color=color)

# # filepath = f"Data/Processed/results260309/evosax_DE_RSVFlexStepwiseNAflexagep012603092.pickle"
# # with open(filepath, "rb") as f:
# #     results = pickle.load(f)
# # metrics_log = results["metrics_log"]
# # generations = metrics_log["generation_counter"]
# # best_fitness = jnp.asarray(metrics_log["best_fitness"]) * N
# # color = population_colors["200*20"]
# # plt.plot(generations, best_fitness, label=f"Pop: {"200*20"}", marker="o", markersize=3, alpha=0.7, color=color)


# plt.title("Best Fitness over generations")
# plt.xlabel("Generation")
# plt.ylabel("Fitness")
# plt.grid(True, alpha=0.3)
# # plt.yscale('log')
# # plt.xscale('log')
# # plt.xlim(500,1000)
# # plt.ylim(10800,11000)
# plt.legend()
# plt.tight_layout()
# plt.savefig("Figures/evosax_DE_RSVSigmoidmimmflexagep01_all_seeds_fitness_comparison260324.png", dpi=300)

# # # ##### Testing optax ######
# pathogen, seed, lockdown, option1, option2, import_multiplier = "RSV", 260224, "FlexStepwise", "NA", "flexagep01", 1e-9

# for pathogen in ["RSV", "Metapneumovirus", "InfluenzaA", "Parainfluenza3", "Adenovirus"]:
#     results_file = "Data/Processed/results"+str(seed)[:6]+"/optax_"+pathogen+lockdown+option1+option2+str(seed)+".pickle"
#     with open(results_file, "rb") as f:
#         results = pickle.load(f)
#         final_xs = results["final_xs"]
#         neglogL_history = results["neglogL_history"]
#     print(final_xs)
#     daily_hospitalization_rates_pd = pd.read_csv('Data/Processed/KPSC_ARI_hospitalization_rates_by_day_age_group.csv',index_col=0,parse_dates=True)
#     N = jnp.prod(jnp.asarray(daily_hospitalization_rates_pd.shape))
#     neglogL_history = jnp.asarray(neglogL_history)*N
#     fig, ax = plt.subplots(3,2,figsize=(8,6), sharex='col')
#     for i in range(3):
#         # Full trajectory
#         ax[i,0].plot(neglogL_history[:,i], color='k')
#         ax[i,0].set_ylabel('Negative log-likelihood')
#         ax[i,0].set_yscale('log')
#         ax[i,0].set_title(f'Optimization trajectory #         if i == 2:
#             ax[i,0].set_xlabel('Iteration')
        
#         # Zoomed in on last 200 iterations
#         ax[i,1].plot(neglogL_history[-200:,i], color='k')
#         ax[i,1].set_yscale('log')
#         ax[i,1].set_title(f'Last 200 iterations #         if i == 2:
#             ax[i,1].set_xlabel('Iteration')

#     plt.tight_layout()
#     plt.savefig(f"Figures/{pathogen}{seed}_optax_optimization_trajectory.png", dpi=300)

# import sys
# sys.argv = [sys.argv[0], pathogen, str(seed), lockdown, option1, option2, str(import_multiplier), "20", "1", "0.7"]
# from fit_opt import likelihood
# jlikelihood = jax.jit(likelihood)
# # with open("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","rb") as f:
# #     opt = pickle.load(f)
# # x = jnp.array(opt.x)

# # x = jnp.array([0.3930, 0.0488, 0.2457, 0.0036, 0.1792, 0.1622, 0.9081, 0.4926, 0.8474, 0.4486, 0.9121, 0.8380, 1.0000, 0.0100, 0.0064, 0.0046, 0.0016, 0.0002, 0.0010, 0.0096])
# # gradient = jax.grad(jlikelihood)(x)
# # print(jnp.linalg.norm(gradient))


# import itertools
# param_names, bounds = parameters_names_bounds(pathogen, lockdown, option1, option2)
# valid_sample = False
# while not valid_sample:
#     x = jnp.array([np.random.uniform(bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))])
#     if likelihood(x) < 10:
#         valid_sample = True

# jlikelihood = jax.jit(likelihood)
# start_likelihood = jlikelihood(x)
# print(f"Negative log-likelihood: {start_likelihood:.2f}")

# import optax
# solver = optax.adabelief(learning_rate=0.003)
# opt_state = solver.init(x)
# jgrad = jax.jit(jax.grad(jlikelihood))
# for i in range(1000):
#     start_time = time.time()
#     grad = jgrad(x)
#     update, opt_state = solver.update(grad, opt_state, x)
#     x = optax.apply_updates(x, update)
#     x = optax.projections.projection_box(x, bounds[:,0], bounds[:,1])
#     current_likelihood = jlikelihood(x)
#     print(f"Negative log-likelihood: {current_likelihood:.2f}")
#     if current_likelihood < start_likelihood:
#         print("Parameters improved:", end=" ")
#         for name, value in zip(param_names, x):
#             print(f"{name}: {value:.4f}", end=", ")
#         print()
#         start_likelihood = current_likelihood
#     elif i % 100 == 0:
#         print("No improvement, current parameters:", end=" ")
#         for name, value in zip(param_names, x):
#             print(f"{name}: {value:.4f}", end=", ")
#         print()
#     # check if gradient is close to zero
#     if jnp.linalg.norm(grad) < 1e-3:
#         print("Gradient close to zero, stopping optimization.")
#         break
#     print(f"Iteration {i+1} completed in {time.time() - start_time:.2f} seconds.")


####### likelihood of offset parameter ######
# orders_of_magnitude = 10
# times = np.zeros(orders_of_magnitude)
# test_lengths = 3**np.arange(1, orders_of_magnitude+1)
# likelihoods = None
# for i, test_length in enumerate(test_lengths):
#     # delete likelihoods to save memory
#     del likelihoods
#     offsets = jnp.array(np.linspace(0.2,0.3,test_length))
#     xs = jnp.array([jnp.concatenate((x[:2], jnp.array([offset]), x[3:])) for offset in offsets])
#     start_time = time.time()
#     likelihoods = vmap_likelihood(xs)
#     end_time = time.time()
#     print(f"Test length: {test_length}, Time taken: {end_time - start_time:.2f} seconds")
#     times[i] = end_time - start_time

# # fit scaling law
# def scaling_law(x, a, b, c):
#     return c + a * x**b
# popt, pcov = sp.optimize.curve_fit(scaling_law, test_lengths, times)

# plt.figure(figsize=(4,4))
# plt.plot(test_lengths, times, marker='o', label='Observed times', color='k')
# plt.xlabel('Test length')
# plt.xscale('log')
# plt.yscale('log')
# plt.plot(test_lengths, scaling_law(test_lengths, *popt), label=f'Fit: a={popt[0]:.2e}, b={popt[1]:.2f}, c={popt[2]:.2e}', color='r')
# plt.legend()
# plt.ylabel('Time # plt.title('Computation time vs. test length')
# plt.tight_layout()
# plt.savefig(f"Figures/{pathogen}260223_offset_likelihood_test_time.png", dpi=300)
# plt.close()

# plt.figure(figsize=(12.5,5.5))
# plt.plot(offsets, likelihoods, color='k')
# plt.xlabel('Offset')
# plt.ylabel('Negative log-likelihood')
# #annotate peak
# peak_idx = jnp.argmin(likelihoods)
# plt.annotate(f'Optimal offset: {offsets[peak_idx]:.3f}', xy=(offsets[peak_idx], likelihoods[peak_idx]), xytext=(offsets[peak_idx]+0.02, likelihoods[peak_idx]+5), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)
# plt.title('Sensitivity of likelihood to offset parameter')
# plt.savefig(f"Figures/{pathogen}260223_offset_likelihood_test.png", dpi=300)
# plt.close()


# ##### updated lockdowns plot #####
# fig, axes = plt.subplots(6, 1, figsize=(6.5, 5), sharex=True)
# pathogens = ["RSV", "Metapneumovirus","InfluenzaA", "InfluenzaB","Adenovirus",  "Parainfluenza3", ]
# seed = 2603172
# option1 = "NA"
# option2 = "flexagep01"
# lockdown = "Exponential"
# prefix = "evosax_DE_"
# colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#000000", "#00BB00", "#648FFF", "#785EF0", "#DC267F"]
# for ax, pathogen, color in zip(axes.flatten(), pathogens, colors):
#     base_path = "Data/Processed/results"+str(seed)[:6]+"/"
#     filename_pattern = pathogen+lockdown+option1+option2+str(seed)+".pickle"
#     filepath = base_path + prefix + filename_pattern
#     with open(filepath, "rb") as f:
#         opt = pickle.load(f)
#     print(f"Loaded: {prefix}{filename_pattern}")
#     x = opt["final_population"][np.argmin(opt["final_fitness"])]
#     params, cntct = x_to_params(x, pathogen, lockdown, option1, option2, print_params=True, return_contact=True)
#     time_range = np.arange(len(cntct))
#     date_range = [t_to_date(t) for t in time_range]
#     ax.plot(date_range, cntct, color=color, linewidth=2)
#     ax.set_title(pathogen)
#     ax.set_ylim(0, 1.1)
#     ax.set_ylabel("")
# axes[-1].set_xlabel("Time")
# # axes[0].set_ylabel("Relative Contact Rate")
# # limit x to 2020-01-01 to 2023-01-01
# axes[-1].set_xlim(pd.to_datetime('2020-01-01'), pd.to_datetime('2023-01-01'))
# # fig.suptitle("Inferred Contact Reductions Over Time", fontsize=14)
# plt.tight_layout()
# plt.savefig("Figures/exponential_lockdown_all_pathogens_stacked2603172.png", dpi=300)

# ##### Lockdowns plot ####
# class FakeOpt:
#     def __init__(self, x):
#         self.x = x
# import pickle
# contact_arrays = {}
# for pathogen, seed in [("RSV", "260217"),("RSV", "2602172"),("RSV", "2602173"), #     lockdown = "FlexStepwise"
#     option1 = "NA"
#     option2 = "flexage"
#     if pathogen != "None":
#         with open("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","rb") as f:
#             opt = pickle.load(f)
#         x = opt.x
#     START = pd.to_datetime('2020-01-01')
#     END = pd.to_datetime('2023-01-01')
#     PERIOD = pd.date_range(start=START, end=END, freq='D')
#     POINTS = jnp.array(date_to_t(PERIOD))
#     n = 4
#     if #         n += 3
#     elif pathogen == 'RSV':
#         n += 2
#     else:
#         n += 4
#     if 'flexage' not in option2:
#         n += 1
#     if pathogen == 'None':
#         n = 0
#         with open("Data/Processed/DE_cm_opt_"+str(seed)+".pickle","rb") as f:
#             opt = pickle.load(f)
#         x = opt.x
#     if 'pathogen' not in option1:
#         if lockdown == 'FlexStepwise':
#             TT = jnp.array([date_to_t(START),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[n]*365,date_to_t('2020-03-19')+(x[n]+x[n+1])*365,date_to_t('2020-03-19')+(x[n]+x[n+1]+x[n+2])*365])
#             # Fs - element 2 must be bigger than element 1, element 3 must be smaller than element 2, element 4 must be bigger than element 2
#             F1 = x[n+3] # value between 0 and 1 #             F2 = F1 + x[n+4] - F1*x[n+4] # value between x[n+3] and 1 #             F3 = F2*x[n+5] # value less than F2 #             F4 = F2 + x[n+6] - F2*x[n+6] # value between F2 and 1 #             FF = jnp.array([1,F1,F2,F3,F4])
#             PIECEWISE_CONTACT = jax.vmap(lambda t: cm.piecewise(t, TT, FF, steepness=0.2))(POINTS)
#             n += 7
#     contact_arrays[(pathogen, seed)] = PIECEWISE_CONTACT
# # plot piecewise contact over time for each pathogen on stacked subplots
# fig, axes = plt.subplots(len(contact_arrays), 1, figsize=(12.5, 8), sharex=True)
# colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#000000", "#00BB00", "#648FFF", "#785EF0", "#DC267F"]

# pathogen_labels = {
#     "RSV": "RSV",
#     "Metapneumovirus": "Metapneumovirus", 
#     "InfluenzaA": "Influenza A",
#     "InfluenzaB": "Influenza B",
#     "Adenovirus": "Adenovirus",
#     "Parainfluenza3": "Parainfluenza 3",
#     "None": "Compromise Fit"
# }

# for i, #     axes[i].plot(PERIOD, contact_data, color=colors[i], linewidth=2)
#     axes[i].set_ylabel("Relative Contact Rate")
#     axes[i].set_title(f"{pathogen_labels[pathogen]} #     axes[i].set_ylim(0, 1)
#     axes[i].grid(True, alpha=0.3)
    
# # Only set x-axis labels on the bottom subplot
# axes[-1].set_xlabel("Time")
# axes[-1].set_xticks(pd.date_range(start='2020-01-01', end='2023-01-01', freq='2YS'))
# axes[-1].set_xticklabels([date.strftime('%Y') for date in pd.date_range(start='2020-01-01', end='2023-01-01', freq='2YS')])

# # Add minor ticks for intermediate years to all subplots
# for ax in axes:
#     ax.set_xticks(pd.date_range(start='2020-06-01', end='2023-06-01', freq='Y'), minor=True)

# plt.tight_layout()
# plt.savefig("Figures/pathogen_specific_contact_rates_stacked_DE260217.png", dpi=300)


# dmflu = pd.read_csv("Data/Processed/DataMartFlu.csv", delimiter=',')
# # make date column into date format
# dmflu['Date'] = pd.to_datetime(dmflu['Date'], format='%d %B %Y')
# # Sum "Influenza A not subtyped,Influenza A H1N1pdm09,Influenza A H3N2" into "Influenza A"
# dmflu['Influenza A Cases'] = dmflu['Influenza A not subtyped'] + dmflu['Influenza A H1N1pdm09'] + dmflu['Influenza A H3N2']
# # rename "Influenza B" to "Influenza B Cases"
# dmflu.rename(columns={'Influenza B':'Influenza B Cases'}, inplace=True)
# # Find relative proportions of Influenza A and Influenza B
# total_flu = dmflu['Influenza A Cases'] + dmflu['Influenza B Cases']
# dmflu['Proportion Influenza A'] = dmflu['Influenza A Cases'] / total_flu
# dmflu['Proportion Influenza B'] = dmflu['Influenza B Cases'] / total_flu
# # use this along with "Percentage testing positive for overall influenza" to get precent positive for Influenza A and B
# dmflu['Influenza A'] = dmflu['Percentage testing positive for overall influenza'] * dmflu['Proportion Influenza A']
# dmflu['Influenza B'] = dmflu['Percentage testing positive for overall influenza'] * dmflu['Proportion Influenza B']

# dmothers = pd.read_csv("Data/Processed/DataMartOthers.csv", delimiter=',')
# # make date column into date format
# dmothers['Date'] = pd.to_datetime(dmothers['Date'], format='%d %B %Y')
# # rename from column format "Percentage testing positive for rhinovirus" to just "Rhinovirus"
# dmothers.rename(columns={'Percentage testing positive for adenovirus':'Adenovirus', 'Percentage testing positive for hMPV':'Metapneumovirus', 'Percentage testing positive for rhinovirus':'Rhinovirus', 'Percentage testing positive for parainfluenza':'Parainfluenza'}, inplace=True)
# # unify the two dataframes into one with columns Date, Pathogen, Percent Positive
# dmflu_melted = dmflu.melt(id_vars=['Date'], value_vars=['Influenza A', 'Influenza B'], var_name='Pathogen', value_name='Percent Positive')
# dmothers_melted = dmothers.melt(id_vars=['Date'], value_vars=['Adenovirus', 'Rhinovirus', 'Parainfluenza', 'Metapneumovirus'], var_name='Pathogen', value_name='Percent Positive')
# dm_all = pd.concat([dmflu_melted, dmothers_melted], ignore_index=True)
# # print earliest and latest date
# print(f"Earliest date: {dm_all['Date'].min()}")
# print(f"Latest date: {dm_all['Date'].max()}")

# # plot pathogen percent positive over time
# fig, ax = plt.subplots(3, 2, figsize=(13.3/2,5.7), sharex=True)
# pathogen_list = ['Influenza A', 'Influenza B', 'Adenovirus', 'Rhinovirus', 'Parainfluenza', 'Metapneumovirus']
# for i, pathogen in enumerate(pathogen_list):
#     row = i // 2
#     col = i % 2
#     data = dm_all[dm_all['Pathogen'] == pathogen]
#     ax[row, col].plot(data['Date'], data['Percent Positive'], color='k')
#     ax[row, col].set_title(pathogen)
#     if row == 2:
#         ax[row, col].set_xlabel('Time')
#     if col == 0:
#         ax[row, col].set_ylabel('Positivity #     ax[row, col].set_xticks(pd.date_range(start='2018-01-01', end='2025-01-01', freq='2YS'))
#     ax[row, col].set_xticklabels([date.strftime('%Y') for date in pd.date_range(start='2018-01-01', end='2025-01-01', freq='2YS')])
#     # Add minor ticks for intermediate years
#     ax[row, col].set_xticks(pd.date_range(start='2018-06-01', end='2025-06-01', freq='Y'), minor=True)
# plt.tight_layout()
# plt.savefig("Figures/DataMart_pathogen_percent_positive.png", dpi=300)

# # mock-up of finding the optimal timing of a vaccination to minimize peak incidence
# # heatmap with x axis being vaccination time, y axis being vaccine coverage, color being peak incidence change from baseline
# # region of low incidence for the middle of vaccination times and high coverage, region of high incidence for early vaccinaiton and low coverage
# n_days = 100
# vaccination_times = jnp.arange(n_days)
# vaccine_coverages = jnp.linspace(0,1,50)
# peak_incidence_change = jnp.zeros((len(vaccine_coverages), len(vaccination_times)))
# for i, coverage in enumerate(vaccine_coverages):
#     for j, vacc_time in enumerate(vaccination_times):
#         # mock peak incidence change as a function of vacc_time and coverage
#         # Add a penalty for early vaccination with low coverage
#         early_penalty = #         peak_incidence_change = peak_incidence_change.at[i,j].set(1 - 0.3 * coverage * jnp.exp(-0.01*(vacc_time - n_days/2)**2) + early_penalty)
# plt.figure(figsize=(13.3/2.2,2.7))
# plt.imshow(peak_incidence_change, extent=[0,n_days,0,1], aspect='auto', origin='lower', cmap='viridis')
# plt.colorbar(label='Peak incidence change')
# plt.xlabel('Vaccination time')
# plt.ylabel('Vaccine coverage')
# plt.xticks([])
# plt.yticks([])
# # plt.title('Finding optimal vaccination timing')
# plt.savefig("Figures/optimal_vaccination_timing_mockup.png", dpi=300)

# # mock-up of forecasting multiple pathogens simultaneously
# # generate two classic SIR curves with differen timing
# n_days = 60
# t = jnp.arange(n_days)
# def sir_model(beta, gamma, S0, I0, R0, days):
#     S = jnp.zeros(days)
#     I = jnp.zeros(days)
#     R = jnp.zeros(days)
#     S = S.at[0].set(S0)
#     I = I.at[0].set(I0)
#     R = R.at[0].set(R0)
#     for day in range(1, days):
#         new_infections = beta * S[day-1] * I[day-1] / #         new_recoveries = gamma * I[day-1]
#         S = S.at[day].set(S[day-1] - new_infections)
#         I = I.at[day].set(I[day-1] + new_infections - new_recoveries)
#         R = R.at[day].set(R[day-1] + new_recoveries)
#     return S, I, R
# beta1, gamma1 = 0.3, 0.1
# beta2, gamma2 = 0.4, 0.1
# S1, I1, R1 = sir_model(beta1, gamma1, 990, 10, 0, n_days)
# S2, I2, R2 = sir_model(beta2, gamma2, 990, 10, 0, n_days)
# # plot both on same graph
# plt.figure(figsize=(13.3/2.2,2.7))
# # Create mock uncertainty intervals that grow then shrink with outbreak size
# uncertainty_factor = 0.5  # Controls width of uncertainty bands
# # Make uncertainty narrow at time 20 # transition_factor = 1 - jnp.exp(-0.5 * # uncertainty1 = uncertainty_factor * I1 * # uncertainty2 = uncertainty_factor * I2 * 
# # Plot the main curves
# plt.plot(t[:21], I1[:21], label='Virus A', color='#648FFF', linestyle='-')
# plt.plot(t[20:], I1[20:], color='#648FFF', linestyle='--')
# plt.plot(t[:21], I2[:21], label='Virus B', color='#DC267F', linestyle='-')
# plt.plot(t[20:], I2[20:], color='#DC267F', linestyle='--')

# # Add uncertainty bands
# plt.fill_between(t[20:], I1[20:] - uncertainty1[20:], I1[20:] + uncertainty1[20:], alpha=0.2, color='#648FFF')
# plt.fill_between(t[20:], I2[20:] - uncertainty2[20:], I2[20:] + uncertainty2[20:], alpha=0.2, color='#DC267F')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Incidence')
# plt.xticks([])
# plt.yticks([])
# # plt.tight_layout()
# # plt.title('Forecasting simultaneous epidemics')
# plt.savefig("Figures/multiple_pathogen_forecast_mockup.png", dpi=300)

# # use the same SIR model as above, but do three pathogens and introduce a small perturbation in beta at day 30
# n_days = 40
# slowdown = 5
# t = jnp.linspace(0, n_days-1, n_days*slowdown)
# def sir_model_perturb(beta, gamma, S0, I0, R0, t, perturb_day, perturb_amount):
#     S = jnp.zeros(t.shape[0])
#     I = jnp.zeros(t.shape[0])
#     R = jnp.zeros(t.shape[0])
#     S = S.at[0].set(S0)
#     I = I.at[0].set(I0)
#     R = R.at[0].set(R0)
#     for i in range(1, t.shape[0]):
#         new_infections = beta * S[i-1] * I[i-1] / #         new_recoveries = gamma * I[i-1] / slowdown
#         S = S.at[i].set(S[i-1] - new_infections)
#         I = I.at[i].set(I[i-1] + new_infections - new_recoveries)
#         R = R.at[i].set(R[i-1] + new_recoveries)
#     return S, I, R
# beta_values = [0.5, 1, 0.4]
# gamma = 0.1
# perturb_day = 17
# perturb_amount = 0
# S_list, I_list, R_list = [], [], []
# for beta in beta_values:
#     S, I, R = sir_model_perturb(beta, gamma, 990, 10, 0, t, perturb_day, perturb_amount)
#     S_list.append(S)
#     I_list.append(I)
#     R_list.append(R)
# # create poisson noisy trajectories from I_list
# key = jax.random.PRNGKey(0)
# for i in range(3):
#     I_noisy = jax.random.poisson(key, I_list[i])
#     I_noisy = I_noisy*(1 - 0.25 * jnp.exp(-0.8 * #     I_list[i] = I_noisy
# # plot three on subgraphs with vertical dashed line at perturbation day
# plt.figure(figsize=(13.3/2.2,2.7))
# for i in range(3):
#     plt.subplot(3,1,i+1)
#     plt.plot(t, I_list[i], label=f'Virus {chr(65+i)}', color="k")
#     plt.axvline(x=perturb_day, color='r', linestyle='--', alpha=0.7)
#     # Add annotation with two asterisks
#     if i==0:
#         plt.annotate('*', xy=(perturb_day, plt.ylim()[1]*0.9), ha='center', fontsize=16, color='r')
#     if i == 1:
#         plt.ylabel('Incidence')
#     else:
#         plt.ylabel('')
#     plt.xticks([])
#     plt.yticks([])
#     if i == 2:
#         plt.xlabel('Time')
# plt.tight_layout()
# plt.savefig("Figures/epidemic_response_perturbation_mockup.png", dpi=300)



# # Load DataMart data from Data/Raw/DataMartAllFlu2425.csv Data/Raw/DataMartInfluenzaB2425.csv Data/Raw/DataMartAdenovirus2425.csv Data/Raw/DataMartMetapneumovirus2425.csv Data/Raw/DataMartParainfluenza2425.csv Data/Raw/DataMartRhinovirus2425.csv Data/Raw/DataMartRSV2425NorthEngland.csv Data/Raw/DataMartCOVID192425NorthEngland.csv
# allflu = pd.read_csv("Data/Raw/DataMartAllFlu2425.csv", delimiter=',', header=None)
# influenzaB = pd.read_csv("Data/Raw/DataMartInfluenzaB2425.csv", delimiter=',', header=None)
# adenovirus = pd.read_csv("Data/Raw/DataMartAdenovirus2425.csv", delimiter=',', header=None)
# metapneumovirus = pd.read_csv("Data/Raw/DataMartMetapneumovirus2425.csv", delimiter=',', header=None)
# parainfluenza = pd.read_csv("Data/Raw/DataMartParainfluenza2425.csv", delimiter=',', header=None)
# rhinovirus = pd.read_csv("Data/Raw/DataMartRhinovirus2425.csv", delimiter=',', header=None)
# # assign column names: Week, Cases
# allflu.columns = ['Week', 'Cases']
# influenzaB.columns = ['Week', 'Cases']
# adenovirus.columns = ['Week', 'Cases']
# metapneumovirus.columns = ['Week', 'Cases']
# parainfluenza.columns = ['Week', 'Cases']
# rhinovirus.columns = ['Week', 'Cases']
# rsv = pd.read_csv("Data/Raw/DataMartRSV2425NorthEngland.csv", delimiter=',', header=None)
# covid19 = pd.read_csv("Data/Raw/DataMartCOVID192425NorthEngland.csv", delimiter=',', header=None)
# # column names: Month, Positivity
# rsv.columns = ['Month', 'Positivity']
# covid19.columns = ['Month', 'Positivity']
# # x axis for everything except covid and rsv is number of weeks since 2024-01-01, convert to dates
# allflu['Date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(allflu['Week']*7, unit='D')
# influenzaB['Date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(influenzaB['Week']*7, unit='D')
# adenovirus['Date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(adenovirus['Week']*7, unit='D')
# metapneumovirus['Date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(metapneumovirus['Week']*7, unit='D')
# parainfluenza['Date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(parainfluenza['Week']*7, unit='D')
# rhinovirus['Date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(rhinovirus['Week']*7, unit='D')
# # x axis for rsv and covid is number of months since 2024-06-01, convert to dates
# rsv['Date'] = pd.to_datetime('2024-06-01') + pd.to_timedelta(rsv['Month']*30, unit='D')
# covid19['Date'] = pd.to_datetime('2024-06-01') + pd.to_timedelta(covid19['Month']*30, unit='D')
# # sort rsv by date
# rsv = rsv.sort_values(by='Date')
# # flu A is allflu - influenzaB
# influenzaA = allflu.copy()
# influenzaA['Cases'] = allflu['Cases'] - influenzaB['Cases']
# # plot all pathogens on grid of subplots
# fig, ax = plt.subplots(4,2,figsize=(13.3/2,6), sharex=True)
# ax[0,0].plot(influenzaA['Date'], influenzaA['Cases'], label='Influenza A', color='k')
# ax[0,0].set_title('Influenza A')
# ax[0,1].plot(influenzaB['Date'], influenzaB['Cases'], label='Influenza B', color='k')
# ax[0,1].set_title('Influenza B')
# ax[1,0].plot(adenovirus['Date'], adenovirus['Cases'], label='Adenovirus', color='k')
# ax[1,0].set_title('Adenovirus')
# ax[1,1].plot(metapneumovirus['Date'], metapneumovirus['Cases'], label='Metapneumovirus', color='k')
# ax[1,1].set_title('Metapneumovirus')
# ax[2,0].plot(parainfluenza['Date'], parainfluenza['Cases'], label='Parainfluenza', color='k')
# ax[2,0].set_title('Parainfluenza')
# ax[2,1].plot(rhinovirus['Date'], rhinovirus['Cases'], label='Rhinovirus', color='k')
# ax[2,1].set_title('Rhinovirus')
# ax[3,0].plot(rsv['Date'], rsv['Positivity'], label='RSV Positivity', color='k')
# ax[3,0].set_title('RSV')
# ax[3,1].plot(covid19['Date'], covid19['Positivity'], label='COVID-19 Positivity', color='k')
# ax[3,1].set_title('SARS-CoV-2')
# for i in range(3):
#     ax[i,0].set_ylabel("Incidence")
#     ax[i,1].set_ylabel("")
# ax[3,0].set_ylabel("Positivity # ax[3,1].set_ylabel("")
# for j in range(2):
#     ax[3,j].set_xlabel("Time")
# for ax in ax.flatten():
#     # set x ticks to every 2 years
#     ax.set_xticks(pd.date_range(start='2024-06-01',end='2025-07-01',freq='2MS'))
#     ax.set_xticklabels([year.strftime('%b') for year in pd.date_range(start='2024-06-01',end='2025-07-01',freq='2MS')], rotation=30)
#     # add minor ticks for non-labelled years
#     ax.set_xticks(pd.date_range(start='2024-06-01',end='2025-07-01',freq='MS'), minor=True)
#     # ax.axvspan(npi_start, npi_end, color='gray', alpha=0.3)
# # save
# plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
# plt.tight_layout()
# plt.savefig("Figures/DataMart_pathogen_incidence.png", dpi=300)

# # axes with paramter 1 from 0 to 1 and parameter 2 from 0 to 1, and a faint grid in the background
# fig, ax = plt.subplots(figsize=(12.5,5.5))

# ax.scatter([0.3,0.59,0.62], [0.11, 0.2, 0.8], s=150, color="white", edgecolor="black")

# # plot a grey triangle between these points
# triangle = plt.Polygon([[0.3,0.11],[0.59,0.2],[0.62,0.8]], color='grey', alpha=0.3)
# ax.add_patch(triangle)

# ax.set_xlabel("Parameter 1")
# ax.set_ylabel("Parameter 2")
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# ax.set_yticks(jnp.arange(0,1.1,0.2))
# ax.set_xticks(jnp.arange(0,1.1,0.2))
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("Figures/parameter_grid_template_points_triangle.png", dpi=300)

# RSV = ["RSV", "251103", "NA"]
# Metapneumovirus = ["Metapneumovirus", "2511032", "NA"]
# InfluenzaA = ["InfluenzaA", "251103", "NA"]
# InfluenzaB = ["InfluenzaB", "2511042", "NA"]
# Adenovirus = ["Adenovirus", "2511032", "NA"]
# Parainfluenza3 = ["Parainfluenza3", "2511032", "NA"]

# short_names = {"RSV":"RSV", "Metapneumovirus":"hMPV", "InfluenzaA":"FluA", "InfluenzaB":"FluB", "Adenovirus":"AdV", "Parainfluenza3":"PIV3"}
# full_names = {"RSV": "RSV", "Metapneumovirus": "Metapneumovirus", "InfluenzaA": "Influenza A", "InfluenzaB": "Influenza B", "Adenovirus": "Adenovirus", "Parainfluenza3": "Parainfluenza 3"}

# good_simulations = [InfluenzaA, RSV, Adenovirus, InfluenzaB, Metapneumovirus, Parainfluenza3]

# POINTS = np.array(date_to_t(PERIOD))
# NAG = len(AGE_GROUP_NAMES)

# fig, ax = plt.subplots(2,3,figsize=(12.5,5.5), sharex=True)
# for i, #     incidence = jnp.asarray(pd.read_csv("Data/Processed/KPSC_ARI_"+pathogen+"_cases_age_daily.csv",index_col=0))
#     obs_summed_age = incidence.sum(axis=1)
#     # sum obs over each season, starting with the first time point
#     n_seasons = int((POINTS[-1] - POINTS[0]) / 365)
#     # Curtail obs to fit exact seasons #     days_to_keep = n_seasons * 365
#     obs_curtailed = incidence[:days_to_keep, :]
#     obs_summed_age_curtailed = obs_summed_age[:days_to_keep]
#     obs_per_season = obs_curtailed.reshape((n_seasons, 365, NAG)).sum(axis=1)
#     obs_summed_age_per_season = obs_summed_age_curtailed.reshape((n_seasons, 365)).sum(axis=1)
#     # concatenate to obs_per_season
#     obs_per_season = jnp.concatenate([obs_per_season, obs_summed_age_per_season[:, None]], axis=1)
#     # plot a line showing the incidence in each age group in the median-sized pre-pandemic season
#     median_pre_season_index = jnp.argsort(obs_per_season[:5, -1])[2]
#     pre_age_incidence = obs_per_season[median_pre_season_index, :-1] / CENSUS_AGE_POP
#     pre_age_incidence /= jnp.sum(pre_age_incidence)
#     rebound_index = 5 + jnp.argmax(obs_per_season[5:, -1])
#     rebound_age_incidence = obs_per_season[rebound_index, :-1] / CENSUS_AGE_POP
#     rebound_age_incidence /= jnp.sum(rebound_age_incidence)
#     ax[i//3, i%3].plot(jnp.arange(NAG), pre_age_incidence, label='Pre-pandemic', color='#DC267F')
#     ax[i//3, i%3].plot(jnp.arange(NAG), rebound_age_incidence, label='Re-emergence', color='#648FFF')
#     ax[i//3, i%3].set_title(full_names[pathogen])
#     if i//3==1:
#         ax[i//3, i%3].set_xlabel("Age Group")
#         ax[i//3, i%3].set_xticks(jnp.arange(NAG))
#         ax[i//3, i%3].set_xticklabels(AGE_GROUP_NAMES)
#         # tilt x tick labels
#         plt.setp(ax[i//3, i%3].get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
# ax[0,0].legend(frameon=False)
# plt.tight_layout()
# plt.savefig("Figures/age_distribution_shift.png", dpi=300)

# pathogen_name = "COVID-19"
# fig, ax = plt.subplots(figsize=(12.3/4,6.5/2))
# aggregation = "Month"
# kpsc_positive_test_plot(ax,pathogen=pathogen_name,AGE_GROUPS=None,AGE_GROUP_NAMES=AGE_GROUP_NAMES, incidence=True, legend=False, aggregation=aggregation, load_data=False, color="#DC267F")
# ax.legend(frameon=False)
# ax.set_xlabel("", fontfamily='Helvetica', fontsize=18)
# ax.set_ylabel("", fontfamily='Helvetica', fontsize=18)
# ax.set_title("COVID-19")
# ax.set_xticklabels(["","2016","","2018","","2020","","2022","","2024"], fontfamily='Helvetica', fontsize=10)
# # ax.set_xticklabels([])
# # set y tick labels to helvetica
# ax.set_yticklabels(ax.get_yticklabels(), fontfamily='Helvetica', fontsize=12)
# # ax.set_xticklabels([])

# plt.tight_layout()
# plt.savefig("Figures/DE_"+pathogen_name+"_noage_minimal.png",dpi=300)

# with open("Data/Processed/DE_cm_opt_maxmimmwane,maxmimmwane,maxmimmwane,maxmimmwane,maxmimmwaneflexage2511032,2511032,2511032,2511032,2511032.pickle","rb") as f:
#     opt = pickle.load(f)
# x = opt.x

# x = np.array([0.9381735039583684,0.807470490095935,0.18641838533050858,0.3003678719530728,0.9374092638648985,0.9725263495329137,0.9763928665615773])
# # make object with x as attribute
# class FakeOpt:
#     def __init__(self, x):
#         self.x = x
# fake_opt_result = FakeOpt(x)

# with open("Data/Processed/DE_cm_opt_251110intermediate.pickle","wb") as f:
#     pickle.dump(fake_opt_result, f)

# START = pd.to_datetime('2015-10-01')
# END = pd.to_datetime('2025-05-01')
# PERIOD = pd.date_range(start=START, end=END, freq='D')
# POINTS = jnp.array(date_to_t(PERIOD))
# n = 0
# TT = jnp.array([date_to_t(START),date_to_t('2020-03-19'),date_to_t('2020-03-19')+x[n]*365,date_to_t('2020-03-19')+(x[n]+x[n+1])*365,date_to_t('2020-03-19')+(x[n]+x[n+1]+x[n+2])*365])
# # Fs - element 2 must be bigger than element 1, element 3 must be smaller than element 2, element 4 must be bigger than element 2
# F1 = x[n+3] # value between 0 and 1 # F2 = F1 + x[n+4] - F1*x[n+4] # value between x[n+3] and 1 # F3 = F2*x[n+5] # value less than F2 # F4 = F2 + x[n+6] - F2*x[n+6] # value between F2 and 1 # FF = jnp.array([1,F1,F2,F3,F4])
# PIECEWISE_CONTACT = jax.vmap(lambda t: cm.piecewise(t, TT, FF, steepness=0.2))(POINTS)

# # plot relative contact rate over time
# plt.figure(figsize=(10, 6))
# plt.plot(PERIOD, PIECEWISE_CONTACT, label='Relative Contact Rate', color='blue')
# plt.xlabel('Date')
# plt.ylabel('Relative Contact Rate')
# plt.title('Relative Contact Rate Over Time')
# plt.legend()
# plt.grid()
# plt.show()

# with open("Data/Processed/DE_cm_opt_250715.pickle","rb") as f:
#     opt = pickle.load(f)
# print(opt.x)
# N_S, NAG = 3, 7
# from Parameters.census_population import CENSUS_AGE_POP
# from Parameters.census_population import AGE_GROUP_NAMES
# # time how long it takes to run 10 flu sims
# ## Initial conditions
# STATE0 = jnp.zeros((2*N_S+1,NAG))
# STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
# STATE0 = STATE0.at[1,:].set(1)
# # # flatten initial state and add maternal immunity compartment
# STATE0 = STATE0.flatten()
# STATE0 = jnp.concatenate((jnp.array([0]), STATE0))
# from Parameters.times_and_contacts import PERIOD
# POINTS = np.array(date_to_t(PERIOD))
# params = x_to_params(jnp.array([0.08761366994242037,0.5630226801200686,0.07059271526314115,0.004182654461622023,0.49925504835496337,0.32124852929495257,0.6595126147878759,0.7452366339350426,0.7043837937366796,0.4236511501951234,0.7340296802182136,0.4564514394851521,0.413152171847781,0.9833702974262226,0.0021935084182965066,0.0007962706480398884,0.0015927128649279709,0.0010406238553368342,0.000632834711347473,0.0007653558748031127,0.0030613178577122554]), "InfluenzaA", "FlexStepwise", "NA", "flexage")
# p_time_to_obs = jnp.asarray(pd.read_csv("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
# sim = run_simulation(params, STATE0, POINTS[-1], POINTS)
# # # plot
# # fig, ax = plt.subplots(1,1,figsize=(10,6))
# # mx = lockdown_incidence_plot(ax,STATE0,params,PERIOD,POINTS,date_to_t('2020-03-19'),365,solution=sim,label="Simulation",by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=30.44*10000,p_time_to_obs=p_time_to_obs)
# # lockdown_incidence_format(ax,date_to_t('2020-03-19'),365,mx,year_window=2)
# # plt.show()
# start_time = time.time()
# for i in range(10):
#     print(i)
#     params = x_to_params(jnp.array([0.08761366994242037,0.5630226801200686,0.07059271526314115,0.004182654461622023,0.49925504835496337,0.32124852929495257,0.6595126147878759,0.7452366339350426,0.7043837937366796,0.4236511501951234,0.7340296802182136,0.4564514394851521,0.413152171847781,0.9833702974262226,0.0021935084182965066,0.0007962706480398884,0.0015927128649279709,0.0010406238553368342,0.000632834711347473,0.0007653558748031127,0.0030613178577122554]), "InfluenzaA", "FlexStepwise", "NA", "flexage")
#     run_simulation(params, STATE0, POINTS[-1], POINTS)
# end_time = time.time()
# print(f"Time taken to run 10 flu sims: {end_time - start_time} seconds")

# p_time_to_obs_RSV = jnp.asarray(pd.read_csv("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
# p_time_to_obs_InfluenzaA = jnp.asarray(pd.read_csv("Data/Processed/Influenza_A_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
# p_time_to_obs_InfluenzaB = jnp.asarray(pd.read_csv("Data/Processed/Influenza_B_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
# print(p_time_to_obs_RSV)
# fig, ax = plt.subplots(1,1,figsize=(8,5))
# ax.plot(p_time_to_obs_RSV, label='RSV', color='blue')
# ax.plot(p_time_to_obs_InfluenzaA, label='Influenza A', color='red')
# ax.plot(p_time_to_obs_InfluenzaB, label='Influenza B', color='green')
# ax.set_xlabel('Days since infection')
# ax.set_ylabel('Probability of hospital admission')
# ax.set_title('Probability Distribution of Time to Hospital Admission')
# ax.legend()
# plt.tight_layout()
# plt.savefig('Figures/time_to_hospital_admission_distribution.png', dpi=300)

# population_size_by_age_and_year = pd.read_csv("Data/Processed/KPSC_population_by_age.csv")
# # split "Older children" group into 3/13 # oldchildren = population_size_by_age_and_year["Older children"]
# population_size_by_age_and_year["Older children"] = # population_size_by_age_and_year["Young adults"] += # # save
# population_size_by_age_and_year.to_csv("Data/Processed/KPSC_population_by_age_split.csv", index=False)

# # load contact matrix
# CONTACT_MATRIX = jnp.asarray(pd.read_csv('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', header=None).values)
# print(np.sum(np.sum(CONTACT_MATRIX,axis=1) * CENSUS_AGE_POP) / np.sum(CENSUS_AGE_POP))
# true_x = jnp.array([0.001,0.1,0.1,0.5
# ,0.1,0.5,0.6,0.5
# ,0.9,0.5,0.4,0.8,0.7,0.8,0.1
# ,2e-03,1e-03,3e-03,2e-04,1e-04,3e-04,4e-03])

# xs_DE = jnp.array([
# [1.76057553e-03,1.17780975e-01,4.07605321e-03,3.26722370e-01
# ,1.06813239e-01,5.57069377e-01,6.91879892e-01,5.69055777e-01
# ,5.26365486e-01,6.10660436e-01,4.14257131e-01,7.12898822e-01
# ,8.55832577e-01,5.19839512e-01,5.56835345e-01,2.84376252e-03
# ,1.82065515e-03,3.93473780e-03,1.85154383e-04,1.27618716e-04
# ,3.51718381e-04,2.47349633e-03],
# [6.83127799e-03,1.05864673e-01,9.97903176e-01,3.08251207e-01
# ,1.61640121e-01,4.98191875e-01,4.12853539e-01,5.08699173e-01
# ,4.99094547e-01,9.95583026e-01,7.14144574e-02,5.21733782e-01
# ,6.34643767e-01,4.00957973e-01,8.61960620e-01,2.90649503e-03
# ,1.54652770e-03,4.30782555e-03,2.21694773e-04,1.47007922e-04
# ,3.09139342e-04,3.00758805e-03],
# [4.55167163e-03,1.18600089e-01,1.22988249e-02,3.22665793e-01
# ,1.39954406e-01,2.61713622e-01,3.69055668e-01,6.42740530e-01
# ,5.56942961e-01,5.76848439e-01,6.18110596e-01,6.56133448e-01
# ,9.13199875e-01,9.60489560e-01,4.96523258e-01,1.48297529e-03
# ,1.44148353e-03,3.87272222e-03,1.83714395e-04,1.26432119e-04
# ,3.42741749e-04,3.58607730e-03],
# [4.42537348e-03,1.27010165e-01,1.33153722e-02,3.43548237e-01
# ,1.27856342e-01,4.26468374e-01,2.14870471e-01,5.86467576e-01
# ,8.40991142e-01,8.12799479e-01,9.98761920e-02,8.43456209e-01
# ,6.28944815e-01,4.03907605e-01,6.48113414e-01,2.56957711e-03
# ,1.94272692e-03,3.72828664e-03,1.63209256e-04,1.58180902e-04
# ,2.93373514e-04,3.89342810e-03]])

# # mean root sum of squares error
# rsses = [jnp.sqrt(jnp.sum((x-true_x)**2)) for x in xs_DE]
# print(rsses, np.mean(rsses), np.std(rsses))

# # plot boxplots of each parameter
# fig, ax = plt.subplots(1,1,figsize=(10,6))
# ax.boxplot(xs_DE,positions=jnp.arange(len(true_x)),showfliers=False)
# ax.scatter(jnp.arange(len(true_x)),true_x,color='red',label='True Value')
# ax.set_yscale('log')
# ax.set_xticks(jnp.arange(len(true_x)))
# ax.set_xticklabels([f'x{i}' for i in range(1,len(true_x)+1)])
# ax.set_ylabel('Parameter Value # ax.set_title('Parameter Estimates from Differential Evolution')
# plt.legend()
# plt.tight_layout()
# plt.savefig('Figures/DE_tests_boxplot.png',dpi=300)

# fig, ax = plt.subplots(3,2)
# vax_opt = np.loadtxt('Data/Processed/KPSC_vaccination_rate_ages_monthly_optimized_unshift.csv',delimiter=',')
# vax_original = np.loadtxt('Data/Processed/vax_rate_example.csv',delimiter=',')
# eff_cov_opt = np.loadtxt('Data/Processed/eff_cov_example_optimized.csv',delimiter=',')
# eff_cov_original = np.loadtxt('Data/Processed/eff_cov_example.csv',delimiter=',')
# for i in range(7):
#     ax[0,0].plot(vax_opt[:,i],label='Optimized',color=hsv_colors[i])
#     ax[0,0].set_title('Vaccination Rate from jax optimized code')
#     ax[1,0].plot(vax_original[:,i],label='Original',color=hsv_colors[i])
#     ax[1,0].set_title('Vaccination Rate from original code')
#     ax[2,0].plot(vax_opt[:,i]/vax_original[:,i],label='Difference',color=hsv_colors[i])
#     ax[2,0].set_title('Ratio of optimized to original')
#     # ax[2,0].set_ylim(0,2)
#     ax[0,1].plot(eff_cov_opt[:,i],label='Optimized',color=hsv_colors[i])
#     ax[0,1].set_title('Effective vaccine Coverage from jax optimized code')
#     ax[1,1].plot(eff_cov_original[:,i],label='Original',color=hsv_colors[i])
#     ax[1,1].set_title('Effective vaccine Coverage from original code')
#     ax[2,1].plot(eff_cov_opt[:,i]/eff_cov_original[:,i],label='Difference',color=hsv_colors[i])
#     ax[2,1].set_title('Ratio of optimized to original')
# plt.tight_layout()
# plt.savefig('Figures/vax_rate_comparison.png',dpi=300)

# covaraiance = np.array([[4.79203036e-07,6.11817303e-08,1.19563281e-06,-5.83700484e-07
# ,-1.70161560e-06,2.85094663e-07,5.17774219e-06,1.43810590e-06
# ,-5.23882675e-06,5.42544315e-06,-7.17794489e-06,8.13229033e-07
# ,2.33812643e-05,2.22158505e-09,1.21885743e-08,-5.96540066e-10
# ,-6.53007740e-09,-5.05128426e-09,9.10091498e-09,2.03414539e-08]
# ,[6.11817303e-08,1.11076320e-04,-1.89797453e-06,4.48372238e-05
# ,-2.62435341e-05,-3.13731653e-05,-1.21375367e-04,1.26777383e-04
# ,-1.27089538e-04,-9.74291882e-06,2.67497791e-04,-4.44528227e-05
# ,3.82569609e-04,-1.01146616e-06,-3.97268227e-07,7.48251486e-07
# ,2.81277659e-08,8.28694542e-08,-8.54035974e-08,4.34110138e-07]
# ,[1.19563281e-06,-1.89797453e-06,2.61372765e-04,7.72046933e-05
# ,-9.41528140e-06,-7.26910001e-05,2.76141533e-04,-1.82176664e-04
# ,2.59256604e-04,1.08978699e-04,-3.72536671e-04,3.86852729e-05
# ,3.03370942e-04,6.75235933e-07,9.48237434e-07,-1.78050938e-06
# ,-8.85045109e-08,1.75618138e-07,-2.79946383e-07,-9.79877483e-07]
# ,[-5.83700484e-07,4.48372238e-05,7.72046933e-05,1.27791243e-03
# ,-4.60736728e-04,1.35050189e-04,-6.65350898e-05,-2.35385464e-04
# ,-3.52636568e-04,-3.62318964e-04,3.97549385e-04,-3.34146509e-05
# ,3.15960993e-04,-1.46760460e-06,-1.25120792e-06,4.35325460e-07
# ,-2.16834998e-07,-1.83949802e-08,-9.79956318e-07,-4.43737844e-07]
# ,[-1.70161560e-06,-2.62435341e-05,-9.41528140e-06,-4.60736728e-04
# ,2.55401146e-04,3.84334554e-05,4.64269313e-05,-8.36311942e-05
# ,3.23148854e-04,9.16067224e-05,-2.79842264e-04,3.54780449e-05
# ,-5.56627310e-04,2.72653103e-07,4.52854379e-07,-5.08132367e-07
# ,3.63904904e-08,8.31447557e-08,1.88576942e-07,-7.51321986e-07]
# ,[2.85094663e-07,-3.13731653e-05,-7.26910001e-05,1.35050189e-04
# ,3.84334554e-05,1.05532382e-02,-2.07219617e-03,1.68053418e-04
# ,-7.51432466e-04,-6.91644209e-05,4.05881323e-04,-1.53171712e-04
# ,1.91332537e-04,-7.87515263e-07,5.79685719e-07,3.34697373e-06
# ,1.12322794e-07,3.06947330e-06,-3.53635662e-07,1.29340923e-06]
# ,[5.17774219e-06,-1.21375367e-04,2.76141533e-04,-6.65350898e-05
# ,4.64269313e-05,-2.07219617e-03,1.65628958e-02,-2.68636949e-03
# ,-4.54056495e-04,7.54260771e-04,-2.34974493e-03,1.04084566e-04
# ,9.04635246e-04,4.89031982e-06,4.44266373e-06,-5.91248123e-06
# ,-4.10187508e-07,-3.18319874e-06,1.13928309e-06,-2.42750105e-06]
# ,[1.43810590e-06,1.26777383e-04,-1.82176664e-04,-2.35385464e-04
# ,-8.36311942e-05,1.68053418e-04,-2.68636949e-03,1.01034442e-02
# ,-4.58366168e-03,-1.39334834e-03,1.12638705e-03,-6.62414560e-04
# ,4.78708503e-03,4.78092016e-06,-2.49460802e-06,9.54097321e-06
# ,8.90043772e-07,-3.56457809e-07,1.59644845e-06,1.01953437e-05]
# ,[-5.23882675e-06,-1.27089538e-04,2.59256604e-04,-3.52636568e-04
# ,3.23148854e-04,-7.51432466e-04,-4.54056495e-04,-4.58366168e-03
# ,1.16978389e-02,2.00808286e-03,3.95447546e-04,1.13551364e-03
# ,-3.23958785e-03,-1.44402573e-06,3.39598112e-06,-1.02000171e-05
# ,2.44874017e-07,2.31148189e-06,-1.16687172e-06,-9.35120022e-06]
# ,[5.42544315e-06,-9.74291882e-06,1.08978699e-04,-3.62318964e-04
# ,9.16067224e-05,-6.91644209e-05,7.54260771e-04,-1.39334834e-03
# ,2.00808286e-03,6.12945387e-03,-1.80092742e-03,-3.02930764e-04
# ,-1.01501019e-03,1.25887518e-06,1.60798234e-06,-4.43510207e-06
# ,-2.28108227e-07,4.41064990e-07,2.78509023e-07,-1.11413557e-06]
# ,[-7.17794489e-06,2.67497791e-04,-3.72536671e-04,3.97549385e-04
# ,-2.79842264e-04,4.05881323e-04,-2.34974493e-03,1.12638705e-03
# ,3.95447546e-04,-1.80092742e-03,1.95170913e-02,-1.69610542e-03
# ,-2.44992321e-03,-1.89320294e-07,-5.48259234e-06,1.58914649e-05
# ,1.01425040e-06,1.78055869e-06,-1.64816992e-06,8.49154423e-06]
# ,[8.13229033e-07,-4.44528227e-05,3.86852729e-05,-3.34146509e-05
# ,3.54780449e-05,-1.53171712e-04,1.04084566e-04,-6.62414560e-04
# ,1.13551364e-03,-3.02930764e-04,-1.69610542e-03,1.31691601e-03
# ,5.83037727e-04,-9.15850577e-07,1.53238590e-06,-2.86800431e-06
# ,-2.94490870e-08,3.54485886e-08,4.63047715e-08,-1.96429319e-06]
# ,[2.33812643e-05,3.82569609e-04,3.03370942e-04,3.15960993e-04
# ,-5.56627310e-04,1.91332537e-04,9.04635246e-04,4.78708503e-03
# ,-3.23958785e-03,-1.01501019e-03,-2.44992321e-03,5.83037727e-04
# ,4.30296488e-02,2.25917998e-06,6.65397532e-06,-2.87472227e-06
# ,1.15418008e-06,-1.18851761e-06,1.16624731e-06,1.13066397e-05]
# ,[2.22158505e-09,-1.01146616e-06,6.75235933e-07,-1.46760460e-06
# ,2.72653103e-07,-7.87515263e-07,4.89031982e-06,4.78092016e-06
# ,-1.44402573e-06,1.25887518e-06,-1.89320294e-07,-9.15850577e-07
# ,2.25917998e-06,3.27823132e-07,1.64381526e-08,-1.20549302e-08
# ,4.14773386e-09,-2.37010468e-09,3.62573523e-09,1.76998343e-08]
# ,[1.21885743e-08,-3.97268227e-07,9.48237434e-07,-1.25120792e-06
# ,4.52854379e-07,5.79685719e-07,4.44266373e-06,-2.49460802e-06
# ,3.39598112e-06,1.60798234e-06,-5.48259234e-06,1.53238590e-06
# ,6.65397532e-06,1.64381526e-08,1.43211552e-07,-1.87489601e-08
# ,-9.78943333e-10,6.52303133e-09,1.65366442e-09,-1.40371752e-08]
# ,[-5.96540066e-10,7.48251486e-07,-1.78050938e-06,4.35325460e-07
# ,-5.08132367e-07,3.34697373e-06,-5.91248123e-06,9.54097321e-06
# ,-1.02000171e-05,-4.43510207e-06,1.58914649e-05,-2.86800431e-06
# ,-2.87472227e-06,-1.20549302e-08,-1.87489601e-08,2.26993960e-07
# ,1.93190117e-09,1.77235765e-09,-4.75925166e-10,3.50311165e-08]
# ,[-6.53007740e-09,2.81277659e-08,-8.85045109e-08,-2.16834998e-07
# ,3.63904904e-08,1.12322794e-07,-4.10187508e-07,8.90043772e-07
# ,2.44874017e-07,-2.28108227e-07,1.01425040e-06,-2.94490870e-08
# ,1.15418008e-06,4.14773386e-09,-9.78943333e-10,1.93190117e-09
# ,5.59336755e-09,1.31348456e-10,-1.39966834e-10,1.85360584e-09]
# ,[-5.05128426e-09,8.28694542e-08,1.75618138e-07,-1.83949802e-08
# ,8.31447557e-08,3.06947330e-06,-3.18319874e-06,-3.56457809e-07
# ,2.31148189e-06,4.41064990e-07,1.78055869e-06,3.54485886e-08
# ,-1.18851761e-06,-2.37010468e-09,6.52303133e-09,1.77235765e-09
# ,1.31348456e-10,5.97553681e-08,-3.18299135e-09,1.76532342e-09]
# ,[9.10091498e-09,-8.54035974e-08,-2.79946383e-07,-9.79956318e-07
# ,1.88576942e-07,-3.53635662e-07,1.13928309e-06,1.59644845e-06
# ,-1.16687172e-06,2.78509023e-07,-1.64816992e-06,4.63047715e-08
# ,1.16624731e-06,3.62573523e-09,1.65366442e-09,-4.75925166e-10
# ,-1.39966834e-10,-3.18299135e-09,1.29580126e-08,3.25279776e-09]
# ,[2.03414539e-08,4.34110138e-07,-9.79877483e-07,-4.43737844e-07
# ,-7.51321986e-07,1.29340923e-06,-2.42750105e-06,1.01953437e-05
# ,-9.35120022e-06,-1.11413557e-06,8.49154423e-06,-1.96429319e-06
# ,1.13066397e-05,1.76998343e-08,-1.40371752e-08,3.50311165e-08
# ,1.85360584e-09,1.76532342e-09,3.25279776e-09,1.26606913e-07]])

# principal_components = np.linalg.eig(covaraiance)
# # how much variance does each principal component explain
# explained_variance = principal_components[0]/np.sum(principal_components[0])
# # sort in descending order
# sorted_indices = np.argsort(explained_variance)[::-1]
# sorted_eigenvalues = principal_components[0][sorted_indices]
# sorted_eigenvectors = principal_components[1][:,sorted_indices]
# print(sorted_eigenvectors[:,0])
# # cumulative variance explained
# cumulative_variance = np.cumsum(sorted_eigenvalues)/np.sum(sorted_eigenvalues)
# # plot cumulative variance explained
# plt.plot(cumulative_variance)
# plt.show()

# fig,ax = plt.subplots(2,1,figsize=(10,6),sharex=True)
# x = sp.stats.norm.rvs(size=100000,loc=0,scale=1)
# y = np.exp(0.1*x)*0.43911709767941026
# ax[0].hist(y,bins=50,density=True)

# mode = 0.43911709767941026
# concentration = 10000
# a = mode * # b = # print(a,b)

# x = sp.stats.beta.rvs(a=a,b=b,size=100000)
# ax[1].hist(x,bins=50,density=True)
# plt.show()

# x = jnp.array([0,10,20,30])
# y = jnp.array([1,2.6,3.2,4.5])
# t = jnp.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
# # z = y[jnp.argmax(x[:,None] >= t[None,:], axis=0)]
# z = jnp.interp(t, x, y)
# plt.plot(t,z)
# plt.show()

# pathogens = ["RSV","InfluenzaA","InfluenzaB","Metapneumovirus","Adenovirus","Parainfluenza3"]
# successes = 4*jnp.ones(len(pathogens))
# idxs = [1,3,6,7,11,13,18,19,20,22]
# for i in idxs:
#     successes[i%len(pathogens)] -= 1
# data_series = pd.Series(successes,index=pathogens)
# print(data_series)

# y = jnp.array([[0.24594159,0.1857486,0.12555562,0.02660094,0.025,0.09684663,1],
# [0.40500959,0.25868484,0.11236009,0.02751321,0.025,0.09997146,1],
# [0.16297547,0.52178387,0.21929624,0.04318071,0.04170955,0.11397942,1],
# [0.10025676,1,0.71161871,0.0370452,0.0293728,0.12880827,0.90285049],
# [0.02507071,0.06970037,0.11433003,0.025,0.025,0.16546965,1],
# [0.025,0.025,0.10121682,0.025,0.025,0.13514337,1]])
# x = jnp.arange(0,7)
# pathogens = ["RSV","InfluenzaA","InfluenzaB","Metapneumovirus","Adenovirus","Parainfluenza3"]
# for i in range(6):
#     plt.plot(x,y[i],label=pathogens[i])
# plt.legend()
# plt.show()
# Ts = jnp.array([date_to_t(date) for date in ['1970-01-01', '2020-03-19', '2020-08-28', '2021-08-28', '2022-04-16']])
# Fs = jnp.array([1,0.74649061,0.97616163,0.84107913,0.97878123])

# x = jnp.linspace(date_to_t('2015-10-01'),date_to_t('2024-01-01'),1000)
# y = [cm.piecewise(t,Ts,Fs) for t in x]
# plt.plot(x,y)
# plt.show()


# hessian = jnp.array([[-1.35654718e+02,6.37871872e+04,-8.30702240e+02,9.71857854e+05
# ,1.63771113e+05,4.87044664e+06,1.58916544e+04,-2.22072870e+02
# ,-7.78329613e-01,-2.26614395e+04,-2.40650601e+03,-3.20770526e+03
# ,-1.94903182e+03,1.21906192e+03,2.73243484e+02,-2.02679020e+03
# ,1.00422895e+03,-1.06344257e+03,-4.47037631e+04,-2.85930763e+03]
# ,[6.37871872e+04,-2.27224663e+05,8.13872706e+04,-1.22120186e+05
# ,-3.06856557e+05,-1.43632002e+06,-1.01962880e+04,-9.22184967e+03
# ,-5.53645928e+01,-9.19731480e+05,-3.57812405e+04,9.07819308e+03
# ,2.03326675e+04,-1.28053171e+04,-1.44306295e+04,-8.38033752e+02
# ,-1.16835357e+04,9.54813193e+03,1.87582924e+04,-1.02223527e+04]
# ,[-8.30702240e+02,8.13872706e+04,-4.80982956e+04,-3.27570688e+04
# ,7.98192322e+04,8.61073411e+05,-5.06581983e+03,-1.79416799e+03
# ,-1.71773552e+01,-1.69649566e+05,-2.12535106e+04,-3.22911392e+04
# ,-4.62256063e+04,-5.27154209e+02,-2.33679986e+02,4.30569606e+03
# ,-1.54814167e+03,1.73298307e+03,4.45361839e+03,1.89438487e+03]
# ,[9.71857854e+05,-1.22120186e+05,-3.27570688e+04,-1.99031169e+05
# ,-5.29541139e+05,-7.56222120e+05,-1.24004808e+05,-2.90817316e+04
# ,-1.79008589e+02,-2.79173428e+06,2.11346788e+04,1.09219791e+05
# ,1.24042739e+05,-6.04542482e+03,-2.56938810e+04,-1.95328462e+04
# ,-1.43648125e+04,2.87184573e+04,5.91151161e+04,-3.47265024e+04]
# ,[1.63771113e+05,-3.06856557e+05,7.98192322e+04,-5.29541139e+05
# ,4.60444607e+04,-2.75861180e+06,-1.28418060e+03,-3.31123151e+04
# ,-1.07574048e+02,-3.05911264e+06,-3.61612441e+04,-4.91198725e+04
# ,-3.64931815e+04,-2.11922608e+03,-1.61057886e+02,-4.21154249e+03
# ,-4.07405659e+03,2.63802936e+04,6.96642083e+04,-4.51058246e+04]
# ,[4.87044664e+06,-1.43632002e+06,8.61073411e+05,-7.56222120e+05
# ,-2.75861180e+06,-2.09926058e+07,9.26531893e+05,-1.42320828e+05
# ,-8.74713036e+02,-1.38520690e+07,1.59359852e+06,1.52486760e+05
# ,2.94069130e+05,5.39556964e+04,1.66026494e+05,-1.48478490e+05
# ,-4.56106810e+04,1.09840836e+05,2.94742527e+05,-1.40613050e+05]
# ,[1.58916544e+04,-1.01962880e+04,-5.06581983e+03,-1.24004808e+05
# ,-1.28418060e+03,9.26531893e+05,-3.85073879e+03,-2.83588338e+03
# ,-2.66995987e+01,-2.50929388e+05,6.29885864e+03,-4.34949367e+03
# ,9.31113949e+03,5.79492000e+01,-4.22668152e+03,-3.71764721e+03
# ,-1.82287973e+03,1.13890186e+03,5.11130953e+03,-9.17502950e+03]
# ,[-2.22072870e+02,-9.22184967e+03,-1.79416799e+03,-2.90817316e+04
# ,-3.31123151e+04,-1.42320828e+05,-2.83588338e+03,-2.06110224e+03
# ,-1.40266785e+01,-2.48301342e+05,1.11841910e+03,7.47483825e+02
# ,1.17328027e+03,3.94198901e+02,-7.20651917e+02,-7.89209704e+01
# ,-9.15833196e+02,9.45780790e+02,5.77133894e+03,-9.78003094e+02]
# ,[-7.78329613e-01,-5.53645928e+01,-1.71773552e+01,-1.79008589e+02
# ,-1.07574048e+02,-8.74713036e+02,-2.66995987e+01,-1.40266785e+01
# ,-3.77296546e-02,-1.19547663e+03,7.07161612e+00,1.40510286e+01
# ,8.70903389e+00,1.04624402e+00,-6.54893090e+00,-4.85147406e-01
# ,-7.72258822e+00,4.08401500e+00,3.04547222e+01,-4.18354065e+00]
# ,[-2.26614395e+04,-9.19731480e+05,-1.69649566e+05,-2.79173428e+06
# ,-3.05911264e+06,-1.38520690e+07,-2.50929388e+05,-2.48301342e+05
# ,-1.19547663e+03,5.33050265e+03,1.06206018e+05,1.36443842e+05
# ,1.20040880e+05,-1.31155573e+04,-7.37123889e+04,-8.53522440e+03
# ,-9.34644156e+04,4.83255736e+05,4.56156591e+05,-5.89125635e+05]
# ,[-2.40650601e+03,-3.57812405e+04,-2.12535106e+04,2.11346788e+04
# ,-3.61612441e+04,1.59359852e+06,6.29885864e+03,1.11841910e+03
# ,7.07161612e+00,1.06206018e+05,-5.56083652e+04,-5.65916924e+04
# ,-5.46586426e+04,-1.89931524e+04,-4.65602847e+03,4.73694695e+03
# ,5.83768188e+03,-7.81324582e+02,1.44655852e+03,-5.73813317e+03]
# ,[-3.20770526e+03,9.07819308e+03,-3.22911392e+04,1.09219791e+05
# ,-4.91198725e+04,1.52486760e+05,-4.34949367e+03,7.47483825e+02
# ,1.40510286e+01,1.36443842e+05,-5.65916924e+04,-5.70325453e+04
# ,-5.56042282e+04,-1.52032185e+03,7.74209054e+03,-8.54764900e+03
# ,9.26998641e+02,-8.42845454e+02,1.50662785e+03,-6.07145357e+03]
# ,[-1.94903182e+03,2.03326675e+04,-4.62256063e+04,1.24042739e+05
# ,-3.64931815e+04,2.94069130e+05,9.31113949e+03,1.17328027e+03
# ,8.70903389e+00,1.20040880e+05,-5.46586426e+04,-5.56042282e+04
# ,-5.43594479e+04,2.87641697e+02,5.80110132e+03,4.85541651e+04
# ,1.55616171e+03,-1.36718269e+03,-2.39841482e+03,1.38346838e+03]
# ,[1.21906192e+03,-1.28053171e+04,-5.27154209e+02,-6.04542482e+03
# ,-2.11922608e+03,5.39556964e+04,5.79492000e+01,3.94198901e+02
# ,1.04624402e+00,-1.31155573e+04,-1.89931524e+04,-1.52032185e+03
# ,2.87641697e+02,-2.37145147e+04,1.60958960e+03,-1.23024054e+04
# ,2.36983928e+03,-4.42201712e+01,2.01472365e+02,3.56568860e+02]
# ,[2.73243484e+02,-1.44306295e+04,-2.33679986e+02,-2.56938810e+04
# ,-1.61057886e+02,1.66026494e+05,-4.22668152e+03,-7.20651917e+02
# ,-6.54893090e+00,-7.37123889e+04,-4.65602847e+03,7.74209054e+03
# ,5.80110132e+03,1.60958960e+03,4.16808602e+03,-9.75400691e+03
# ,1.16161406e+04,5.31410103e+02,2.05445595e+03,1.89120925e+03]
# ,[-2.02679020e+03,-8.38033752e+02,4.30569606e+03,-1.95328462e+04
# ,-4.21154249e+03,-1.48478490e+05,-3.71764721e+03,-7.89209704e+01
# ,-4.85147406e-01,-8.53522440e+03,4.73694695e+03,-8.54764900e+03
# ,4.85541651e+04,-1.23024054e+04,-9.75400691e+03,7.64506454e+03
# ,3.89651134e+02,1.32678778e+02,2.17937090e+02,2.43160780e+02]
# ,[1.00422895e+03,-1.16835357e+04,-1.54814167e+03,-1.43648125e+04
# ,-4.07405659e+03,-4.56106810e+04,-1.82287973e+03,-9.15833196e+02
# ,-7.72258822e+00,-9.34644156e+04,5.83768188e+03,9.26998641e+02
# ,1.55616171e+03,2.36983928e+03,1.16161406e+04,3.89651134e+02
# ,-2.16677122e+02,6.06349021e+02,1.66424402e+03,-6.07956795e+00]
# ,[-1.06344257e+03,9.54813193e+03,1.73298307e+03,2.87184573e+04
# ,2.63802936e+04,1.09840836e+05,1.13890186e+03,9.45780790e+02
# ,4.08401500e+00,4.83255736e+05,-7.81324582e+02,-8.42845454e+02
# ,-1.36718269e+03,-4.42201712e+01,5.31410103e+02,1.32678778e+02
# ,6.06349021e+02,-6.63391236e+04,-8.54700855e-11,6.75581301e+04]
# ,[-4.47037631e+04,1.87582924e+04,4.45361839e+03,5.91151161e+04
# ,6.96642083e+04,2.94742527e+05,5.11130953e+03,5.77133894e+03
# ,3.04547222e+01,4.56156591e+05,1.44655852e+03,1.50662785e+03
# ,-2.39841482e+03,2.01472365e+02,2.05445595e+03,2.17937090e+02
# ,1.66424402e+03,-8.54700855e-11,-2.27907640e+04,-5.55148120e-08]
# ,[-2.85930763e+03,-1.02223527e+04,1.89438487e+03,-3.47265024e+04
# ,-4.51058246e+04,-1.40613050e+05,-9.17502950e+03,-9.78003094e+02
# ,-4.18354065e+00,-5.89125635e+05,-5.73813317e+03,-6.07145357e+03
# ,1.38346838e+03,3.56568860e+02,1.89120925e+03,2.43160780e+02
# ,-6.07956795e+00,6.75581301e+04,-5.55148120e-08,-7.09564928e+04]])
# fisher_info = jnp.linalg.inv(-hessian)
# print(jnp.diag(fisher_info))
# n=0
# x = [0.2,0.2,0.2,0.2,0.3,0.9]
# OBS_AGE = jnp.zeros((7))
# remaining = 1.0
# for i in range(1,7):
#     allocation = x[n+i-1]*remaining
#     OBS_AGE[i-1] = allocation
#     remaining -= allocation
# OBS_AGE[6] = remaining
# OBS_AGE = OBS_AGE/jnp.max(OBS_AGE)
# print(OBS_AGE)
# i=0
# for pathogen in ["RSV","InfluenzaA","InfluenzaB","Metapneumovirus","Adenovirus","Parainfluenza3"]:
#     for option1 in ["X","setimport"]:
#         i+=1
#         print("\""+pathogen,"250325","FlexStepwise",option1,"flexage",str(0.01),"15 1 0.7\"")
# print(i)

# infectious_contact = jnp.genfromtxt("Data/Processed/infectious_contact_rsv.csv",delimiter=',',dtype=jnp.float64)
# import_contact = jnp.genfromtxt("Data/Processed/import_contact_rsv.csv",delimiter=',',dtype=jnp.float64)

# print(jnp.median(infectious_contact/import_contact,axis=0))
# print(jnp.sum(infectious_contact,axis=0)/jnp.sum(import_contact,axis=0))
# print(jnp.max(infectious_contact,axis=0)/jnp.max(import_contact,axis=0))
# print(jnp.mean(infectious_contact,axis=0)/jnp.mean(import_contact,axis=0))
# print(jnp.median(infectious_contact,axis=0)/jnp.median(import_contact,axis=0))

# files = [
# "InfluenzaB_deFBfl250304"
# ]

# for filefraction in files:
#     with open("Data/Processed/DE_opt_"+filefraction+".pickle","rb") as f:
#         opt = pickle.load(f)
#     print(filefraction,opt.x)

# with open("Data/Processed/results250303/DE_opt_InfluenzaA_deFluAmo250228.pickle","rb") as f:
#     opt = pickle.load(f)

# print(opt)

# with open("Data/Processed/mcmc_trajectory_from_FluA_DE.pickle","rb") as f:
#     mcmc = pickle.load(f)

# # corner plot
# variables = ["WANE","SEASONALITY","OFFSET","BETA","IMPORT_RATE","EXTRA_IMMUNITY","FIRST_IMMUNITY","FIRST_DIS_INF_FACTOR","P_OBS","OBS_AGE_YOUNG","OBS_AGE_OLD","OBS_AGE_YOUNG_OLD"]
# corner.corner(mcmc,labels=variables,show_titles=True)
# plt.savefig("Figures/Corner_plot_FluA_MCMC_from_DE.png")