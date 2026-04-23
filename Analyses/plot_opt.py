## BJS March 2025
## Plotting results of fitting

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys

from Parameters.times_and_contacts import *

from utils import *
from demography import *
from mobility_and_import import *
from sim_grid import *
from plotting import *
from fit_MCMC import *

if __name__ == "__main__":
    pathogen, seed, lockdown, option1, option2, import_multiplier = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6])

    NAG = 7 + ("split" in option1)
    
    if len(sys.argv) > 10:
        prefix = sys.argv[10]
    else:
        prefix = ""
    
    birth_rate_multiplier = 1.0
    if "brm" in option1:
        match = re.search(r'brm(\d*\.?\d+)', option1)
        if match:
            birth_rate_multiplier = float(match.group(1))

    option1_label = option1
    if int(str(seed)[:6]) < 260203:
        option1 = "orig_incidence_data" + option1
    if int(str(seed)[:6]) < 260406:
        hosp = False
    else:
        hosp = True

    start_date = '2015-07-04'
    end_date = '2025-05-01'

    option2_label = option2
    if re.match(r'\d{4}-\d{2}-\d{2}',option1):
        start_date = option1
    if re.match(r'\d{4}-\d{2}-\d{2}',option2):
        end_date = option2[0:10]
        option2 = option2[10:]

    if end_date < '2024-10-01':
        mask = [0,0]
    else:
        mask = [3135,3288]

    print(pathogen, seed)

    # set seed
    np.random.seed(seed)
    prefix, x, log_likelihood = load_optimization_results(prefix, pathogen, seed, lockdown, option1_label, option2_label)
    
    # prefix = "sampling_parameters_"
    # x = jnp.asarray([0.12032066,0.14603744,0.05796923,0.00512616,0.5582736 ,0.95180595,0.31741548,0.00618303,0.25081336,0.28657508,0.15262091,0.01914573,0.15551174,0.20378447,0.99823165])
    # log_likelihood = 12014.02

    ## artificial split
    # option1 = "split"
    # NAG = 8
    # # make x one element longer, repeat x[-4] in place
    # x = jnp.zeros(len(x_temp)+1)
    # x = x.at[:-5].set(x_temp[:-4])
    # x = x.at[-5].set(x_temp[-4])
    # x = x.at[-4].set(x_temp[-4])
    # x = x.at[-3:].set(x_temp[-3:])


    # prefix = "evosax_DE_"
    # x = jnp.array([1.1706531e-01, 7.3374316e-02, 2.2380880e-01, 9.8890215e-03, 4.5175752e-01,
    #     2.7518633e-01, 9.3584144e-01, 5.9982330e-01, 2.3880145e-01, 9.4480757e-03,
    #     2.7231514e-03, 5.5968524e-03, 2.5407155e-03, 9.6670951e-04, 1.9762345e-04,
    #     9.9309359e-04, 9.4849337e-03,])
    # log_likelihood = 0.3636939227581024
    # x = x.at[6].set(0.88)
    # if pathogen == "RSV":
    #     n = 6
    # elif pathogen == "InfluenzaA" or pathogen == "InfluenzaB":
    #     n = 7
    # else:
    #     n = 8
    # x = x.at[n].set(1)
    # x = x.at[n+1].set(0.4)
    # x = x.at[n+2].set(1)
    # x = x.at[n+3].set(0.75)
    # x = x.at[n+4].set(0)
    # x = x.at[n+5].set(1)
    # option1 = "incidence_data"
    if "incidence_data" in option1:
        if "old" in option1:
            REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=import_multiplier, incidence_data="Old", smoothed=False, hosp=hosp, NAG=NAG)
        elif "smoothed" in option1:
            REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=import_multiplier, incidence_data=True, smoothed=True, hosp=hosp, NAG=NAG)
        else:
            REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=import_multiplier, incidence_data=True, smoothed=False, hosp=hosp, NAG=NAG)
    else:
        REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=import_multiplier, incidence_data=False, hosp=hosp, NAG=NAG)
    print(data_full.shape)
    N_S = 3
    CONTACT_MATRIX = np.asarray(pd.read_csv('Data/Processed/contact_matrices/KP'+['','_split'][NAG>7]+'_contact_all_US_Census.csv', delimiter=',', header=None).values)
    BIRTH_RATE = np.genfromtxt('Data/Processed/birth_rate_daily.csv', delimiter=',')

    EPOCH = pd.to_datetime('1970-01-01')
    START = pd.to_datetime(start_date) 
    END = pd.to_datetime(end_date)
    FULL_PERIOD = pd.date_range(start=EPOCH, end=END, freq='D')
    FULL_POINTS = np.array(date_to_t(FULL_PERIOD))
    PERIOD = pd.date_range(start=START, end=END, freq='D')
    POINTS = np.array(date_to_t(PERIOD))

    start_idx = int(date_to_t(start_date) + 90 - date_to_t('2015-10-01'))
    end_idx = int(date_to_t(end_date) - date_to_t('2015-10-01'))
    data = data_full[start_idx:end_idx]

    daily_hospitalization_rates_pd = pd.read_csv('Data/Processed/KPSC_ARI_nonCOVID_hospitalization_rates_by_day_age_group'+['','_split'][NAG>7]+['','_detrended']["detrend" in option1]+'.csv',index_col=0,parse_dates=True)
    daily_hospitalization_rates_pd = daily_hospitalization_rates_pd.fillna(0)
    daily_hospitalization_rates_full = jnp.asarray(daily_hospitalization_rates_pd.values)
    daily_hospitalization_rates = daily_hospitalization_rates_full[start_idx:end_idx,]

    N = np.prod(daily_hospitalization_rates.shape)

    # print(x)
    # print("Log-Likelihood:", log_likelihood*N)

    if NAG == 7:
        from Parameters.census_population import CENSUS_AGE_POP, AGE_GROUPS, AGE_GROUP_NAMES, MEDIAN_AGE
    else:
        from Parameters.census_population import CENSUS_AGE_POP_split as CENSUS_AGE_POP, AGE_GROUPS_split as AGE_GROUPS, AGE_GROUP_NAMES_split as AGE_GROUP_NAMES, MEDIAN_AGE_split as MEDIAN_AGE
    ## Initial conditions
    STATE0 = jnp.zeros((2*N_S+1,NAG))
    STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
    STATE0 = STATE0.at[1,:].set(1)
    # # flatten initial state and add maternal immunity compartment
    STATE0 = STATE0.flatten()
    STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

    params, cntct = x_to_params(x, pathogen, lockdown, option1, option2, print_params=True, return_contact=True, NAG=NAG, wrong_aging=int(str(seed)[:6])<260414, birth_rate_multiplier=birth_rate_multiplier)
    print(cntct.shape)

    # names, bounds = parameters_names_bounds(pathogen, lockdown, option1, option2)
    # for i in range(len(names)):
    #     print(names[i]+ " (bounds: "+str(bounds[i])+")")

    solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
    values = solution.ys.T
    times = solution.ts

    likelihood = SIS_likelihood(data, daily_hospitalization_rates, params, POINTS, STATE0, p_time_to_obs, solution=solution, mask=mask, incidence_data=("incidence_data" in option1), return_sum=True, NAG=NAG)
    print(likelihood)
    # # print(likelihood.shape)
    # age_summed_likelihood = jnp.sum(likelihood, axis=1)
    # # print(jnp.min(age_summed_likelihood))
    # normalized_likelihood = age_summed_likelihood/jnp.min(age_summed_likelihood)
    # # print(jnp.max(normalized_likelihood))
    # mask = [3135,3288]
    # full_likelihood = jnp.zeros(len(POINTS))
    # full_likelihood = full_likelihood.at[90:90+mask[0]].set(normalized_likelihood[:mask[0]]).at[90+mask[1]:90+mask[1]+(len(normalized_likelihood)-mask[0])].set(normalized_likelihood[mask[0]:])
    # # print(full_likelihood)

    # # for each season from the 2015/16 season onwards, sum the total number of infections
    seasons = np.array([date_to_t(date) for date in ['2015-10-01','2016-10-01','2017-10-01','2018-10-01','2019-10-01','2020-10-01','2021-10-01','2022-10-01','2023-10-01','2024-10-01','2025-05-01']])
    season_infection_array = np.zeros((len(seasons)-1,3))
    season_infection_by_age = np.zeros((len(seasons)-1,NAG,3))
    first_infections = np.zeros((len(seasons)-1,NAG))
    population_size = calculate_population_size(values, NAG=NAG)
    for i in range(len(seasons)-1):
        # get the number of infections in each season
        season_start = np.argmax(times>=seasons[i])
        season_end = np.argmax(times>=seasons[i+1])
        pop_size = np.sum(values[:-NAG,season_start])
        age_pops = population_size[season_start]
        first_infections[i,:] = np.sum(values[1:1+NAG,season_start:season_end],axis=1)
        season_infection_array[i,0] = np.sum(values[1+NAG:1+2*NAG,season_start:season_end])*REC_UP[0]/pop_size
        season_infection_array[i,1] = np.sum(values[1+3*NAG:1+4*NAG,season_start:season_end])*REC_UP[1]/pop_size
        season_infection_array[i,2] = np.sum(values[1+5*NAG:1+6*NAG,season_start:season_end])*REC_SAME[2]/pop_size
        season_infection_by_age[i,:,0] = np.sum(values[1+NAG:1+2*NAG,season_start:season_end],axis=1)*REC_UP[0]/age_pops
        season_infection_by_age[i,:,1] = np.sum(values[1+3*NAG:1+4*NAG,season_start:season_end],axis=1)*REC_UP[1]/age_pops
        season_infection_by_age[i,:,2] = np.sum(values[1+5*NAG:1+6*NAG,season_start:season_end],axis=1)*REC_SAME[2]/age_pops 
    average_age_of_first_infection = np.sum(first_infections*jnp.array(MEDIAN_AGE).reshape((1,NAG)),axis=1)/jnp.sum(first_infections,axis=1)
    season_infections = np.sum(season_infection_array,axis=1)
    season_infection_by_age = np.sum(season_infection_by_age,axis=2)
    # print("Average age of first infection per season:",average_age_of_first_infection/12)
    print("Proportion infected per season (including reinfections):",season_infections)
    print("Proportion infected in last season (by age):",season_infection_by_age[-1,:])

    incidence = calculate_proportion_positive_incidence(pathogen, aggregation="D", window_size=1, weighting_factor=0, sum_age_groups=False, save_counts=False, pp_only=False, hosp=hosp, return_counts=True, NAG=NAG)
    incidence = incidence.fillna(0)
    obs_per_season = jnp.asarray(calculate_observations_per_season(incidence, age_groups=True, aggregation="D"))
    def center_of_gravity(season_data):
        """Calculate center of gravity: sum(days * incidence) / sum(incidence)"""
        season_start_idx = season_data.index[0]
        season_start_numeric = (season_start_idx - EPOCH).days
        days = np.arange(len(season_data))
        if season_data.ndim == 1:
            total = np.sum(season_data.values)
            if total == 0:
                return np.nan
            return season_start_numeric + np.sum(days * season_data.values) / total
        else:
            # For multi-column data, compute center of gravity for each column
            totals = np.sum(season_data.values, axis=0)
            result = season_start_numeric + np.sum(days[:, None] * season_data.values, axis=0) / np.maximum(totals, 1e-10)
            result[totals == 0] = np.nan
            return result
    peak_times_by_season = incidence.groupby(incidence.index.map(get_season_start)).apply(center_of_gravity)
    # peak_times_by_season is now a Series of arrays; convert to list of time indices per season
    peak_times = [pd.to_timedelta(np.asarray(pt, dtype=int), unit='D') if isinstance(pt, np.ndarray) else pd.to_timedelta(int(pt), unit='D') for pt in peak_times_by_season]
    # print("Peak time of each season:\n", peak_times)
    # difference between 2017/18 and 2022/23 seasons
    print(peak_times[2])
    peak_time_diff = (peak_times[7] - peak_times[2]).days - 365*5
    # this pritns in vertical format, just printa s a list
    print("Difference in peak times between 2017/18 and 2022/23 seasons (in days):", peak_time_diff.tolist())

    population_size = calculate_population_size(values, N_S=N_S, NAG=NAG)
    expected_obs = calculate_expected_obs(values, p_time_to_obs, len(times), NAG=NAG)
    cut_times = times[:-1]
    expected_obs_per_season = calculate_observations_per_season_jax(expected_obs, age_groups=True, NAG=NAG)
    # Assign each time point to a season (numeric season id/start)
    season_ids = jax.vmap(get_season_start_jax)(cut_times)
    unique_seasons = 16684 + 365 * jnp.arange(10)  # Assuming seasons start on day 259 of each year
    
    # For each season, find the center of gravity of expected observation (per age group)
    def season_peak_times(season_id):
        season_mask = season_ids == season_id                           # (T,)
        masked_obs = jnp.where(season_mask[:, None], expected_obs, 0)   # (T, NAG)
        # Calculate center of gravity: sum(time_idx * obs) / sum(obs)
        time_indices = jnp.arange(len(cut_times))
        numerator = jnp.sum(time_indices[:, None] * masked_obs, axis=0)  # (NAG,)
        denominator = jnp.sum(masked_obs, axis=0)                       # (NAG,)
        peak_idx = numerator / jnp.maximum(denominator, 1e-10)          # (NAG,) - avoid division by zero
        return times[jnp.asarray(peak_idx, dtype=int)]                  # (NAG,)
    
    expected_peak_times = jax.vmap(season_peak_times)(unique_seasons) 
    # Convert peak times to dates and display in a nice table format
    peak_dates = [[t_to_date(t).strftime('%Y-%m-%d') for t in season] for season in expected_peak_times]
    peak_df = pd.DataFrame(peak_dates, columns=[age_name for age_name in AGE_GROUP_NAMES], index=[f"Season {i+1}" for i in range(len(unique_seasons))])
    peak_df.index.name = "Season"
    # print(peak_df.to_string())
    # difference in peak times between 2017/18 and 2022/23 seasons
    expected_peak_time_diff = (expected_peak_times[7] - expected_peak_times[2]) - 365*5
    print("Difference in expected peak times between 2017/18 and 2022/23 seasons (in days):", expected_peak_time_diff)

    # Calculate ratio of ratios for observed data
    pre_pandemic_observed_obs = obs_per_season[:5]
    # rebound_season_idx is the first season after 2019/20 with total infections exceeding threshold of median pre-pandemic season
    threshold = 1/2
    rebound_season_idx = np.where(np.sum(obs_per_season[5:], axis=1) > threshold * np.median(np.sum(obs_per_season[:5], axis=1)))[0][0]
    rebound_observed_obs = obs_per_season[5+rebound_season_idx]

    from sim_grid import age_ratio_of_rebound
    print(age_ratio_of_rebound(jnp.stack([obs_per_season, obs_per_season], axis=0)))
    
    obs_ratio_matrix = np.zeros((NAG, NAG))
    for i in range(NAG):
        for j in range(NAG):
            ratio_of_ratios = age_ratio_of_rebound(jnp.stack([obs_per_season, obs_per_season], axis=0), idx_num=i, idx_den=j, threshold_factor=threshold)
            obs_ratio_matrix[i, j] = ratio_of_ratios

    expected_ratio_matrix = np.zeros((NAG, NAG))
    for i in range(NAG):
        for j in range(NAG):
            ratio_of_ratios = age_ratio_of_rebound(jnp.stack([expected_obs_per_season, expected_obs_per_season], axis=0), idx_num=i, idx_den=j, threshold_factor=threshold)
            expected_ratio_matrix[i, j] = ratio_of_ratios

    
    # Calculate differences between observed and expected, weighted by distance from 1
    diff_matrix = np.abs(obs_ratio_matrix - expected_ratio_matrix)
    # Weight by distance from 1 (prioritize values far from 1)
    weight_matrix = np.abs(obs_ratio_matrix - 1) + np.abs(expected_ratio_matrix - 1)
    
    # Create weighted diff only for lower triangle
    weighted_diff_matrix = np.full_like(diff_matrix, np.nan)
    for i in range(NAG):
        for j in range(i):  # Only lower triangle
            with np.errstate(divide='ignore', invalid='ignore'):
                weighted_diff_matrix[i, j] = diff_matrix[i, j] / weight_matrix[i, j]
    
    # Find three smallest and three largest weighted differences (lower triangle only)
    flat_diffs = weighted_diff_matrix[np.isfinite(weighted_diff_matrix)].flatten()
    sorted_diffs = np.sort(flat_diffs)
    three_smallest = sorted_diffs[:3]
    three_largest = sorted_diffs[-3:]
    
    # Create formatted dataframes with markers (lower triangle only)
    obs_ratio_df = pd.DataFrame(obs_ratio_matrix, index=AGE_GROUP_NAMES, columns=AGE_GROUP_NAMES)
    obs_ratio_display = obs_ratio_df.astype(object)
    
    expected_ratio_df = pd.DataFrame(expected_ratio_matrix, index=AGE_GROUP_NAMES, columns=AGE_GROUP_NAMES)
    expected_ratio_display = expected_ratio_df.astype(object)
    
    # Mark three closest and three most different (lower triangle only)
    for i in range(NAG):
        for j in range(NAG):
            if i <= j:  # Skip upper triangle and diagonal
                obs_ratio_display.iat[i, j] = ""
                expected_ratio_display.iat[i, j] = ""
            else:
                marker = ""
                if np.isfinite(weighted_diff_matrix[i, j]):
                    if weighted_diff_matrix[i, j] in three_smallest:
                        marker = "*"
                    elif weighted_diff_matrix[i, j] in three_largest:
                        marker = "$"
                
                obs_ratio_display.iat[i, j] = f"{obs_ratio_matrix[i, j]:.3f}{marker}"
                expected_ratio_display.iat[i, j] = f"{expected_ratio_matrix[i, j]:.3f}{marker}"
    
    print("Observed ratio of ratios matrix (lower triangle only):")
    print(obs_ratio_display.to_string())
    print()
    print("Expected ratio of ratios matrix (lower triangle only):")
    print(expected_ratio_display.to_string())
    print(f"\n* = three closest matches furthest from 1 (weighted diffs: {three_smallest})")
    print(f"$ = three most different furthest from 1 (weighted diffs: {three_largest})")

    # population_size = calculate_population_size(values, N_S=N_S, NAG=NAG)
    # # trajectory is total proportion infected over time
    # infectious = jnp.sum(values[1:].reshape((2*N_S+1, NAG, -1))[1:2*N_S:2], axis=0).T
    # expected_infectious = jax.nn.softplus(infectious[-len(tests):]*100)/100
    # expected_prevalence = jnp.divide(expected_infectious, population_size[-len(tests):])
    # print("Average prevalence over observed period:",jnp.mean(expected_prevalence, axis=0))
    # print("Peak prevalence over observed period:",jnp.max(expected_prevalence, axis=0))

    # # # get R(t)
    # # R0s = jnp.zeros(len(times))
    # # Rts = jnp.zeros(len(times))
    # # contact_ratios = jnp.zeros(len(times))
    # # SEASONALITY = x[-1]
    # # OFFSET = x[-2]

    # # for idx in range(len(times)):
    # #     pop_size = jnp.sum(values[:,idx],dtype=jnp.float64)
    # #     age_pops = jnp.array([jnp.sum(values[range(i_age,(2*N_S+1)*NAG,NAG),idx],axis=0) for i_age in range(NAG)])
    # #     contact_t = contact(times[idx],SEASONALITY,OFFSET)
    # #     infectious_contact_equal = jnp.dot(contact_t,jnp.sum(jnp.array([values[j,idx] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=jnp.float64).reshape((N_S,NAG))*I_REL,axis=0))/jnp.sum(jnp.array([values[j,idx] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=jnp.float64))
    # #     infectious_contact = jnp.dot(contact_t,jnp.sum(jnp.array([values[j,idx] for j in range(NAG,(2*N_S+1)*NAG) if (j//NAG)%2==0],dtype=jnp.float64).reshape((N_S,NAG))*I_REL,axis=0))/pop_size
    # #     import_contact = IMPORT_RATE*regional_positivity(times[idx])*arrivals(times[idx])*jnp.dot(contact_t,age_pops)/pop_size
    # #     contact_ratios[idx] = jnp.sum(jnp.repeat(S_REL,NAG*N_C)*jnp.tile(S_AGE,N_S*N_C)*BETA*jnp.tile(import_contact,N_S*N_C)*jnp.repeat(jnp.tile(jnp.array([0,1]+[0]*(N_C-2)),N_S),NAG)*jnp.array([jnp.tile(values[(2*i+1)*NAG:(2*i+2)*NAG,idx],N_C) for i in range(N_S)]).flatten())/jnp.sum(jnp.repeat(S_REL,NAG*N_C)*jnp.tile(S_AGE,N_S*N_C)*BETA*jnp.tile(infectious_contact,N_S*N_C)*jnp.repeat(jnp.tile(jnp.array([0,1]+[0]*(N_C-2)),N_S),NAG)*jnp.array([jnp.tile(values[(2*i+1)*NAG:(2*i+2)*NAG,idx],N_C) for i in range(N_S)]).flatten())
    # #     R0s[idx] = BETA*jnp.sum(infectious_contact_equal)/REC_UP[0]
    # #     Rts[idx] = (1/pop_size)*(1/REC_UP[0])*jnp.sum(jnp.repeat(S_REL,NAG*N_C)*jnp.tile(S_AGE,N_S*N_C)*BETA*jnp.tile(infectious_contact_equal,N_S*N_C)*jnp.repeat(jnp.tile(jnp.array([1,0]+[0]*(N_C-2)),N_S),NAG)*jnp.array([jnp.tile(values[(2*i+1)*NAG:(2*i+2)*NAG,idx],N_C) for i in range(N_S)]).flatten())
    # # print("R0:",jnp.median(R0s),"("+str(jnp.min(R0s))+"–"+str(jnp.max(R0s))+")")
    # # print("Rt:",jnp.median(Rts),"("+str(jnp.min(Rts))+"–"+str(jnp.max(Rts))+")")
    # # print("Ratio of import-caused cases to internal transmission:",jnp.median(contact_ratios),"("+str(jnp.min(contact_ratios))+"–"+str(jnp.max(contact_ratios))+")")

    # remove "free" from pathogen name for plotting
    if "free" in pathogen:
        pathogen_name = pathogen.replace("free","")
    else:
        pathogen_name = pathogen

    plt.rcParams.update({'font.size':8})
    # text type is palatino
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Palatino']

    hsv_colors = colormaps.hsv(-0.02+np.arange(NAG)/NAG)
    hsv_colors[3] = colormaps.hsv((3/NAG)+0.28/NAG)

    fig = plt.figure(figsize=(5.5,5.5))
    # ax = fig.add_subplot(1,1,1)
    # aggregation = "MS"
    # mx = lockdown_incidence_plot(ax,STATE0,params,POINTS,date_to_t('2020-03-19'),solution=solution,label=None,by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=[1,7,30.44][[None,"W","MS"].index(aggregation)]*10000,p_time_to_obs=p_time_to_obs, color=hsv_colors, linewidth=0.5, NAG=NAG)
    # lockdown_incidence_format(ax,date_to_t('2020-03-19'),365,mx,year_window=2)
    # plt.savefig("Figures/"+prefix+"incidence_plot_"+pathogen+"_"+str(seed)+"_"+option1_label+"_"+option2_label+".png", dpi=300, bbox_inches='tight')

    ax1 = fig.add_subplot(3,1,1)
    
    # Create a grid of 2 rows x 4 columns in the middle
    gs = fig.add_gridspec(2, 4, top=0.65, bottom=0.35)

    ax_grid = [[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(2)]
    
    ax4 = fig.add_subplot(3,1,3)
    ax = [ax1, ax_grid, ax4]
    aggregation = "MS"

    # if option1 == "old_incidence_data":
    #     dmx = kpsc_positive_test_plot(ax[1], pathogen, AGE_GROUPS, AGE_GROUP_NAMES, color=hsv_colors, legend=False, aggregation=aggregation, factor=10000)
    # else:
    #     dmx = kpsc_proportion_positive_incidence_plot(ax[1], pathogen, AGE_GROUPS, AGE_GROUP_NAMES, aggregation=aggregation, factor=10000, hosp=hosp)
    # ax[1].set_xlabel("")
    pnamedict = {"RSV":"RSV","InfluenzaA":"Influenza A","InfluenzaB":"Influenza B","Parainfluenza3":"Parainfluenza 3","Adenovirus":"Adenovirus","Metapneumovirus":"Metapneumovirus", "test":"test"}
    # # ax.set_title("Observed incidence of "+pnamedict[pathogen_name])
    # ax[1].set_ylabel("Monthly incidence per 10k")
    # # legend
    # ax[1].legend(frameon=False, fontsize=6, ncol=3)

    # # fig, ax = plt.subplots(1,2,figsize=(14.5,2.8))
    # mx = lockdown_incidence_plot(ax[2],STATE0,params,POINTS,date_to_t('2020-03-19'),solution=solution,label="Simulation",by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=[1,7,30.44][[None,"W","MS"].index(aggregation)]*10000,p_time_to_obs=p_time_to_obs)
    # lockdown_incidence_format(ax[2],date_to_t('2020-03-19'),365,mx,year_window=2)
    # if lockdown == "ExponentialByAge":
    #     ax[2].plot(POINTS, np.maximum(dmx,mx)*cntct[-len(POINTS):,0], label="Relative contact rate", color="black", linestyle="dashed")
    #     ax[2].plot(POINTS, np.maximum(dmx,mx)*cntct[-len(POINTS):,-1], label="Relative contact rate", color="silver", linestyle="dashed")
    # else:
    #     ax[2].plot(POINTS, np.maximum(dmx,mx)*cntct[-len(POINTS):], label="Relative contact rate", color="black", linestyle="dashed")
    # # ax[2].plot(POINTS, mx*full_likelihood, label="Normalized likelihood", color="black", alpha=0.5)

    for i_age in range(NAG):
        age_ax = ax_grid[i_age // 4][i_age % 4]
        if "orig_incidence_data" in option1:
            dmx = kpsc_positive_test_plot(age_ax, pathogen, AGE_GROUPS, AGE_GROUP_NAMES, color="k", legend=False, aggregation=aggregation, factor=10000, select_age_group=i_age, orig=True, linewidth=0.5)
        elif option1 == "old_incidence_data":
            dmx = kpsc_positive_test_plot(age_ax, pathogen, AGE_GROUPS, AGE_GROUP_NAMES, color="k", legend=False, aggregation=aggregation, factor=10000, select_age_group=i_age, linewidth=0.5)
        else:
            dmx = kpsc_proportion_positive_incidence_plot(age_ax, pathogen, AGE_GROUPS, AGE_GROUP_NAMES, select_age_group=i_age, aggregation=aggregation, factor=10000, color="black", hosp=hosp, detrend=("detrend" in option1), linewidth=0.5)
        mx = lockdown_incidence_plot(age_ax,STATE0,params,POINTS,date_to_t('2020-03-19'),solution=solution,label=None,by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=[1,7,30.44][[None,"W","MS"].index(aggregation)]*10000,p_time_to_obs=p_time_to_obs, select_age_group=i_age, color=hsv_colors[i_age], linewidth=0.5, NAG=NAG)
        lockdown_incidence_format(age_ax,date_to_t('2020-03-19'),365,mx,year_window=2)
        age_ax.legend(frameon=False, fontsize=6)
    if NAG < 8:
        omx = lockdown_incidence_plot(ax_grid[-1][-1],STATE0,params,POINTS,date_to_t('2020-03-19'),solution=solution,label="Simulation",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=[1,7,30.44][[None,"W","MS"].index(aggregation)]*10000,p_time_to_obs=p_time_to_obs, color="grey", linewidth=0.5, NAG=NAG)
        if lockdown == "ExponentialByAge":
            ax_grid[-1][-1].plot(POINTS, omx*cntct[-len(POINTS):,0], label="<40y contacts", color="black", linestyle="dashed")
            ax_grid[-1][-1].plot(POINTS, omx*cntct[-len(POINTS):,-1], label=">40y contacts", color="silver", linestyle="dashed")
        else:
            ax_grid[-1][-1].plot(POINTS, omx*cntct[-len(POINTS):], label="Relative contact rate", color="black", linestyle="dashed")
    # ax_grid[-1][-1].legend(frameon=False, fontsize=6)
   
    for ax_row in ax_grid:
        for gridax in ax_row:
            # strip of title, x and y labels, ticks etc.
            gridax.set_title("")
            gridax.set_xlabel("")
            gridax.set_ylabel("")
            gridax.set_xticklabels("")
            gridax.set_yticklabels("")
            gridax.set_yticks([])
            gridax.set_xticks([])

    lockdown_susceptibility_plot(ax[2],STATE0,params,PERIOD,POINTS,date_to_t('2020-03-19'),solution=solution,relative=True,proportion=True, by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES, NAG=NAG)
    lockdown_susceptibility_format(ax[2],date_to_t('2020-03-19'),365,year_window=2,ymax=None,ymin=None)
    # ax[3].set_title("Effective susceptibles")

    if option1 == "old_incidence_data":
        kpsc_positive_test_plot(ax[0], pathogen, None, AGE_GROUP_NAMES, aggregation=aggregation, factor=10000, color="black", label="Data")
    elif "orig_incidence_data" in option1:
        kpsc_positive_test_plot(ax[0], pathogen, None, AGE_GROUP_NAMES, aggregation=aggregation, factor=10000, color="black", label="Data", orig=True)
    else:
        kpsc_proportion_positive_incidence_plot(ax[0], pathogen, None, AGE_GROUP_NAMES, aggregation=aggregation, factor=10000, color="black", label="Data", hosp=hosp, detrend=("detrend" in option1))
    mx = lockdown_incidence_plot(ax[0],STATE0,params,POINTS,date_to_t('2020-03-19'),solution=solution,label="Simulation",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=[1,7,30.44][[None,"W","MS"].index(aggregation)]*10000,p_time_to_obs=p_time_to_obs)
    lockdown_incidence_format(ax[0],date_to_t('2020-03-19'),365,mx,year_window=2)
    ax[0].legend(frameon=False, fontsize=6)

    ax[0].set_title("")
    ax[0].set_xlabel("")
    ax[0].set_xticklabels("")
    ax[0].set_ylabel("Incidence\nper 10k")
    # ax[1].set_title("")
    # ax[1].set_xlabel("")
    # ax[1].set_xticklabels("")
    # ax[1].set_ylabel("Age-structured\ndata")
    # ax[2].set_title("")
    # ax[2].set_xlabel("")
    # ax[2].set_xticklabels("")
    # ax[2].set_ylabel("Age-structured\nsimulation")
    ax[2].set_title("")
    ax[2].set_xlabel("Date")
    ax[2].set_ylabel("Effective susceptibility")

    # pathogen as title
    fig.suptitle(pnamedict[pathogen_name], fontsize=10)
    # likelihood as subtitle
    fig.text(0.5, 0.92, "Log-Likelihood: "+str(np.round(-log_likelihood*N,0)), ha='center', fontsize=8)

    # plt.tight_layout()
    plt.savefig("Figures/"+prefix+pathogen+lockdown+option1+option2_label+str(seed)+".png",dpi=300)
    plt.close()

    # fig, ax = plt.subplots(figsize=(4,4))
    # aggregation = "W"
    # kpsc_proportion_positive_incidence_plot(ax, pathogen, None, AGE_GROUP_NAMES, aggregation=aggregation, factor=10000, color="black", label="Data", hosp=hosp, detrend=("detrend" in option1))
    # mx = lockdown_incidence_plot(ax,STATE0,params,POINTS,date_to_t('2020-03-19'),solution=solution,by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=[1,7,30.44][[None,"W","MS"].index(aggregation)]*10000,p_time_to_obs=p_time_to_obs, color="silver", label="Simulation")
    # lockdown_incidence_format(ax,date_to_t('2020-03-19'),365,mx,year_window=2)
    # plt.tight_layout()
    # plt.savefig("Figures/"+prefix+pathogen+lockdown+option1+option2_label+str(seed)+"_monthly_noage.png",dpi=300)
    # plt.close()

    # ax[1].set_title("Simulated incidence of "+pnamedict[pathogen])
    # ax[1].set_xlabel("")
    # ax[1].set_ylabel("")
    # ax[1].set_xlabel("")
    # ax[1].set_xticklabels(["","2016","","2018","","2020","","2022","","2024"])
    # ax[0].set_yticks([])
    # lockdown_susceptibility_plot(ax[2],STATE0,params,PERIOD,POINTS,date_to_t('2020-03-19'),solution=solution,relative=True,proportion=True, by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES)
    # lockdown_susceptibility_format(ax[2],date_to_t('2020-03-19'),365,year_window=2,ymax=None,ymin=None)
    # ax[2].set_title("Effective susceptibles")
    # ax[1].set_xlabel("")
    # ax[1].set_ylabel("")
    # ax[1].set_xlabel("")
    # ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # ax.set_xticklabels(["","2016","","2018","","2020","","2022","","2024",""])
    # # ax.set_yscale('log')
    # # ax.set_ylim(1e-3,)
    # # multiply y lables by 100
    # ylabls = ax.get_yticks()
    # ax.set_yticklabels([str(int(np.round(yl*100))) for yl in ylabls])
    # plt.tight_layout()
    # plt.savefig("Figures/DE_"+pathogen+lockdown+option1+option2+str(seed)+"_mini_noage_weekly.png",dpi=300)