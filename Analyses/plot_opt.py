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
    option1_label = option1
    option2_label = option2

    AGE_GROUPS = None

    # print("-------!!!!!!!----- adding months to option1 -----!!!!!!!-------")
    # option1 = option1 + "months"

    # print("-------!!!!!!!----- adding to option2 -----!!!!!!!-------")
    # option2 = option2 + "

    if "months" in option1:
        from Parameters.census_population import CENSUS_AGE_POP_months as CENSUS_AGE_POP
        from Parameters.census_population import AGE_GROUPS_split as AGE_GROUPS
        NAG = 65
        max_month = 12*5
    elif "split" in option1:
        from Parameters.census_population import CENSUS_AGE_POP_split as CENSUS_AGE_POP
        NAG = 8
        max_month = None
    else:
        from Parameters.census_population import CENSUS_AGE_POP
        NAG = 7
        max_month = None
    
    if NAG == 65:
        NAG_eff = 7 + ("split" in option1)
    else:
        NAG_eff = NAG

    if len(sys.argv) > 10:
        prefix = sys.argv[10]
    else:
        prefix = ""
    
    birth_rate_multiplier = 1.0
    if "brm" in option1:
        match = re.search(r'brm(\d*\.?\d+)', option1)
        if match:
            birth_rate_multiplier = float(match.group(1))

    if int(str(seed)[:6]) < 260203:
        option1 = "orig_incidence_data" + option1
    if int(str(seed)[:6]) < 260406:
        hosp = False
    else:
        hosp = True

    start_date = '2015-07-04'
    end_date = '2025-05-01'

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

    # x[1] = 0.03
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
            REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=import_multiplier, incidence_data="Old", smoothed=False, hosp=hosp, NAG=NAG, dedup=("dedup" in option1))
        elif "smoothed" in option1:
            REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=import_multiplier, incidence_data=True, smoothed=True, hosp=hosp, NAG=NAG, dedup=("dedup" in option1))
        else:
            REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=import_multiplier, incidence_data=True, smoothed=False, hosp=hosp, NAG=NAG, dedup=("dedup" in option1))
    else:
        REC_UP, REC_SAME, IMPORT_STRENGTH, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=import_multiplier, incidence_data=False, hosp=hosp, NAG=NAG, dedup=("dedup" in option1))
    print(data_full.shape)
    N_S = 3
    if "months" in option1:
        CONTACT_MATRIX = jnp.asarray(pd.read_csv('Data/Processed/contact_matrices/MONTHS_contact_all_US_Census.csv', delimiter=',', header=None).values)
    else:
        CONTACT_MATRIX = jnp.asarray(pd.read_csv('Data/Processed/contact_matrices/KP'+['', '_split'][NAG>7]+'_contact_all_US_Census.csv', delimiter=',', header=None).values)
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

    daily_hospitalization_rates_pd = pd.read_csv('Data/Processed/KPSC_ARI_nonCOVID_hospitalization_rates_by_day_age_group'+['','_split'][NAG>7]+['','_detrended']["detrend" in option1]+["","_dedup"]["dedup" in option1]+'.csv',index_col=0,parse_dates=True)
    daily_hospitalization_rates_pd = daily_hospitalization_rates_pd.fillna(0)
    daily_hospitalization_rates_full = jnp.asarray(daily_hospitalization_rates_pd.values)
    daily_hospitalization_rates = daily_hospitalization_rates_full[start_idx:end_idx,]

    N = np.prod(daily_hospitalization_rates.shape)

    # print(x)
    # print("Log-Likelihood:", log_likelihood*N)

    if NAG == 7:
        from Parameters.census_population import CENSUS_AGE_POP, AGE_GROUPS, AGE_GROUP_NAMES, MEDIAN_AGE
    elif "months" in option1:
        from Parameters.census_population import CENSUS_AGE_POP_months as CENSUS_AGE_POP, AGE_GROUPS_split as AGE_GROUPS, AGE_GROUP_NAMES_split as AGE_GROUP_NAMES, MEDIAN_AGE_split as MEDIAN_AGE
    else:
        from Parameters.census_population import CENSUS_AGE_POP_split as CENSUS_AGE_POP, AGE_GROUPS_split as AGE_GROUPS, AGE_GROUP_NAMES_split as AGE_GROUP_NAMES, MEDIAN_AGE_split as MEDIAN_AGE
    ## Initial conditions
    STATE0 = jnp.zeros((2*N_S+1,NAG))
    STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
    STATE0 = STATE0.at[1,:].set(1)
    # # flatten initial state and add maternal immunity compartment
    STATE0 = STATE0.flatten()
    STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

    params = x_to_params(x, pathogen, lockdown, option1, option2, print_params=True, return_contact=False, NAG=NAG, wrong_aging=int(str(seed)[:6])<260414, birth_rate_multiplier=birth_rate_multiplier)
    # print(cntct.shape)

    # names, bounds = parameters_names_bounds(pathogen, lockdown, option1, option2)
    # for i in range(len(names)):
    #     print(names[i]+ " (bounds: "+str(bounds[i])+")")

    solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
    values = solution.ys.T
    times = solution.ts

    # likelihood = SIS_likelihood(data, daily_hospitalization_rates, params, POINTS, STATE0, p_time_to_obs, mask=mask, incidence_data=("incidence_data" in option1), return_sum=True, NAG=NAG, AGE_GROUPS=AGE_GROUPS, max_month=max_month)
    # print("DE likelihood:", -N*log_likelihood, ", calculated likelihood:", likelihood)
    # # # print(likelihood.shape)
    # # age_summed_likelihood = jnp.sum(likelihood, axis=1)
    # # # print(jnp.min(age_summed_likelihood))
    # # normalized_likelihood = age_summed_likelihood/jnp.min(age_summed_likelihood)
    # # # print(jnp.max(normalized_likelihood))
    # # mask = [3135,3288]
    # # full_likelihood = jnp.zeros(len(POINTS))
    # # full_likelihood = full_likelihood.at[90:90+mask[0]].set(normalized_likelihood[:mask[0]]).at[90+mask[1]:90+mask[1]+(len(normalized_likelihood)-mask[0])].set(normalized_likelihood[mask[0]:])
    # # # print(full_likelihood)

    # # # for each season from the 2015/16 season onwards, sum the total number of infections
    # seasons = np.array([date_to_t(date) for date in ['2015-10-01','2016-10-01','2017-10-01','2018-10-01','2019-10-01','2020-10-01','2021-10-01','2022-10-01','2023-10-01','2024-10-01','2025-05-01']])
    # season_infection_array = np.zeros((len(seasons)-1,3))
    # season_infection_by_age = np.zeros((len(seasons)-1,NAG,3))
    # first_infections = np.zeros((len(seasons)-1,NAG))
    # population_size = calculate_population_size(values, NAG=NAG)
    # for i in range(len(seasons)-1):
    #     # get the number of infections in each season
    #     season_start = np.argmax(times>=seasons[i])
    #     season_end = np.argmax(times>=seasons[i+1])
    #     pop_size = np.sum(values[:-NAG,season_start])
    #     age_pops = population_size[season_start]
    #     first_infections[i,:] = np.sum(values[1:1+NAG,season_start:season_end],axis=1)
    #     season_infection_array[i,0] = np.sum(values[1+NAG:1+2*NAG,season_start:season_end])*REC_UP[0]/pop_size
    #     season_infection_array[i,1] = np.sum(values[1+3*NAG:1+4*NAG,season_start:season_end])*REC_UP[1]/pop_size
    #     season_infection_array[i,2] = np.sum(values[1+5*NAG:1+6*NAG,season_start:season_end])*REC_SAME[2]/pop_size
    #     season_infection_by_age[i,:,0] = np.sum(values[1+NAG:1+2*NAG,season_start:season_end],axis=1)*REC_UP[0]/age_pops
    #     season_infection_by_age[i,:,1] = np.sum(values[1+3*NAG:1+4*NAG,season_start:season_end],axis=1)*REC_UP[1]/age_pops
    #     season_infection_by_age[i,:,2] = np.sum(values[1+5*NAG:1+6*NAG,season_start:season_end],axis=1)*REC_SAME[2]/age_pops 
    # # average_age_of_first_infection = np.sum(first_infections*jnp.array(MEDIAN_AGE).reshape((1,NAG)),axis=1)/jnp.sum(first_infections,axis=1)
    # season_infections = np.sum(season_infection_array,axis=1)
    # season_infection_by_age = np.sum(season_infection_by_age,axis=2)
    # # print("Average age of first infection per season:",average_age_of_first_infection/12)
    # print("Proportion infected per season (including reinfections):",season_infections)
    # print("Proportion infected in last season (by age):",season_infection_by_age[-1,:])

    shaped_values = values[1:,].reshape((1+2*N_S, NAG, -1))
    infectious = shaped_values[1:2*N_S:2, :, :]
    first_infectious_by_age = infectious[0, :, :]
    from Parameters.census_population import MEDIAN_AGE_split
    mean_ages_of_first_infection = jnp.mean(first_infectious_by_age[:,:365*4],axis=1)/jnp.sum(jnp.mean(first_infectious_by_age[:,:365*4],axis=1))
    mean_age_of_first_infection = jnp.sum(first_infectious_by_age*jnp.array(MEDIAN_AGE_split).reshape((NAG,1)),axis=0)/jnp.sum(first_infectious_by_age,axis=0)
    # mean over time to get average age of first infection
    print("--!!!--- Mean Age of First Infection --!!!---")
    print(f"{mean_ages_of_first_infection.sum():.2g}")
    print("Age distribution of first infection (%):", [f"{x:.2g}" for x in 100*mean_ages_of_first_infection])
    # print sum of first three elements of mean_ages_of_first_ifnection
    print("Proportion of first infections under five:", f"{mean_ages_of_first_infection[:3].sum():.2g}")
    print("Average age of first infection:", f"{jnp.mean(mean_age_of_first_infection[:365*4])/12:.2g}")

    SUM_FIRST_TWO_AGE_GROUPS = False
    if SUM_FIRST_TWO_AGE_GROUPS:
        print("--!!!--- Summing first two age groups to get 0-4 age group --!!!---")

    incidence = calculate_proportion_positive_incidence(pathogen, aggregation="D", window_size=1, weighting_factor=0, sum_age_groups=False, save_counts=False, pp_only=False, hosp=hosp, return_counts=False, NAG=NAG, dedup=("dedup" in option1))    
    incidence = incidence.fillna(0)
    incidence_jax = jnp.asarray(incidence.values)
    if SUM_FIRST_TWO_AGE_GROUPS:
        # sum first two age groups to get 0-1 age group
        incidence_jax = incidence_jax.at[:,:2].set(jnp.sum(incidence_jax[:,:2],axis=1,keepdims=True))
    obs_per_season = jnp.asarray(calculate_observations_per_season_jax(incidence_jax, age_groups=True, NAG=NAG_eff))
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
    def peak_time(season_data):
        """Calculate peak time: numeric day with maximum incidence relative to EPOCH"""
        if season_data.ndim == 1:
            # Handle 1D case (Series)
            if np.sum(season_data.values) == 0:
                return np.nan
            
            # Get the timestamp of the peak
            peak_timestamp = season_data.index[np.argmax(season_data.values)]
            # Return as numeric days from EPOCH
            return (peak_timestamp - EPOCH).days
        
        else:
            # Handle multi-column case (DataFrame)
            results = []
            for col in range(season_data.shape[1]):
                col_data = season_data.iloc[:, col]
                
                if np.sum(col_data.values) == 0:
                    results.append(np.nan)
                else:
                    peak_timestamp = col_data.index[np.argmax(col_data.values)]
                    results.append((peak_timestamp - EPOCH).days)
            
            # Convert list to a standard NumPy array to match center_of_gravity
            return np.array(results)

    peak_times_by_season = incidence.groupby(incidence.index.map(get_season_start)).apply(center_of_gravity)
    print("Peak times by season:\n", peak_times_by_season)
    # peak_times_by_season is now a Series of arrays; convert to list of time indices per season
    peak_times = [pd.to_timedelta(np.asarray(pt, dtype=int), unit='D') if isinstance(pt, np.ndarray) else pd.to_timedelta(int(pt), unit='D') for pt in peak_times_by_season]
    array_of_year_starts = np.array([date_to_t(date) for date in ['2015-10-01','2016-10-01','2017-10-01','2018-10-01','2019-10-01','2020-10-01','2021-10-01','2022-10-01','2023-10-01','2024-10-01','2025-10-01']])
    peak_times_in_year = [pt - pd.to_timedelta(array_of_year_starts[i], unit='D') for i, pt in enumerate(peak_times)]
    # convert from TimedeltaIndex to numeric days
    peak_times_in_year = np.array([pt.days.to_numpy() if not isinstance(pt, np.ndarray) else np.asarray([p.days for p in pt]) for pt in peak_times_in_year])
    # convert negative values to NA
    peak_times_in_year = np.array([np.where(pt < 0, np.nan, pt) for pt in peak_times_in_year])
    def plot_peak_times_by_age_group(ax, peak_times_in_year, obs_per_season, first_age_group=0, last_age_group=7, marker='o'):
        print("Peak time of each season (in days since season start):\n", peak_times_in_year)
        if first_age_group == "<1":
            first_age_group = 0
            first_age_group_name = "<1y"
        else:
            first_age_group_name = AGE_GROUP_NAMES[first_age_group]
        # plot peak time for fist age group vs last age group in each season, color points, darker for later seasons
        for i in range(len(peak_times_in_year)-1):
            if np.mean(obs_per_season[i]) < np.mean(obs_per_season)/10:
                print(f"Skipping season {2015+i}/{2016+i} due to low incidence")
                continue
            ax.scatter(peak_times_in_year[i][first_age_group], peak_times_in_year[i][last_age_group], color=plt.cm.viridis(i/(len(peak_times_in_year)-1)), label=f"{2015+i}/{2016+i}", marker=marker)
        ax.plot([0, 200], [0, 200], color='gray', linestyle='--')
        ax.set_xlabel(f"Center of gravity in {first_age_group_name} (days)")
        ax.set_ylabel(f"Center of gravity in {AGE_GROUP_NAMES[last_age_group]} (days)")
        # ax.set_title(f"Center of gravitys by age group for {pathogen}")
        # ax.legend()
    first_age_group = 0
    if SUM_FIRST_TWO_AGE_GROUPS:
        first_age_group = "<1"
    last_age_group = NAG_eff-1
    # fig, ax = plt.subplots(figsize=(3,3))
    # plot_peak_times_by_age_group(ax, peak_times_in_year, obs_per_season, first_age_group=first_age_group, last_age_group=last_age_group)

    # print("Peak time of each season:\n", peak_times)
    # difference between 2017/18 and 2022/23 seasons
    peak_time_diff = (peak_times[7] - peak_times[2]).days - 365*5
    # this pritns in vertical format, just printa s a list
    print("Difference in peak times between 2017/18 and 2022/23 seasons (in days):", peak_time_diff.tolist())

    population_size = calculate_population_size(values, N_S=N_S, NAG=NAG)
    if (AGE_GROUPS is not None) and (max_month is not None):
        population_size = sum_age_to(population_size, max_month, AGE_GROUPS)
    expected_obs = calculate_expected_obs(values, p_time_to_obs, len(times), NAG=NAG)
    if (AGE_GROUPS is not None) and (max_month is not None):
        expected_obs = sum_age_to(expected_obs, max_month, AGE_GROUPS)
    if SUM_FIRST_TWO_AGE_GROUPS:
        expected_obs = expected_obs.at[:,:2].set(jnp.sum(expected_obs[:,:2],axis=1,keepdims=True))
        population_size = population_size.at[:2].set(jnp.sum(population_size[:2]))
    cut_times = times[:-1]
    expected_obs_per_season = calculate_observations_per_season_jax(expected_obs, age_groups=True, NAG=NAG_eff)
    # Assign each time point to a season (numeric season id/start)
    season_ids = jax.vmap(get_season_start_jax)(cut_times)
    unique_seasons = 16684 + 365 * jnp.arange(10)  # Assuming seasons start on day 259 of each year
    
    population_size_by_season = np.array([population_size[np.argmax(times >= season)] for season in unique_seasons])
    expected_obs_per_season = expected_obs_per_season / population_size_by_season


    # print("Observed per season:\n", obs_per_season)

    # # plot bar chart of obs per season
    # fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    # axes = axes.flatten()
    
    # for i_age in range(NAG_eff):
    #     ax = axes[i_age]
    #     ax.bar(unique_seasons, obs_per_season[:, i_age], width=300, alpha=0.5, label="Observed")
    #     # Resample population size to yearly (seasonal) values
    #     ax.bar(unique_seasons, expected_obs_per_season[:, i_age], width=300, alpha=0.5, label="Expected")
    #     ax.set_title(AGE_GROUP_NAMES[i_age], fontsize=9)
    #     ax.set_ylabel("Observations per season")
    #     if i_age == 0:
    #         ax.legend(frameon=False, fontsize=8)
    
    # plt.tight_layout()
    # plt.savefig("Figures/"+prefix+pathogen+"_obs_per_season_"+str(seed)+".png", dpi=300, bbox_inches='tight')
    # plt.close()
    # print("Expected per season:\n", expected_obs_per_season)

    # For each season, find the center of gravity of expected observation (per age group)
    def season_center_of_gravity(season_id):
        season_mask = season_ids == season_id                           # (T,)
        masked_obs = jnp.where(season_mask[:, None], expected_obs, 0)   # (T, NAG)
        # Calculate center of gravity: sum(time_idx * obs) / sum(obs)
        time_indices = jnp.arange(len(cut_times))
        numerator = jnp.sum(time_indices[:, None] * masked_obs, axis=0)  # (NAG,)
        denominator = jnp.sum(masked_obs, axis=0)                       # (NAG,)
        peak_idx = numerator / jnp.maximum(denominator, 1e-10)          # (NAG,) - avoid division by zero
        return times[jnp.asarray(peak_idx, dtype=int)]                  # (NAG,)

    # define an alternative season_peak_times that just finds the actual maximum time point instead of the center of gravity, to avoid issues with multiple peaks
    def season_peak_times(season_id):
        season_mask = season_ids == season_id                           # (T,)
        masked_obs = jnp.where(season_mask[:, None], expected_obs, 0)   # (T, NAG)
        peak_idx = jnp.argmax(masked_obs, axis=0)                       # (NAG,)
        return times[peak_idx]                  # (NAG,)
    
    expected_peak_times = jax.vmap(season_peak_times)(unique_seasons)
    expected_peak_times_in_year = np.array(expected_peak_times) - unique_seasons[:, None]
    # plot_peak_times_by_age_group(ax, expected_peak_times_in_year, obs_per_season, first_age_group=first_age_group, last_age_group=last_age_group, marker='x')
    # plt.savefig("Figures/"+prefix+pathogen+"_semigrav_peak_and_expected_peak_times_by_age_group_"+str(first_age_group)+"vs"+str(last_age_group)+"_"+str(seed)+".png", dpi=300, bbox_inches='tight')
    # plt.close()
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
    # threshold = 1
    # rebound_season_idx = np.where(np.sum(obs_per_season[5:], axis=1) > threshold * np.median(np.sum(obs_per_season[:5], axis=1)))[0][0]
    rebound_season_idx = 2
    rebound_observed_obs = obs_per_season[5+rebound_season_idx]

    from sim_grid import age_ratio_of_rebound

    print(age_ratio_of_rebound(jnp.stack([obs_per_season, obs_per_season], axis=0), idx_num=2, idx_den=None, season_idx=5+rebound_season_idx))
    print(age_ratio_of_rebound(jnp.stack([expected_obs_per_season, expected_obs_per_season], axis=0), idx_num=2, idx_den=None, season_idx=5+rebound_season_idx))
    
    obs_ratio_matrix = np.zeros((NAG_eff, NAG_eff))
    for i in range(NAG_eff):
        for j in range(NAG_eff):
            ratio_of_ratios = age_ratio_of_rebound(jnp.stack([obs_per_season, obs_per_season], axis=0), idx_num=i, idx_den=j, season_idx=5+rebound_season_idx)
            obs_ratio_matrix[i, j] = ratio_of_ratios

    expected_ratio_matrix = np.zeros((NAG_eff, NAG_eff))
    for i in range(NAG_eff):
        for j in range(NAG_eff):
            ratio_of_ratios = age_ratio_of_rebound(jnp.stack([expected_obs_per_season, expected_obs_per_season], axis=0), idx_num=i, idx_den=j, season_idx=5+rebound_season_idx)
            expected_ratio_matrix[i, j] = ratio_of_ratios

    
    # Calculate differences between observed and expected, weighted by distance from 1
    diff_matrix = np.abs(obs_ratio_matrix - expected_ratio_matrix)
    # Weight by distance from 1 (prioritize values far from 1)
    weight_matrix = np.abs(obs_ratio_matrix - 1) + np.abs(expected_ratio_matrix - 1)
    
    # Create weighted diff only for lower triangle
    weighted_diff_matrix = np.full_like(diff_matrix, np.nan)
    for i in range(NAG_eff):
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
    for i in range(NAG_eff):
        for j in range(NAG_eff):
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

    hsv_colors = colormaps.hsv(-0.02+np.arange(NAG_eff)/NAG_eff)
    hsv_colors[3] = colormaps.hsv((3/NAG_eff)+0.28/NAG_eff)

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


    for i_age in range(NAG_eff):
        age_ax = ax_grid[i_age // 4][i_age % 4]
        if "orig_incidence_data" in option1:
            dmx = kpsc_positive_test_plot(age_ax, pathogen, AGE_GROUPS, AGE_GROUP_NAMES, color="k", legend=False, aggregation=aggregation, factor=10000, select_age_group=i_age, orig=True, linewidth=0.5)
        elif option1 == "old_incidence_data":
            dmx = kpsc_positive_test_plot(age_ax, pathogen, AGE_GROUPS, AGE_GROUP_NAMES, color="k", legend=False, aggregation=aggregation, factor=10000, select_age_group=i_age, linewidth=0.5)
        else:
            dmx = kpsc_proportion_positive_incidence_plot(age_ax, pathogen, AGE_GROUPS, AGE_GROUP_NAMES, select_age_group=i_age, aggregation=aggregation, factor=10000, color="black", hosp=hosp, detrend=("detrend" in option1), linewidth=0.5, dedup=("dedup" in option1))
        mx = lockdown_incidence_plot(age_ax,STATE0,params,POINTS,date_to_t('2020-03-19'),label=None,by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES,solution=solution,factor=[1,7,30.44][[None,"W","MS"].index(aggregation)]*10000,p_time_to_obs=p_time_to_obs, select_age_group=i_age, color=hsv_colors[i_age], linewidth=0.5, NAG=NAG,AGE_GROUPS=AGE_GROUPS,max_month=max_month,
                                     test_data=data_full,daily_hospitalization_rates=daily_hospitalization_rates,aggregation=aggregation,
                                     )
        lockdown_incidence_format(age_ax,date_to_t('2020-03-19'),365,mx,year_window=2)
        age_ax.legend(frameon=False, fontsize=6)
    if NAG < 8:
        omx = lockdown_incidence_plot(ax_grid[-1][-1],STATE0,params,POINTS,date_to_t('2020-03-19'),solution=solution,label="Simulation",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=[1,7,30.44][[None,"W","MS"].index(aggregation)]*10000,p_time_to_obs=p_time_to_obs, color="grey", linewidth=0.5, NAG=NAG,AGE_GROUPS=AGE_GROUPS,max_month=max_month)
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

    lockdown_susceptibility_plot(ax[2],STATE0,params,PERIOD,POINTS,date_to_t('2020-03-19'),solution=solution,relative=True,proportion=True, by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES, NAG=NAG, NAG_eff=NAG_eff, AGE_GROUPS=AGE_GROUPS, max_month=max_month)
    lockdown_susceptibility_format(ax[2],date_to_t('2020-03-19'),365,year_window=2,ymax=None,ymin=None)
    # ax[3].set_title("Effective susceptibles")

    if option1 == "old_incidence_data":
        kpsc_positive_test_plot(ax[0], pathogen, None, AGE_GROUP_NAMES, aggregation=aggregation, factor=10000, color="black", label="Data")
    elif "orig_incidence_data" in option1:
        kpsc_positive_test_plot(ax[0], pathogen, None, AGE_GROUP_NAMES, aggregation=aggregation, factor=10000, color="black", label="Data", orig=True)
    else:
        kpsc_proportion_positive_incidence_plot(ax[0], pathogen, None, AGE_GROUP_NAMES, aggregation=aggregation, factor=10000, color="black", label="Data", hosp=hosp, detrend=("detrend" in option1), dedup=("dedup" in option1))
    mx = lockdown_incidence_plot(ax[0],STATE0,params,POINTS,date_to_t('2020-03-19'),solution=solution,label="Simulation",by_age=False,AGE_GROUP_NAMES=AGE_GROUP_NAMES,factor=[1,7,30.44][[None,"W","MS"].index(aggregation)]*10000,p_time_to_obs=p_time_to_obs,NAG=NAG,AGE_GROUPS=AGE_GROUPS,max_month=max_month,
                                 test_data=data_full,daily_hospitalization_rates=daily_hospitalization_rates,aggregation=aggregation,
                                 )
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
    ax[2].set_ylabel("Relative effective\nsusceptibility")

    # pathogen as title
    fig.suptitle(pnamedict[pathogen_name], fontsize=10)
    # likelihood as subtitle
    fig.text(0.5, 0.92, "Log-Likelihood: "+str(np.round(-log_likelihood*N,0)), ha='center', fontsize=8)

    # plt.tight_layout()
    plt.savefig("Figures/"+prefix+pathogen+lockdown+option1+option2_label+str(seed)+".pdf",dpi=300)
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