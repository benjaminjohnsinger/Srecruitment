## January 2026
## Converting new vaccination data into rates

import jax
import numpy as np
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from plotting import hsv_colors
from Parameters.census_population import AGE_GROUPS, AGE_GROUP_NAMES, POP_SIZE

age_category_names = ['<3mo','3-12mo','1-4y','5-17y','18-39y','40-64y','>=65y']

############## Calculating effective vaccination rates ###############
def get_jax_arrays_from_vax_csv(file, age_group_names=age_category_names):
    vax_df = pd.read_csv(file)
    vax_df['month_start'] = pd.to_datetime(vax_df['month_start'], format='%Y-%m-%d')
    vax_df.loc[:,'time'] = (vax_df['month_start'] - pd.to_datetime('1970-01-01')).dt.days
    if age_group_names is not None:
        vax_pivot = vax_df.pivot(index='time', columns='age_group', values='rate')
        # Reorder columns to match age_group_names order
        vax_pivot = vax_pivot.reindex(columns=age_group_names)
    else:
        vax_pivot = vax_df[["time", "cumulative_proportion_vaccinated"]].set_index("time")
    vax_idx = jnp.array(vax_pivot.index)
    vax_np = jnp.array(vax_pivot.values)
    return vax_idx, vax_np

#### Influenza vax rates #####
# ratio of vaccination in the general population to vaccination in KPSC patients
KPSC_RATIO = 0.455 / 0.5793469238813651
# flu vaccine efficacy by year
EFF = (1/100) * jnp.array([37, 61, 51, 44, 39, 53, 7, 52, 19, 33, 25, 34, 37, 32, 23, 35, 37, 37]) # VE in 18-49yo (or age group containing this range) from studies that went into Data/Raw/vaccine-effectiveness-chart-2024.xlsx, with missing data filled in with mean (37)
EFF_IDX = jnp.array([(pd.to_datetime('2008-10-01') + pd.DateOffset(years=i) - pd.to_datetime('1970-01-01')).days for i in range(18)])
# rate of vaccination in the KPSC population by month and age group
VAX_RATE_IDX, VAX_RATE_NP = get_jax_arrays_from_vax_csv('Data/Processed/Influenza_vaccination_rates_by_month_and_age_group.csv')
# adjust for KPSC ratio
VAX_RATE_NP = VAX_RATE_NP * KPSC_RATIO
# calculate effective rate for an array of time points given a maximum efficacy, with seasons before 2015 using VAX_RATE data from 2015-2016 and seasons after 2025 using data from 2024-2025
@jax.jit
def flu_eff_vax_rate(t_arr, max_eff):
    # Vectorized lookup for raw efficacy using searchsorted
    # Find the index of the season for each time point in t_arr
    eff_indices = jnp.searchsorted(EFF_IDX, t_arr, side='right')
    raw_eff = EFF[jnp.maximum(0, eff_indices)] # Use maximum to handle t before first season

    adj_eff = raw_eff / max_eff
    adj_eff = jnp.minimum(adj_eff, 1.0)

    # Define time boundaries
    time_2015_start = 16709  # Days from 1970-01-01 to 2015-10-01
    time_2024_end = 19997    # Days from 1970-01-01 to 2025-10-01

    # --- Logic for times before the main data range ---
    day_in_season_pre = (t_arr - time_2015_start) % 365
    time_in_2015_season = time_2015_start + day_in_season_pre
    vax_indices_pre = jnp.searchsorted(VAX_RATE_IDX, time_in_2015_season, side='right')
    rate_pre = adj_eff[:, None] * VAX_RATE_NP[vax_indices_pre]

    # --- Logic for times after the main data range ---
    day_in_season_post = (t_arr - time_2024_end) % 365
    time_in_2024_season = time_2024_end + day_in_season_post
    vax_indices_post = jnp.searchsorted(VAX_RATE_IDX, time_in_2024_season, side='right')
    rate_post = adj_eff[:, None] * VAX_RATE_NP[vax_indices_post]

    # --- Logic for times within the main data range ---
    vax_indices_mid = jnp.searchsorted(VAX_RATE_IDX, t_arr, side='right')
    rate_mid = adj_eff[:, None] * VAX_RATE_NP[vax_indices_mid]

    # Combine results based on time conditions
    rate = jnp.where(t_arr[:, None] < time_2015_start, rate_pre,
                     jnp.where(t_arr[:, None] > time_2024_end, rate_post, rate_mid))
    return rate

#### RSV vax rates #####
RSV_VAX_RATE_IDX, RSV_VAX_RATE_NP = get_jax_arrays_from_vax_csv('Data/Processed/RSV_vaccination_rates_by_month_and_age_group.csv')
RSV_VAX_EFF = 0.73 # CDC website
# nirsevimab rates
NIRSEVIMAB_RATE_IDX, NIRSEVIMAB_RATE_NP = get_jax_arrays_from_vax_csv('Data/Processed/Nirsevimab_rates_by_month_and_age_group.csv')
NIRSEVIMAB_EFF = 0.98 # Hsiao et al., Pediatrics 2025
@jax.jit
def rsv_eff_vax_rate(t_arr, max_eff0, max_eff1):
    # RSV vaccination rate
    vax_indices = jnp.searchsorted(RSV_VAX_RATE_IDX, t_arr, side='right')
    rate_vax = RSV_VAX_RATE_NP[vax_indices]
    # Nirsevimab administration rate
    nirsev_indices = jnp.searchsorted(NIRSEVIMAB_RATE_IDX, t_arr, side='right')
    rate_nirsev = NIRSEVIMAB_RATE_NP[nirsev_indices]
    # adjusted efficacy
    adj_eff_vax = jnp.maximum(RSV_VAX_EFF / max_eff1, 1.0)
    adj_eff_nirsev = jnp.maximum(NIRSEVIMAB_EFF / max_eff0, 1.0)
    # Effective rates
    eff_rate_vax = rate_vax * adj_eff_vax
    eff_rate_nirsev = rate_nirsev * adj_eff_nirsev
    total_rate = eff_rate_vax + eff_rate_nirsev
    return total_rate

#### RSV maternal immunity rates #####
MATERNAL_PVAX_IDX, MATERNAL_PVAX_NP = get_jax_arrays_from_vax_csv('Data/Processed/RSV_pregnant_vaccination_by_month.csv', age_group_names=None)
MATERNAL_VAX_EFF = 0.82 # Kampmann et al., NEJM 2023
@jax.jit
def rsv_maternal_immunity(t_arr):
    vax_indices = jnp.searchsorted(MATERNAL_PVAX_IDX, t_arr, side='right')
    p = MATERNAL_PVAX_NP[vax_indices].squeeze()
    eff_p = p * MATERNAL_VAX_EFF
    return eff_p

if __name__ == "__main__":
    # # Test the function and plot results
    FULL_POINTS = jnp.arange(0, 22000)  # Example time points
    # S_REL = jnp.array([1, 0.3, 0.3*0.25])
    # P_OBS = jnp.array([1, 0.46, 0.31])
    # protection_param = S_REL * P_OBS
    # max_eff0 = 1-protection_param[-1]
    # max_eff1 = (protection_param[-2]-protection_param[-1])/protection_param[-2]
    # rates = rsv_eff_vax_rate(FULL_POINTS, max_eff0, max_eff1)
    mat_imm = rsv_maternal_immunity(FULL_POINTS)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Convert time points to dates
    dates = pd.to_datetime(FULL_POINTS, unit='D', origin='1970-01-01')
    ax.plot(dates, mat_imm)
    ax.set_title("RSV Maternal Immunity")
    ax.set_ylabel("Proportion with Maternal Immunity")
    ax.set_xlabel("Date")
    ax.set_xlim(pd.to_datetime('2015-01-01'), pd.to_datetime('2026-01-01'))
    plt.tight_layout()
    plt.savefig("Figures/RSV_effective_maternal_immunity.png")

    # S_REL = jnp.array([1, 0.51, 0.21])
    # P_OBS = jnp.array([1, 0.51, 0.21])
    # protection_param = S_REL * P_OBS
    # max_eff_test = (protection_param[-2]-protection_param[-1])/protection_param[-2]
    # rates = flu_eff_vax_rate(FULL_POINTS, max_eff_test)
    # fig, ax = plt.subplots(figsize=(10, 6))
    
    # # Convert time points to dates
    # dates = pd.to_datetime(FULL_POINTS, unit='D', origin='1970-01-01')
    
    # for i, age_group in enumerate(AGE_GROUP_NAMES):
    #     ax.plot(dates, rates[:, i], label=age_group, color=hsv_colors[i])
    # ax.set_title("Effective Influenza Vaccination Rates by Age Group")
    # ax.set_ylabel("Vaccination Rate Per Day")
    # ax.set_xlabel("Date")
    # ax.set_xlim(pd.to_datetime('2015-01-01'), pd.to_datetime('2026-01-01'))
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig("Figures/Influenza_effective_vaccination_rates_by_age_test.png")

# ############## Processing vaccine/demographic data ###############
# demographics = pd.read_sas("Data/Raw/KPSC/demographics_20251118.sas7bdat", format="sas7bdat", encoding="utf-8")

# age_category_names = ['<3mo','3-12mo','1-4y','5-17y','18-39y','40-64y','>=65y']
# age_categories = [
#     np.arange(0,3),        # <3 months
#     np.arange(3,13),       # 3-12 months
#     np.arange(13,5*12),    # 1-4 years
#     np.arange(5*12,18*12), # 5-17 years
#     np.arange(18*12,40*12),# 18-39 years
#     np.arange(40*12,65*12),# 40-64 years
#     np.arange(65*12,100*12)# >=65 years
# ]

# # Adjust for turnover in youngest age groups due to births and aging
# # This ignores immunity carrying over as groups age, which is valid because infants don't get vaccinated for flu, nirsevimab immunity is short-lived, and the actual RSV vaccines are mostly for older adults and have only been approved recently.
# def adjust_cumulative_for_turnover(group):
#     age_cat = group['age_category'].iloc[0]
#     turnover_rate = 1/len(age_categories[age_category_names.index(age_cat)]) # turnover rate per month
#     cumulative_prop = 0
#     cumulative_props = []
#     for _, row in group.iterrows():
#         # Decay previous cumulative proportion due to turnover
#         cumulative_prop *= (1 - turnover_rate)
#         # Add new vaccinations this month
#         cumulative_prop += row['proportion_vaccinated']
#         cumulative_props.append(cumulative_prop)
#     return pd.Series(cumulative_props, index=group.index)

# ##### RSV VACCINATION RATES #####
# # pivot table to get number vaccinated each month by age category
# vax_by_month_age = pd.pivot_table(demographics, values="n_count", index=["month_start", "age_category"], columns="rsv_immune_fl", aggfunc="sum")
# # set nans to zero
# vax_by_month_age = vax_by_month_age.fillna(0)
# # calculate proportion vaccinated each month by age category
# vax_by_month_age.loc[:,"proportion_vaccinated"] = vax_by_month_age[1] / (vax_by_month_age[0] + vax_by_month_age[1])
# # reset index to make plotting easier
# vax_by_month_age = vax_by_month_age.reset_index()
# vax_by_month_age.loc[:,"month_start"] = pd.to_datetime(vax_by_month_age["month_start"], format="%Y-%m-%d")
# # Calculate the true vaccination rate for susceptibles
# vax_rate_by_month_age = vax_by_month_age.copy()
# vax_rate_by_month_age.loc[:,"days_in_month"] = vax_rate_by_month_age["month_start"].dt.daysinmonth
# # for RSV you only get vaccinated once, so sum the proportion vaccinated over time to get the cumulative proportion vaccinated
# # vax_rate_by_month_age.loc[:,"cumulative_proportion_vaccinated"] = vax_rate_by_month_age.groupby("age_category")["proportion_vaccinated"].cumsum()
# cumulative_result = vax_rate_by_month_age.groupby("age_category").apply(adjust_cumulative_for_turnover)
# vax_rate_by_month_age.loc[:,"cumulative_proportion_vaccinated"] = cumulative_result.droplevel(0).reindex(vax_rate_by_month_age.index)
# # adjusted proportion vaccinated to account for only vaccinating susceptibles
# vax_rate_by_month_age.loc[:,"adjusted_proportion_vaccinated"] = vax_rate_by_month_age["proportion_vaccinated"] / (1 - vax_rate_by_month_age["cumulative_proportion_vaccinated"] + vax_rate_by_month_age["proportion_vaccinated"])
# # calculate vaccination rate per day
# vax_rate_by_month_age.loc[:,"vaccination_rate_per_day"] = -np.log(1 - vax_rate_by_month_age["adjusted_proportion_vaccinated"]) / vax_rate_by_month_age["days_in_month"]
# # save vacination rate to csv
# # filter to just month_start, age_category, rate
# rate = vax_rate_by_month_age[["month_start", "age_category", "vaccination_rate_per_day"]]
# # change column names
# rate.columns = ["month_start", "age_group", "rate"]
# rate.to_csv("Data/Processed/RSV_vaccination_rates_by_month_and_age_group.csv")
# # plot proportion vaccinated by age category
# fig, ax = plt.subplots(figsize=(10, 6))
# for i, age_cat in enumerate(age_category_names):
#     data = vax_rate_by_month_age[vax_rate_by_month_age["age_category"] == age_cat]
#     print(data[1].sum())
#     ax.plot(data["month_start"], data["vaccination_rate_per_day"], label=age_cat, color=hsv_colors[i])
# ax.set_title("Rate of Vaccination for RSV Each Month by Age Category")
# ax.set_ylabel("Vaccination Rate Per Day")
# ax.set_xlabel("Month")
# ax.legend()
# plt.savefig("Figures/RSV_vaccination_rates_by_age.png")
# plt.close()

# ##### RATE OF NIRSEVIMAB ADMINISTRATION #####
# # calculate proportion protected by age category over time
# nirsev_by_month_age = pd.pivot_table(demographics, values="n_count", index=["month_start", "age_category"], columns="nirsevimab", aggfunc="sum")
# # set nans to zero
# nirsev_by_month_age = nirsev_by_month_age.fillna(0)
# # calculate proportion protected each month by age category
# nirsev_by_month_age.loc[:,"proportion_vaccinated"] = nirsev_by_month_age[1] / (nirsev_by_month_age[0] + nirsev_by_month_age[1])
# # reset index to make plotting easier
# nirsev_by_month_age = nirsev_by_month_age.reset_index()
# nirsev_by_month_age.loc[:,"month_start"] = pd.to_datetime(nirsev_by_month_age["month_start"], format="%Y-%m-%d")
# # calculate rate of protection given that individuals only get immunized once per season
# nirsev_by_month_age.loc[:,"days_in_month"] = nirsev_by_month_age["month_start"].dt.daysinmonth
# # for nirsevimab you only get immunized once, so sum the proportion protected over time to get the cumulative proportion protected
# cumulative_result = nirsev_by_month_age.groupby("age_category").apply(adjust_cumulative_for_turnover)
# nirsev_by_month_age.loc[:,"cumulative_proportion_vaccinated"] = cumulative_result.droplevel(0).reindex(nirsev_by_month_age.index)
# # adjusted proportion protected to account for only protecting susceptibles
# nirsev_by_month_age.loc[:,"adjusted_proportion_vaccinated"] = nirsev_by_month_age["proportion_vaccinated"] / (1 - nirsev_by_month_age["cumulative_proportion_vaccinated"] + nirsev_by_month_age["proportion_vaccinated"])
# nirsev_by_month_age.loc[:,"nirsev_rate_per_day"] = -np.log(1 - nirsev_by_month_age["adjusted_proportion_vaccinated"]) / nirsev_by_month_age["days_in_month"]
# # save rate to csv
# nirsev_rate = nirsev_by_month_age[["month_start", "age_category", "nirsev_rate_per_day"]]
# nirsev_rate.columns = ["month_start", "age_group", "rate"]
# nirsev_rate.to_csv("Data/Processed/Nirsevimab_rates_by_month_and_age_group.csv")
# # plot proportion protected by age category
# fig, ax = plt.subplots(figsize=(10, 6))
# for i, age_cat in enumerate(age_category_names):
#     data = nirsev_by_month_age[nirsev_by_month_age["age_category"] == age_cat]
#     ax.plot(data["month_start"], data["nirsev_rate_per_day"], label=age_cat, color=hsv_colors[i])
# ax.set_title("Rate of Nirsevimab Administration Each Month by Age Category")
# ax.set_ylabel("Rate of Administration Per Day")
# ax.set_xlabel("Month")
# ax.legend()
# plt.savefig("Figures/Nirsevimab_rate_of_administration_by_age.png")
# plt.close()

# ##### PROPORTION OF VACCINATION IN PREGNANT WOMEN #####
# # filter to pregnant women only
# preg_demographics = demographics[demographics["preg_fl"] == 1]
# # pivot table to get number of pregnant women vaccinated each month
# pregvax_by_month = pd.pivot_table(preg_demographics, values="n_count", index=["month_start"], columns="rsv_immune_fl", aggfunc="sum")
# print(pregvax_by_month)
# # set nans to zero
# pregvax_by_month = pregvax_by_month.fillna(0)
# # calculate proportion vaccinated each month
# pregvax_by_month.loc[:,"proportion_vaccinated"] = pregvax_by_month[1] / (pregvax_by_month[0] + pregvax_by_month[1])
# # reset index to make plotting easier
# pregvax_by_month = pregvax_by_month.reset_index()
# pregvax_by_month.loc[:,"month_start"] = pd.to_datetime(pregvax_by_month["month_start"], format="%Y-%m-%d")
# # cumulate over nine month window
# pregvax_by_month.loc[:,"cumulative_proportion_vaccinated"] = pregvax_by_month["proportion_vaccinated"].rolling(window=9, min_periods=1).sum()
# # save vaccination rate to csv
# pregvax_rate = pregvax_by_month[["month_start", "cumulative_proportion_vaccinated"]]
# pregvax_rate.columns = ["month_start", "cumulative_proportion_vaccinated"]
# pregvax_rate.to_csv("Data/Processed/RSV_pregnant_vaccination_by_month.csv")
# # plot vaccination rate over time
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(pregvax_by_month["month_start"], pregvax_by_month["cumulative_proportion_vaccinated"], label="Vaccinated proportion", color="blue")
# ax.set_title("Proportion of Pregnant Women Vaccinated for RSV Each Month")
# ax.set_ylabel("Proportion Vaccinated")
# ax.set_xlabel("Month")
# ax.legend()
# plt.savefig("Figures/RSV_pregnant_vaccination_over_time.png")
# plt.close()

# ##### INFLUENZA VACCINATION RATES #####
# # similar process for influenza vaccination rates
# vax_by_month_age_flu = pd.pivot_table(demographics, values="n_count", index=["month_start", "age_category"], columns="flu_immune_fl", aggfunc="sum")
# # set nans to zero
# vax_by_month_age_flu = vax_by_month_age_flu.fillna(0)
# # calculate proportion vaccinated each month by age category
# vax_by_month_age_flu.loc[:,"proportion_vaccinated"] = vax_by_month_age_flu[1] / (vax_by_month_age_flu[0] + vax_by_month_age_flu[1])
# # reset index to make plotting easier
# vax_by_month_age_flu = vax_by_month_age_flu.reset_index()
# vax_by_month_age_flu.loc[:,"month_start"] = pd.to_datetime(vax_by_month_age_flu["month_start"], format="%Y-%m-%d")
# # Calculate the true vaccination rate for susceptibles
# vax_rate_by_month_age_flu = vax_by_month_age_flu.copy()
# vax_rate_by_month_age_flu.loc[:,"days_in_month"] = vax_rate_by_month_age_flu["month_start"].dt.daysinmonth
# # for influenza you only get vaccinated once per season, so sum the proportion vaccinated over each season to get the cumulative proportion vaccinated. new vaccines are released in august
# vax_rate_by_month_age_flu.loc[:,"season"] = vax_rate_by_month_age_flu["month_start"].dt.year
# vax_rate_by_month_age_flu.loc[vax_rate_by_month_age_flu["month_start"].dt.month < 8, "season"] -= 1

# cumulative_result = vax_rate_by_month_age_flu.groupby(["age_category", "season"]).apply(adjust_cumulative_for_turnover)
# vax_rate_by_month_age_flu.loc[:,"cumulative_proportion_vaccinated"] = cumulative_result.droplevel([0,1]).reindex(vax_rate_by_month_age_flu.index)
# print(vax_rate_by_month_age_flu[(vax_rate_by_month_age_flu["age_category"] == "1-4y") & (vax_rate_by_month_age_flu["season"] == 2016)])
# # plot to check cumulative proportions look reasonable
# fig, ax = plt.subplots(figsize=(10, 6))
# for i, age_cat in enumerate(age_category_names):
#     data = vax_rate_by_month_age_flu[vax_rate_by_month_age_flu["age_category"] == age_cat]
#     ax.plot(data["month_start"], data["cumulative_proportion_vaccinated"], label=age_cat, color=hsv_colors[i])
# ax.set_title("Cumulative Proportion Vaccinated for Influenza Each Month by Age Category")
# ax.set_ylabel("Cumulative Proportion Vaccinated")
# ax.set_xlabel("Month")
# ax.legend()
# plt.savefig("Figures/Influenza_cumulative_proportion_vaccinated_by_age.png")
# plt.close()
# # adjusted proportion vaccinated to account for only vaccinating susceptibles
# vax_rate_by_month_age_flu.loc[:,"adjusted_proportion_vaccinated"] = vax_rate_by_month_age_flu["proportion_vaccinated"] / (1 - vax_rate_by_month_age_flu["cumulative_proportion_vaccinated"] + vax_rate_by_month_age_flu["proportion_vaccinated"])
# # calculate vaccination rate per day
# vax_rate_by_month_age_flu.loc[:,"vaccination_rate_per_day"] = -np.log(1 - vax_rate_by_month_age_flu["adjusted_proportion_vaccinated"]) / vax_rate_by_month_age_flu["days_in_month"]
# # save vacination rate to csv
# # filter to just month_start, age_category, rate
# rate_flu = vax_rate_by_month_age_flu[["month_start", "age_category", "vaccination_rate_per_day"]]
# # change column names
# rate_flu.columns = ["month_start", "age_group", "rate"]
# rate_flu.to_csv("Data/Processed/Influenza_vaccination_rates_by_month_and_age_group.csv")
# # plot proportion vaccinated by age category
# fig, ax = plt.subplots(figsize=(10, 6))
# for i, age_cat in enumerate(age_category_names):
#     data = vax_rate_by_month_age_flu[vax_rate_by_month_age_flu["age_category"] == age_cat]
#     print(data[1].sum())
#     ax.plot(data["month_start"], data["vaccination_rate_per_day"], label=age_cat, color=hsv_colors[i])
# ax.set_title("Rate of Vaccination for Influenza Each Month by Age Category")
# ax.set_ylabel("Vaccination Rate Per Day")
# ax.set_xlabel("Month")
# ax.legend()
# plt.savefig("Figures/Influenza_vaccination_rates_by_age.png")
# plt.close()