## January 2026
## Converting new vaccination data into rates

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotting import hsv_colors
from Parameters.census_population import AGE_GROUPS, AGE_GROUP_NAMES, POP_SIZE

############## Calculating effective vaccination rates ###############


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
# # sum rsv_immune_fl and nirsevimab columns to get total protected
# summed_demographics = demographics.copy()
# summed_demographics.loc[:,"rsv_immune_fl"] = (summed_demographics["rsv_immune_fl"].astype(bool) | summed_demographics["nirsevimab"].astype(bool)).astype(int)
# # pivot table to get number vaccinated each month by age category
# vax_by_month_age = pd.pivot_table(summed_demographics, values="n_count", index=["month_start", "age_category"], columns="rsv_immune_fl", aggfunc="sum")
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

# ##### RATE OF VACCINATION IN PREGNANT WOMEN #####
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