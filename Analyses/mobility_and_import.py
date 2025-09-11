## Code to construct a mobility time series for California from Google Mobility Data
## with optional separation between work, residential, and other locations

# import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import date_to_t, t_to_date

# # # MOBILITY
# # load mobility data and concatenate
# MOBILITY2020 = pd.read_csv('Data/Raw/Google_mobility_reports/2020_US_Region_Mobility_Report.csv', delimiter=',')
# MOBILITY2021 = pd.read_csv('Data/Raw/Google_mobility_reports/2021_US_Region_Mobility_Report.csv', delimiter=',')
# MOBILITY2022 = pd.read_csv('Data/Raw/Google_mobility_reports/2022_US_Region_Mobility_Report.csv', delimiter=',')
# MOBILITY = pd.concat([MOBILITY2020, MOBILITY2021, MOBILITY2022])

# # extract California data
# MOBILITY_CA = MOBILITY.loc[MOBILITY['iso_3166_2_code'] == 'US-CA']
# # MOBILITY_CA['date'] = pd.to_datetime(MOBILITY_CA['date'])
# MOBILITY_CA = MOBILITY_CA.sort_values(by='date')
# MOBILITY_CA.index = (pd.to_datetime(MOBILITY_CA['date'])-pd.to_datetime('1970-01-01')).dt.days


# work_mobility = 1+MOBILITY_CA['workplaces_percent_change_from_baseline']/100
# other_mobility = 1+(1/100)*(1/4)*(MOBILITY_CA['retail_and_recreation_percent_change_from_baseline'] + MOBILITY_CA['grocery_and_pharmacy_percent_change_from_baseline'] + MOBILITY_CA['parks_percent_change_from_baseline'] + MOBILITY_CA['transit_stations_percent_change_from_baseline'])

# work_contact_model = -0.2273 + 1.3280*work_mobility
# other_contact_model = -0.0488 + 1.0398*other_mobility

# work_contact_model_quadratic = 1.3169 - 4.7718*work_mobility + 5.7062*work_mobility**2
# other_contact_model_quadratic = 2.9441 - 9.2762*other_mobility + 8.5566*other_mobility**2

# # moving average of mobility
# ma_work_mobility = work_mobility.rolling(window=28).mean()
# ma_other_mobility = other_mobility.rolling(window=28).mean()
# ma_work_contacts = work_contact_model.rolling(window=28).mean()
# ma_other_contacts = other_contact_model.rolling(window=28).mean()
# ma_work_contacts_quadratic = work_contact_model_quadratic.rolling(window=28).mean()
# ma_other_contacts_quadratic = other_contact_model_quadratic.rolling(window=28).mean()

# work_quadratic_model_of_ma = 1.3169 - 4.7718*ma_work_mobility + 5.7062*ma_work_mobility**2
# other_quadratic_model_of_ma = 2.9441 - 9.2762*ma_other_mobility + 8.5566*ma_other_mobility**2

# # find the time at which ma_work_contacts_quadratic first becomes greater than 1, after January 1 2021
# t = jnp.array(ma_work_contacts_quadratic.index)
# t = t[t > date_to_t(pd.to_datetime('2021-01-01'))]
# t = t[jnp.argmax(ma_work_contacts_quadratic[t] > 0.95)]
# print(t)

# fig, ax = plt.subplots(figsize=(6,6))
# ma_work_mobility.plot(ax=ax,legend=False,color="#648FFF")
# ma_work_contacts.plot(ax=ax,legend=False,color="#DC267F",linestyle='--')
# ma_work_contacts_quadratic.plot(ax=ax,legend=False,color="#FFB000",linestyle='--')
# ax.legend(['Workplaces','Workplaces linear model','Workplaces quadratic model'])
# ax.set_xlabel('Date')
# ax.set_ylabel('Relative mobility')
# ax.set_title('California mobility')
# ax.set_ylim(0.35,1.2)
# # ax.vlines(t,0.4,1.2,linestyle=':',color='black')
# ax.hlines(1,*ax.get_xlim(),linestyle=":",color='black')
# plt.tight_layout()
# plt.savefig('Figures/mobility_CA_28day_average_work_models_test.png',dpi=300)

# # moving average of residential mobility
# MOBILITY_CA['residential_percent_change_from_baseline_ma'] = 1 - MOBILITY_CA['residential_percent_change_from_baseline'].rolling(window=7).mean()/100

# MOBILITY_CA['date'] = pd.to_datetime(MOBILITY_CA['date'])
# MOBILITY_CA.index = (MOBILITY_CA['date'] - pd.to_datetime('1970-01-01')).dt.days

# fig, ax = plt.subplots(figsize=(6,6))
# MOBILITY_CA.plot('date','residential_percent_change_from_baseline_ma',ax=ax,legend=False,color="#648FFF")
# ax.set_xlabel('Date')
# ax.set_ylabel('Relative non-residential visits')
# ax.set_title('California mobility outside of residential locations')
# plt.tight_layout()
# # plt.show()
# plt.savefig('Figures/mobility_CA_residential_7day_average.png',dpi=300)

## IMPORT
# load flights data
FLIGHTS = pd.read_csv('Data/Raw/Los_Angeles_International_Airport_-_Passenger_Traffic_By_Terminal.csv', delimiter=',')

MONTHLY_ARRIVALS = FLIGHTS.loc[FLIGHTS['Arrival_Departure'] == 'Arrival'].groupby('ReportPeriod').sum('Passenger_Count')
MONTHLY_ARRIVALS.index = pd.to_datetime(MONTHLY_ARRIVALS.index, format='%m/%d/%Y %H:%M:%S %p')
MONTHLY_ARRIVALS = MONTHLY_ARRIVALS.sort_index()
MONTHLY_ARRIVALS.index = (MONTHLY_ARRIVALS.index - pd.to_datetime('1970-01-01')).days
PASSENGER_COUNT = np.asarray(MONTHLY_ARRIVALS['Passenger_Count'])

IDX = np.array(MONTHLY_ARRIVALS.index)
MONTHLY_ARRIVALS_NP = np.array(PASSENGER_COUNT)/np.max(PASSENGER_COUNT)

# POINTS = jnp.arange(IDX[0], IDX[-1]+1, 1)

# # calculate cumulative arrivals
# cum_arrivals = jnp.cumsum(MONTHLY_ARRIVALS_NP)
# # calculate daily arrivals by interpolating the cumulative arrivals
# z_cum = jnp.interp(POINTS, IDX, cum_arrivals)
# # transform from cumulative to daily arrivals
# z = jnp.diff(z_cum, prepend=0)

# print(IDX[1:]-IDX[:-1])
# rep_idx = jnp.repeat(IDX[1:]-IDX[:-1],IDX[1:]-IDX[:-1])
# # append 30 to end
# IDX = jnp.append(IDX, IDX[-1]+30)
# z = jnp.interp(POINTS, IDX[:-1], MONTHLY_ARRIVALS_NP/(IDX[1:]-IDX[:-1]))
# plt.plot(POINTS, z, color='r', label='Daily arrivals')
# plt.plot(IDX[:-1], MONTHLY_ARRIVALS_NP/(IDX[1:]-IDX[:-1]), color='b', alpha=0.5, label='Monthly arrivals')
# plt.show()
# gap = IDX[1] - IDX[0]
# zmooth = jnp.convolve(z, jnp.ones(gap)/gap, mode='valid')
# plt.plot(POINTS[gap//2:-(gap//2)], zmooth, color='r', label='Daily arrivals')
# plt.plot(IDX, MONTHLY_ARRIVALS_NP/gap, color='b', alpha=0.5, label='Monthly arrivals')
# plt.show()

# # show that the sum of z over each month is equal to the monthly arrivals
# x = jnp.zeros(len(IDX)-1)
# x2 = jnp.zeros(len(IDX)-1)
# for i in range(len(IDX)-1):
#     x = x.at[i].set(jnp.sum(z[(POINTS >= IDX[i]) & (POINTS < IDX[i+1])]))
#     x2 = x2.at[i].set(jnp.sum(zmooth[(POINTS[gap//2:-(gap//2)] >= IDX[i]) & (POINTS[gap//2:-(gap//2)] < IDX[i+1])]))
# plt.scatter(MONTHLY_ARRIVALS_NP[:-1], x2, color='b', alpha=0.5, label='Monthly arrivals')
# plt.scatter(MONTHLY_ARRIVALS_NP[:-1], x, color='r', alpha=0.5, label='Monthly arrivals')
# plt.plot(MONTHLY_ARRIVALS_NP[:-1],MONTHLY_ARRIVALS_NP[:-1], color='k', linestyle='--', label='Ideal')
# plt.show()



# z = jnp.interp(POINTS, IDX, MONTHLY_ARRIVALS_NP)/30.44
# # # smooth z
# gap = IDX[1] - IDX[0]
# # gap = 7
# zmooth = jnp.convolve(z, jnp.ones(gap)/gap, mode='valid')
# print(jnp.sum(z))
# print(jnp.sum(zmooth))
# plt.plot(POINTS, z, color='r')
# plt.plot(POINTS[gap//2:-(gap//2)], zmooth, color='k')
# plt.show()

# @jit
def arrivals(t, MONTHLY_ARRIVALS_NP=MONTHLY_ARRIVALS_NP, IDX=IDX):
    """
    Return daily of arrivals at time t, with t the number of days since 1970-01-01
    """
    # Calculate all possible values
    day_in_season_2006 = (t - 13149) % 365
    time_2006 = 13149 + day_in_season_2006
    arrivals_2006 = MONTHLY_ARRIVALS_NP[np.argmax(IDX >= time_2006)] / 30.44
    
    day_in_season_2022 = (t - 19266) % 365
    time_2022 = 19266 + day_in_season_2022
    arrivals_2022 = MONTHLY_ARRIVALS_NP[np.argmax(IDX >= time_2022)] / 30.44
    
    arrivals_normal = MONTHLY_ARRIVALS_NP[np.argmax(IDX >= t)] / 30.44
    
    # Use nested np.where to select the appropriate value
    return np.where(
        t < 13149,
        arrivals_2006,
        np.where(
            t > 19266,
            arrivals_2022,
            arrivals_normal
        )
    )

# arrivals_by_month = jnp.array([arrivals(t) for t in jnp.array(date_to_t(pd.date_range(start=pd.to_datetime('2016-01-01'), end=pd.to_datetime('2024-01-01'), freq='MS')))])
# plt.plot(arrivals_by_month)
# plt.show()

# plt.rcParams.update({'font.size':20})
# # text type is palatino
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Palatino']

# fig, ax = plt.subplots(2,1,figsize=(7,5),sharex=True)
# MONTHLY_ARRIVALS.plot(ax=ax[0],legend=False,color="k")
# ax[0].set_ylabel('')
# ax[0].set_xlabel('')
# ax[0].set_title('Monthly arrivals at LAX')
# ax[0].set_xlim([date_to_t("2015-10-01"),date_to_t("2024-01-01")])
# ax[0].set_xticks([date_to_t("2016-01-01"),date_to_t("2018-01-01"),date_to_t("2020-01-01"),date_to_t("2022-01-01"),date_to_t("2024-01-01")])
# ax[0].set_xticklabels(["2016","2018","2020","2022","2024"])
# import contact_model as cm
# Ts = jnp.array([date_to_t(date) for date in ['1970-01-01', '2020-03-19', '2020-08-28', '2021-08-28', '2022-04-16']])
# Fs = jnp.array([1,0.74649061,0.97616163,0.84107913,0.97878123])
# x = jnp.linspace(date_to_t('2015-10-01'),date_to_t('2024-01-01'),1000)
# y = [cm.piecewise(t,Ts,Fs) for t in x]
# ax[1].plot(x,y,color="k")
# plt.tight_layout()
# ax[1].set_title('Fit contact model')
# plt.savefig('Figures/arrivals_LAX_poster.svg',transparent=True)