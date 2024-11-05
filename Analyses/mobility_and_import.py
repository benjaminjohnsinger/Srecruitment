## Code to construct a mobility time series for California from Google Mobility Data
## with optional separation between work, residential, and other locations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# MOBILITY
# load mobility data and concatenate
MOBILITY2020 = pd.read_csv('Data/Raw/Google_mobility_reports/2020_US_Region_Mobility_Report.csv', delimiter=',')
MOBILITY2021 = pd.read_csv('Data/Raw/Google_mobility_reports/2021_US_Region_Mobility_Report.csv', delimiter=',')
MOBILITY2022 = pd.read_csv('Data/Raw/Google_mobility_reports/2022_US_Region_Mobility_Report.csv', delimiter=',')
MOBILITY = pd.concat([MOBILITY2020, MOBILITY2021, MOBILITY2022])

# extract California data
MOBILITY_CA = MOBILITY.loc[MOBILITY['iso_3166_2_code'] == 'US-CA']

# moving average of residential mobility
MOBILITY_CA['residential_percent_change_from_baseline_ma'] = 1 - MOBILITY_CA['residential_percent_change_from_baseline'].rolling(window=7).mean()/100

fig, ax = plt.subplots(figsize=(6,6))
MOBILITY_CA.plot('date','residential_percent_change_from_baseline_ma',ax=ax,legend=False,color="#648FFF")
ax.set_xlabel('Date')
ax.set_ylabel('Relative non-residential visits')
ax.set_title('California mobility outside of residential locations')
plt.tight_layout()
plt.savefig('Figures/mobility_CA_residential_7day_average.png',dpi=300)

## IMPORT
# load flights data
FLIGHTS = pd.read_csv('Data/Raw/Los_Angeles_International_Airport_-_Passenger_Traffic_By_Terminal.csv', delimiter=',')

# same but only for 'Arrival_Departure' euql 'Arrival'
MONTHLY_ARRIVALS = FLIGHTS.loc[FLIGHTS['Arrival_Departure'] == 'Arrival'].groupby('ReportPeriod').sum('Passenger_Count')
MONTHLY_ARRIVALS.index = pd.to_datetime(MONTHLY_ARRIVALS.index, format='%m/%d/%Y %H:%M:%S %p')
MONTHLY_ARRIVALS = MONTHLY_ARRIVALS.sort_index()

fig, ax = plt.subplots(figsize=(6,6))
MONTHLY_ARRIVALS.plot(ax=ax,legend=False,color="#648FFF")
ax.set_ylabel('Monthly arrivals')
ax.set_xlabel('Date')
ax.set_title('Monthly arrivals at LAX')
plt.tight_layout()
plt.savefig('Figures/arrivals_LAX.png',dpi=300)