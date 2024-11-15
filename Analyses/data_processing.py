import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sas7bdat import SAS7BDAT

# with SAS7BDAT('Data/Raw/KPSC/clinical.sas7bdat') as f:
#     clinical_data = f.to_data_frame()

# load clinical data sample
# clinical_data = pd.read_csv('Data/Processed/KPSC_clinical_sample.csv')
# print(clinical_data.head())

# date is 1st of October of each year (in YEAR column), plus dx_days
# clinical_data["Date"] = pd.to_datetime(clinical_data["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(clinical_data["dx_days"],unit='D')

# hospitalizations = clinical_data[(clinical_data["setting"] == 'Hospital admission')]

# save hostpitalizations to csv
# hospitalizations.to_csv('Data/Processed/KPSC_clinical_hospitalizations.csv',index=False)

# load hospitalizations
hospitalizations = pd.read_csv('Data/Processed/KPSC_clinical_hospitalizations.csv')
hospitalizations["Date"] = pd.to_datetime(hospitalizations["Date"])

hospitalizations["Month"] = hospitalizations["Date"].dt.month
hospitalizations["Year"] = hospitalizations["Date"].dt.year

# get only influenza hospitalizations, i.e. CODE starts with J09, J10, or J11
flu_hospitalizations = hospitalizations[hospitalizations["CODE"].str.contains('J09|J10|J11')]
# get only RSV hospitalizations, i.e. CODE is J12.1, J21.0, or B97.4 as used in Pitzer et al. 2015
rsv_hospitalizations = hospitalizations[hospitalizations["CODE"].str.contains('J12.1|J21.0|B97.4')]
# get only COVID-19 hospitalizations, i.e. CODE is U07.1
covid_hospitalizations = hospitalizations[hospitalizations["CODE"].str.contains('U07.1')]

# # filter out just one row for each value of StudyID
flu_hospitalizations = flu_hospitalizations.drop_duplicates(subset="StudyID")
rsv_hospitalizations = rsv_hospitalizations.drop_duplicates(subset="StudyID")
covid_hospitalizations = covid_hospitalizations.drop_duplicates(subset="StudyID")

monthly_flu_hospitalizations = flu_hospitalizations.groupby(["Year","Month"]).size().reset_index(name='Count')
monthly_rsv_hospitalizations = rsv_hospitalizations.groupby(["Year","Month"]).size().reset_index(name='Count')
monthly_covid_hospitalizations = covid_hospitalizations.groupby(["Year","Month"]).size().reset_index(name='Count')

# fill in zeros for missing months
for date in pd.date_range(start='2015-10-01',end='2023-09-30',freq='MS'):
    year = date.year
    month = date.month
    if not ((monthly_flu_hospitalizations["Year"] == year) & (monthly_flu_hospitalizations["Month"] == month)).any():
        monthly_flu_hospitalizations = pd.concat([monthly_flu_hospitalizations,pd.DataFrame({"Year":[year],"Month":[month],"Count":[0]})],ignore_index=True)
    if not ((monthly_rsv_hospitalizations["Year"] == year) & (monthly_rsv_hospitalizations["Month"] == month)).any():
        monthly_rsv_hospitalizations = pd.concat([monthly_rsv_hospitalizations,pd.DataFrame({"Year":[year],"Month":[month],"Count":[0]})],ignore_index=True)
    if not ((monthly_covid_hospitalizations["Year"] == year) & (monthly_covid_hospitalizations["Month"] == month)).any():
        monthly_covid_hospitalizations = pd.concat([monthly_covid_hospitalizations,pd.DataFrame({"Year":[year],"Month":[month],"Count":[0]})],ignore_index=True)

monthly_flu_hospitalizations["Date"] = pd.to_datetime(monthly_flu_hospitalizations["Year"].astype(str) + '-' + monthly_flu_hospitalizations["Month"].astype(str) + '-01')
monthly_rsv_hospitalizations["Date"] = pd.to_datetime(monthly_rsv_hospitalizations["Year"].astype(str) + '-' + monthly_rsv_hospitalizations["Month"].astype(str) + '-01')
monthly_covid_hospitalizations["Date"] = pd.to_datetime(monthly_covid_hospitalizations["Year"].astype(str) + '-' + monthly_covid_hospitalizations["Month"].astype(str) + '-01')

# sort by date
monthly_flu_hospitalizations = monthly_flu_hospitalizations.sort_values(by="Date")
monthly_rsv_hospitalizations = monthly_rsv_hospitalizations.sort_values(by="Date")
monthly_covid_hospitalizations = monthly_covid_hospitalizations.sort_values(by="Date")

# # save to csv
monthly_flu_hospitalizations.to_csv('Data/Processed/KPSC_flu_hosp.csv',index=False)
monthly_rsv_hospitalizations.to_csv('Data/Processed/KPSC_rsv_hosp.csv',index=False)
monthly_covid_hospitalizations.to_csv('Data/Processed/KPSC_covid_hosp.csv',index=False)

# monthly_flu_hospitalizations["Relative"] = monthly_flu_hospitalizations["Count"]/monthly_flu_hospitalizations["Count"].max()
# monthly_rsv_hospitalizations["Relative"] = monthly_rsv_hospitalizations["Count"]/monthly_rsv_hospitalizations["Count"].max()
# monthly_covid_hospitalizations["Relative"] = monthly_covid_hospitalizations["Count"]/monthly_covid_hospitalizations["Count"].max()

# fig, ax = plt.subplots(2,1,figsize=(6.5,6.5),sharex=True)
# ax[0].plot(monthly_flu_hospitalizations["Date"],monthly_flu_hospitalizations["Count"],color='#648FFF',label='Flu')
# ax[0].plot(monthly_rsv_hospitalizations["Date"],monthly_rsv_hospitalizations["Count"],color='#DC267F',label='RSV')
# ax[0].plot(monthly_covid_hospitalizations["Date"],monthly_covid_hospitalizations["Count"],color='#FFB000',label='COVID-19')
# ax[0].set_title('Hospitalizations by month')
# ax[0].set_ylabel('Number of hospitalizations')
# ax[0].legend()
# ax[1].plot(monthly_flu_hospitalizations["Date"],monthly_flu_hospitalizations["Relative"],color='#648FFF',label='Flu')
# ax[1].plot(monthly_rsv_hospitalizations["Date"],monthly_rsv_hospitalizations["Relative"],color='#DC267F',label='RSV')
# ax[1].plot(monthly_covid_hospitalizations["Date"],monthly_covid_hospitalizations["Relative"],color='#FFB000',label='COVID-19')
# ax[1].set_xlabel('Date')
# ax[1].set_ylabel('Proportion of maximum hospitalizations')
# plt.tight_layout()
# plt.savefig('Figures/KPSC_flu_COVID_and_RSV_hospitalizations_by_month_relative.png')