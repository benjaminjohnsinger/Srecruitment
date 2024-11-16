import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sas7bdat import SAS7BDAT
from matplotlib.cm import hsv

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
# format date
hospitalizations["Date"] = pd.to_datetime(hospitalizations["Date"])
hospitalizations["Month"] = hospitalizations["Date"].dt.month
hospitalizations["Year"] = hospitalizations["Date"].dt.year

# load age population data
age_by_year = pd.read_csv("Data/Processed/KPSC_population_by_age.csv")
age_by_year["Year"] = np.arange(2015,2023)
age_by_year = age_by_year.set_index("Year")
age_by_year.columns = ['<3mo','3-12mo','1-4y','5-17y','18-39y','40-64y','>=65y']
# sum '<3mo','3-12mo' into <1y
age_by_year['<1y'] = age_by_year['<3mo'] + age_by_year['3-12mo']
age_by_year = age_by_year.drop(columns=['<3mo','3-12mo'])
# order columns
age_by_year = age_by_year[['<1y','1-4y','5-17y','18-39y','40-64y','>=65y']]
# repeat last row for 2023
age_by_year.loc[2023] = age_by_year.loc[2022]


# assign each row to age group
AGE_GROUPS = [range(0),range(1),range(1,5),range(5,18),range(18,40),range(40,65),range(65,100)]
AGE_GROUP_NAMES = ['0','<1y','1-4y','5-17y','18-39y','40-64y','>=65y']
for i in range(len(AGE_GROUPS)):
    hospitalizations.loc[hospitalizations["age"].isin(AGE_GROUPS[i]),"age group"] = AGE_GROUP_NAMES[i]


# get only influenza hospitalizations, i.e. CODE starts with J09, J10, or J11
flu_hospitalizations = hospitalizations[hospitalizations["CODE"].str.contains('J09|J10|J11')]
# get only RSV hospitalizations, i.e. CODE is J12.1, J21.0, or B97.4 as used in Pitzer et al. 2015
rsv_hospitalizations = hospitalizations[hospitalizations["CODE"].str.contains('J12.1|J21.0|B97.4')]
# get only COVID-19 hospitalizations, i.e. CODE is U07.1
covid_hospitalizations = hospitalizations[hospitalizations["CODE"].str.contains('U07.1')]

def month_format(hosps,age_strat=True,incidence=False):
    # # filter out just one row for each value of StudyID
    hosps = hosps.drop_duplicates(subset="StudyID")
    if age_strat:
        hosps = hosps.groupby(["Year","Month","age group"]).size().reset_index(name='Count')
    else:
        hosps = hosps.groupby(["Year","Month"]).size().reset_index(name='Count')
    # # fill in zeros for missing months and age groups
    for date in pd.date_range(start='2015-10-01',end='2023-09-30',freq='MS'):
        year = date.year
        month = date.month
        if age_strat:
            for age_group in AGE_GROUP_NAMES:
                if not (((hosps["Year"] == year) & (hosps["Month"] == month)) & (hosps["age group"] == age_group)).any():
                    hosps = pd.concat([hosps,pd.DataFrame({"Year":[year],"Month":[month],"age group":[age_group],"Count":[0]})],ignore_index=True)
        else:
            if not ((hosps["Year"] == year) & (hosps["Month"] == month)).any():
                hosps = pd.concat([hosps,pd.DataFrame({"Year":[year],"Month":[month],"Count":[0]})],ignore_index=True)
    hosps["Date"] = pd.to_datetime(hosps["Year"].astype(str) + '-' + hosps["Month"].astype(str) + '-01')
    # sort by date
    hosps = hosps.sort_values(by="Date")
    # date as index, remove year month and date as columns
    hosps = hosps.set_index("Date")
    hosps = hosps.drop(columns=["Year","Month"])
    if age_strat:
        # columns as age groups
        hosps = hosps.pivot(columns="age group",values="Count")
        # drop 0 column
        hosps = hosps.drop(columns='0')
        # order columns
        hosps = hosps[['<1y','1-4y','5-17y','18-39y','40-64y','>=65y']]
    if incidence:
        # divide by age group population, matched by year
        hosps = hosps.div(age_by_year.loc[hosps.index.year].values)
    return hosps

monthly_flu_hospitalizations = month_format(flu_hospitalizations,incidence=True)
monthly_rsv_hospitalizations = month_format(rsv_hospitalizations,incidence=True)
monthly_covid_hospitalizations = month_format(covid_hospitalizations,incidence=True)

fig, ax = plt.subplots(3,1,figsize=(6.5,8.5),sharex=True)
colors = hsv(-0.02+np.arange(6)/6)
# colors = ['#648FFF','#DC267F','#FFB000','#785EF0','#FF832B','#000000']
# colors = ['red','orange','yellow','green','blue','purple']
monthly_flu_hospitalizations.plot(ax=ax[0],color=colors,label=AGE_GROUP_NAMES[1:],title='Flu hospitalizations')
monthly_rsv_hospitalizations.plot(ax=ax[1],color=colors,label=AGE_GROUP_NAMES[1:],title='RSV hospitalizations')
monthly_covid_hospitalizations.plot(ax=ax[2],color=colors,label=AGE_GROUP_NAMES[1:],title='COVID-19 hospitalizations')
ax[1].set_ylabel('Incidence of hospitalizations')
ax[2].set_xlabel('Date')
plt.tight_layout()
plt.savefig('Figures/KPSC_hospitalizations_incidence_by_age.png')

# # # save to csv
# monthly_flu_hospitalizations.to_csv('Data/Processed/KPSC_flu_hosp_incidence_by_age.csv',index=False)
# monthly_rsv_hospitalizations.to_csv('Data/Processed/KPSC_rsv_hosp_incidence_by_age.csv',index=False)
# monthly_covid_hospitalizations.to_csv('Data/Processed/KPSC_covid_hosp_incidence_by_age.csv',index=False)

# # flu_hospitalizations["Relative"] = flu_hospitalizations["Count"]/flu_hospitalizations["Count"].max()
# # rsv_hospitalizations["Relative"] = rsv_hospitalizations["Count"]/rsv_hospitalizations["Count"].max()
# # covid_hospitalizations["Relative"] = covid_hospitalizations["Count"]/covid_hospitalizations["Count"].max()

# fig, ax = plt.subplots(1,1,figsize=(6.5,6.5),sharex=True)
# # ax[0].plot(flu_hospitalizations["Date"],flu_hospitalizations["Count"],color='#648FFF',label='Flu')
# ax.plot(rsv_hospitalizations["Date"],rsv_hospitalizations["Count"],color='#DC267F',label='RSV')
# # ax[0].plot(covid_hospitalizations["Date"],covid_hospitalizations["Count"],color='#FFB000',label='COVID-19')
# ax.set_title('Hospitalizations by month')
# ax.set_ylabel('Number of hospitalizations')
# ax.legend()
# # ax[1].plot(flu_hospitalizations["Date"],flu_hospitalizations["Relative"],color='#648FFF',label='Flu')
# # ax[1].plot(rsv_hospitalizations["Date"],rsv_hospitalizations["Relative"],color='#DC267F',label='RSV')
# # ax[1].plot(covid_hospitalizations["Date"],covid_hospitalizations["Relative"],color='#FFB000',label='COVID-19')
# ax.set_xlabel('Date')
# # ax[1].set_ylabel('Proportion of maximum hospitalizations')
# plt.tight_layout()
# plt.savefig('Figures/KPSC_RSV_hospitalizations_by_month_relative.png')