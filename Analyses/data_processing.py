import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sas7bdat import SAS7BDAT
from matplotlib.cm import hsv
import pickle


pathogen_names = {"RSV": ["RESPIRATORY SYNCYTIAL VIRUS","RESPIRATORY SYNCYTIAL VIRUS SUBTYPE A","RESPIRATORY SYNCYTIAL VIRUS SUBTYPE B"],
"Influenza A": ["INFLUENZA A","INFLUENZA A H1N1 2009","INFLUENZA A VIRUS","INFLUENZA A VIRUS SUBTYPE H1","INFLUENZA A VIRUS SUBTYPE/HEMAGGLUTININ H3","INFLUENZA VIRUS A","INFLUENZA VIRUS A+B"],
"Influenza A H1": ["INFLUENZA A H1N1 2009","INFLUENZA A VIRUS SUBTYPE H1"],
"Influenza A H3": ["INFLUENZA A VIRUS SUBTYPE/HEMAGGLUTININ H3"],
"Influenza B": ["INFLUENZA B","INFLUENZA VIRUS B","INFLUENZA VIRUS A+B"],
"Influenza": ["INFLUENZA A","INFLUENZA A H1N1 2009","INFLUENZA A VIRUS","INFLUENZA A VIRUS SUBTYPE H1","INFLUENZA A VIRUS SUBTYPE/HEMAGGLUTININ H3","INFLUENZA VIRUS A","INFLUENZA VIRUS A+B","INFLUENZA B","INFLUENZA VIRUS B"],
"Metapneumovirus": ["HUMAN METAPNEUMOVIRUS VIRUS",],
"Adenovirus": ["ADENOVIRUS",],
"Parainfuenza": ["PARAINFLUENZA VIRUS 1","PARAINFLUENZA VIRUS 2","PARAINFLUENZA VIRUS 3","PARAINFLUENZA VIRUS 4"],
"Parainfluenza 3": ["PARAINFLUENZA VIRUS 3"]}
AGE_GROUP_NAMES = ['<3m','3-11m','1-4y','5-17y','18-39y','40-64y','>=65y']
respiratory_codes = pd.read_csv('Data/Processed/respiratory_codes.csv')
positive_tests = pd.read_csv('Data/Processed/KPSC_positive_matched_hospitalizations.csv')
for incidence in [False, True]:
    for AGE_GROUPS in [[range(0,3), range(3,12),range(12,5*12),range(5*12,18*12),range(18*12,40*12),range(40*12,65*12),range(65*12,90*12)]]:
        for pathogen in pathogen_names.keys():
            print(pathogen)
            if incidence:
                # load age population data
                age_by_year = pd.read_csv("Data/Processed/KPSC_population_by_age.csv")
                if AGE_GROUPS is not None:
                    age_by_year.columns = AGE_GROUP_NAMES
                age_by_year["Year"] = np.arange(2015,2023)
                age_by_year = age_by_year.set_index("Year")
                # repeat last row for 2023
                age_by_year.loc[2023] = age_by_year.loc[2022]
            names = pathogen_names[pathogen]
            cases = positive_tests[positive_tests['pathogen'].isin(names) & positive_tests['CODE'].isin(respiratory_codes)]
            cases = cases.drop_duplicates(subset=cases.columns.difference(['CODE','dxgroup']))
            cases["Date"] = pd.to_datetime(cases["Hospitalization date"])
            cases["Month"] = cases["Date"].dt.month
            cases["Year"] = cases["Date"].dt.year
            if AGE_GROUPS is not None:
                for i in range(len(AGE_GROUPS)):
                    cases.loc[cases["age_in_mo"].isin(AGE_GROUPS[i]),"age_group"] = AGE_GROUP_NAMES[i]
                cases = cases.groupby(["Year","Month","age_group"]).size().reset_index(name='Count')
            else:
                cases = cases.groupby(["Year","Month"]).size().reset_index(name='Count')

            for date in pd.date_range(start='2015-10-01',end='2023-09-30',freq='MS'):
                year = date.year
                month = date.month
                if AGE_GROUPS is not None:
                    for age_group in AGE_GROUP_NAMES:
                        if not ((cases["Year"]==year) & (cases["Month"]==month) & (cases["age_group"]==age_group)).any():
                            cases = pd.concat([cases,pd.DataFrame({"Year":[year],"Month":[month],"age_group":[age_group],"Count":[0]})])
                if not ((cases["Year"]==year) & (cases["Month"]==month)).any():
                    cases = pd.concat([cases,pd.DataFrame({"Year":[year],"Month":[month],"Count":[0]})])
            cases["Date"] = pd.to_datetime(cases["Year"].astype(str) + '-' + cases["Month"].astype(str) + '-01')
            cases = cases.sort_values(by="Date")
            cases = cases.set_index("Date")
            cases = cases.drop(columns=["Year","Month"])
            if AGE_GROUPS is not None:
                cases = cases.pivot(columns="age_group",values="Count")
                cases = cases[AGE_GROUP_NAMES]
                if incidence:
                    cases = cases.div(age_by_year.loc[cases.index.year].values)
            elif incidence:
                cases = cases.div(np.sum(age_by_year.loc[cases.index.year].values,axis=1).reshape(-1,1))
            filename = f'Data/Processed/KPSC_{pathogen}_{["cases","incidence"][incidence]}_{["all","age"][AGE_GROUPS is not None]}.csv'
            print(cases.shape)
            cases.to_csv(filename,index=True)


# with SAS7BDAT('Data/Raw/KPSC/clinical_20241202.sas7bdat') as f:
#     clinical_data = f.to_data_frame()

# # with SAS7BDAT('Data/Raw/KPSC/clinical.sas7bdat') as f:
# #     clinical_data_no_mo = f.to_data_frame()

# with SAS7BDAT('Data/Raw/KPSC/testing.sas7bdat') as f:
#     test_data = f.to_data_frame()

# # save sample of clinical data
# clinical_data.sample(10000).to_csv('Data/Processed/KPSC_clinical_sample.csv',index=False)

# print(clinical_data.head())

# # for each row in clinical_data, find the row in clinical_data_no_mo with the same age, StudyID, and dx_days, and copy the value of the YEAR column
# clinical_data["YEAR"] = clinical_data.apply(lambda row: clinical_data_no_mo[(clinical_data_no_mo["age"] == row["age"]) & (clinical_data_no_mo["StudyID"] == row["StudyID"]) & (clinical_data_no_mo["dx_days"] == row["dx_days"])]["YEAR"].values[0],axis=1)

# # save to csv
# clinical_data.to_csv('Data/Processed/KPSC_clinical_with_year_and_age_in_months.csv',index=False)

# # load clinical data sample
# clinical_data = pd.read_csv('Data/Processed/KPSC_clinical_sample.csv')
# print(clinical_data.head())

# # date is 1st of October of each year (in YEAR column), plus dx_days
# clinical_data["Date"] = pd.to_datetime(clinical_data["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(clinical_data["dx_days"],unit='D')

# test_data = test_data[test_data["StudyID"].isin(clinical_data["StudyID"])]
# positive_tests = test_data[test_data["result_val"] == 'Positive']
# positive_tests["Date"] = pd.to_datetime(positive_tests["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(positive_tests["lab_days"],unit='D')
# positive_tests = clinical_data.merge(positive_tests[["StudyID","Date","pathogen","lab_type","lab_days"]],on="StudyID",how="left")
# positive_tests = positive_tests.rename(columns={"Date_x":"Clinical date","Date_y":"Test date"})
# positive_tests = positive_tests[np.abs((pd.to_datetime(positive_tests["Clinical date"]) - pd.to_datetime(positive_tests["Test date"])).dt.days) <= 14]

# # save to csv
# positive_tests.to_csv('Data/Processed/KPSC_positive_matched_all_clinical.csv',index=False)
# # load
# positive_tests = pd.read_csv('Data/Processed/KPSC_positive_matched_all_clinical.csv')
# hospitalizations = clinical_data[(clinical_data["setting"] == 'Hospital admission')]

# # save hostpitalizations to csv
# hospitalizations.to_csv('Data/Processed/KPSC_clinical_hospitalizations.csv',index=False)
# # load
# hospitalizations = pd.read_csv('Data/Processed/KPSC_clinical_hospitalizations.csv')


# respiratory_codes = pd.read_csv('Data/Processed/respiratory_codes.csv',dtype=str)
# gastroenteritis_codes = pd.read_csv('Data/Processed/gastroenteritis_codes.csv',dtype=str)
# # # # get only hospitalizations with respiratory or gastroenteritis codes
# respiratory_hospitalizations = hospitalizations[hospitalizations["CODE"].isin(respiratory_codes)]
# gastroenteritis_hospitalizations = hospitalizations[hospitalizations["CODE"].isin(gastroenteritis_codes)]
# # # # save
# respiratory_hospitalizations.to_csv('Data/Processed/KPSC_clinical_respiratory_hospitalizations.csv',index=False)
# gastroenteritis_hospitalizations.to_csv('Data/Processed/KPSC_clinical_gastroenteritis_hospitalizations.csv',index=False)
# # # respiratory_hospitalizations = pd.read_csv('Data/Processed/KPSC_clinical_respiratory_hospitalizations.csv')
# # # gastroenteritis_hospitalizations = pd.read_csv('Data/Processed/KPSC_clinical_gastroenteritis_hospitalizations.csv')

# # # # filter test data to only include tests with StudyID matching a value in hospitalizations
# # test_data = test_data[test_data["StudyID"].isin(hospitalizations["StudyID"])]
# # test_data["Date"] = pd.to_datetime(test_data["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(test_data["lab_days"],unit='D')
# # # # # save test data
# # test_data.to_csv('Data/Processed/KPSC_hospitalized_tests.csv',index=False)
# # # # load test data
# test_data = pd.read_csv('Data/Processed/KPSC_hospitalized_tests.csv')
# positive_tests = test_data[test_data["result_val"] == 'Positive']

# # # # find date of from hospitalizations for each study ID and match to positive tests
# positive_tests = hospitalizations.merge(positive_tests[["StudyID","Date","pathogen","lab_type","lab_days"]],on="StudyID",how="left")
# # # # rename columns
# positive_tests = positive_tests.rename(columns={"Date_x":"Hospitalization date","Date_y":"Test date"})
# # # # keep only rows where Hospitalization date is within 14 days of Test date
# positive_tests = positive_tests[np.abs((pd.to_datetime(positive_tests["Hospitalization date"]) - pd.to_datetime(positive_tests["Test date"])).dt.days) <= 14]

# # # save to csv
# positive_tests.to_csv('Data/Processed/KPSC_positive_matched_hospitalizations.csv',index=False)
# # load
# positive_tests = pd.read_csv('Data/Processed/KPSC_positive_matched_hospitalizations.csv')
# rota_names = ["ROTAVIRUS",]
# rsv_names = ["RESPIRATORY SYNCYTIAL VIRUS","RESPIRATORY SYNCYTIAL VIRUS SUBTYPE A","RESPIRATORY SYNCYTIAL VIRUS SUBTYPE B"]
# flu_a_names = ["INFLUENZA A","INFLUENZA A H1N1 2009","INFLUENZA A VIRUS","INFLUENZA A VIRUS SUBTYPE H1","INFLUENZA A VIRUS SUBTYPE/HEMAGGLUTININ H3","INFLUENZA VIRUS A","INFLUENZA VIRUS A+B"]
# flu_b_names = ["INFLUENZA B","INFLUENZA VIRUS B","INFLUENZA VIRUS A+B"]
# all_flu_names = flu_a_names + flu_b_names
# paraflu_names = ["PARAINFLUENZA VIRUS 3",]
# # get only RSV tests
# paraflu = positive_tests[(positive_tests["pathogen"].isin(paraflu_names)) & (positive_tests["CODE"].isin(respiratory_codes))]

# # remove lines that are identical except for CODE and dxgroup
# paraflu = paraflu.drop_duplicates(subset=paraflu.columns.difference(['CODE','dxgroup']))
# print(paraflu["pathogen"].value_counts())
# # # save
# # paraflu.to_csv('Data/Processed/KPSC_paraflu_positive_hospitalizations.csv',index=False)
# # plot paraflu cases over time
# paraflu["Date"] = pd.to_datetime(paraflu["Hospitalization date"])
# paraflu["Month"] = paraflu["Date"].dt.month
# paraflu["Year"] = paraflu["Date"].dt.year
# paraflu = paraflu.groupby(["Year","Month"]).size().reset_index(name='Count')
# for date in pd.date_range(start='2015-10-01',end='2023-09-30',freq='MS'):
#     year = date.year
#     month = date.month
#     if not ((paraflu["Year"] == year) & (paraflu["Month"] == month)).any():
#         paraflu = pd.concat([paraflu,pd.DataFrame({"Year":[year],"Month":[month],"Count":[0]})],ignore_index=True)
# paraflu["Date"] = pd.to_datetime(paraflu["Year"].astype(str) + '-' + paraflu["Month"].astype(str) + '-01')
# paraflu = paraflu.set_index("Date")
# paraflu = paraflu.sort_values(by="Date")
# paraflu = paraflu.drop(columns=["Year","Month"])
# paraflu.plot()
# plt.title('paraflu 3 positive hospitalizations')
# plt.ylabel('Number of positive tests')
# plt.xlabel('Date')
# # plt.yticks([0,1,2])
# plt.tight_layout()
# plt.savefig('Figures/KPSC_paraflu3_positive_hospitalizations.png')

# # remove duplicates with same study ID
# # print(print(positive_tests[positive_tests["pathogen"] == "ROTAVIRUS"]))
# # print(np.sum(positive_tests["pathogen"] == "ROTAVIRUS"))

# # # for each study ID, get the distance between the date of the test and the date of hospitalization, and plot a histogram, with log-log axes
# # test_data["Test to admission"] = (pd.to_datetime(positive_tests["Hospitalization Date"])-pd.to_datetime(positive_tests["Test Date"])).dt.days
# # # remove values greater than 20 or lower than -20
# # test_data = test_data[(test_data["Test to admission"] >= -20) & (test_data["Test to admission"] <= 20)]
# # # test_data["Test to admission"] = test_data["Test to admission"].clip(-20,20)
# # test_data["Test to admission"].hist(bins=41)
# # # plt.yscale('log')
# # # plt.xscale('log')
# # plt.xlabel('Days from positive test')
# # plt.ylabel('Number of tests')
# # plt.title('Histogram of days from positive test until hospital admission')
# # plt.tight_layout()
# # plt.savefig('Figures/KPSC_days_to_test_histogram_crop_unique.png')


# # load hospitalizations
# hospitalizations = pd.read_csv('Data/Processed/KPSC_clinical_hospitalizations.csv')

# # format date
# hospitalizations["Date"] = pd.to_datetime(hospitalizations["Date"])
# hospitalizations["Month"] = hospitalizations["Date"].dt.month
# hospitalizations["Year"] = hospitalizations["Date"].dt.year

# # load age population data
# age_by_year = pd.read_csv("Data/Processed/KPSC_population_by_age.csv")
# print(age_by_year)
# age_by_year["Year"] = np.arange(2015,2023)
# age_by_year = age_by_year.set_index("Year")
# age_by_year.columns = ['<3mo','3-12mo','1-4y','5-17y','18-39y','40-64y','>=65y']
# # sum '<3mo','3-12mo' into <1y
# age_by_year['<1y'] = age_by_year['<3mo'] + age_by_year['3-12mo']
# age_by_year = age_by_year.drop(columns=['<3mo','3-12mo'])
# # order columns
# age_by_year = age_by_year[['<1y','1-4y','5-17y','18-39y','40-64y','>=65y']]
# # repeat last row for 2023
# age_by_year.loc[2023] = age_by_year.loc[2022]
# print(age_by_year)

# # assign each row to age group
# AGE_GROUPS = [range(0),range(1),range(1,5),range(5,18),range(18,40),range(40,65),range(65,100)]
# AGE_GROUP_NAMES = ['0','<1y','1-4y','5-17y','18-39y','40-64y','>=65y']
# for i in range(len(AGE_GROUPS)):
#     hospitalizations.loc[hospitalizations["age"].isin(AGE_GROUPS[i]),"age group"] = AGE_GROUP_NAMES[i]


# # get only influenza hospitalizations, i.e. CODE starts with J09, J10, or J11
# flu_hospitalizations = hospitalizations[hospitalizations["CODE"].str.contains('J09|J10|J11')]
# # get only RSV hospitalizations, i.e. CODE is J12.1, J21.0, or B97.4 as used in Pitzer et al. 2015
# rsv_hospitalizations = hospitalizations[hospitalizations["CODE"].str.contains('J12.1|J21.0|B97.4')]
# # get only COVID-19 hospitalizations, i.e. CODE is U07.1
# covid_hospitalizations = hospitalizations[hospitalizations["CODE"].str.contains('U07.1')]

# ge_hospitalizations = hospitalizations[hospitalizations["CODE"].isin(gastroenteritis_codes)]

# def month_format(hosps,age_strat=True,incidence=False):
#     # # filter out just one row for each value of StudyID
#     hosps = hosps.drop_duplicates(subset="StudyID")
#     if age_strat:
#         hosps = hosps.groupby(["Year","Month","age group"]).size().reset_index(name='Count')
#     else:
#         hosps = hosps.groupby(["Year","Month"]).size().reset_index(name='Count')
#     # # fill in zeros for missing months and age groups
#     for date in pd.date_range(start='2015-10-01',end='2023-09-30',freq='MS'):
#         year = date.year
#         month = date.month
#         if age_strat:
#             for age_group in AGE_GROUP_NAMES:
#                 if not (((hosps["Year"] == year) & (hosps["Month"] == month)) & (hosps["age group"] == age_group)).any():
#                     hosps = pd.concat([hosps,pd.DataFrame({"Year":[year],"Month":[month],"age group":[age_group],"Count":[0]})],ignore_index=True)
#         else:
#             if not ((hosps["Year"] == year) & (hosps["Month"] == month)).any():
#                 hosps = pd.concat([hosps,pd.DataFrame({"Year":[year],"Month":[month],"Count":[0]})],ignore_index=True)
#     hosps["Date"] = pd.to_datetime(hosps["Year"].astype(str) + '-' + hosps["Month"].astype(str) + '-01')
#     # sort by date
#     hosps = hosps.sort_values(by="Date")
#     # date as index, remove year month and date as columns
#     hosps = hosps.set_index("Date")
#     hosps = hosps.drop(columns=["Year","Month"])
#     if age_strat:
#         # columns as age groups
#         hosps = hosps.pivot(columns="age group",values="Count")
#         # drop 0 column
#         hosps = hosps.drop(columns='0')
#         # order columns
#         hosps = hosps[['<1y','1-4y','5-17y','18-39y','40-64y','>=65y']]
#     if incidence:
#         # divide by age group population, matched by year
#         hosps = hosps.div(age_by_year.loc[hosps.index.year].values)
#     return hosps

# monthly_flu_hospitalizations = month_format(flu_hospitalizations,incidence=False)
# monthly_rsv_hospitalizations = month_format(rsv_hospitalizations,incidence=False)
# monthly_covid_hospitalizations = month_format(covid_hospitalizations,incidence=False)
# monthly_ge_hospitalizations = month_format(ge_hospitalizations,incidence=False)
# monthly_ge_hospitalizations_noage = month_format(ge_hospitalizations,incidence=False,age_strat=False)

# print(monthly_rsv_hospitalizations)

# fig, axs = plt.subplots(2,1,figsize=(6.5,6.5),sharex=True)
# colors = hsv(-0.02+np.arange(6)/6)
# monthly_ge_hospitalizations_noage.plot(ax=axs[0],title='Gastroentiritis hospitalizations')
# monthly_ge_hospitalizations.plot(ax=axs[1],color=colors,label=AGE_GROUP_NAMES[1:])

# axs[0].set_ylabel('Hospitalizations')
# axs[1].set_ylabel('Hospitalizations')
# axs[1].set_xlabel('Date')
# plt.tight_layout()
# plt.savefig('Figures/KPSC_ge_hospitalizations_by_age.png')


# # fig, ax = plt.subplots(4,1,figsize=(6.5,8.5),sharex=True)
# # colors = hsv(-0.02+np.arange(6)/6)
# # # colors = ['#648FFF','#DC267F','#FFB000','#785EF0','#FF832B','#000000']
# # # colors = ['red','orange','yellow','green','blue','purple']
# # monthly_flu_hospitalizations.plot(ax=ax[0],color=colors,label=AGE_GROUP_NAMES[1:],title='Flu hospitalizations')
# # monthly_rsv_hospitalizations.plot(ax=ax[1],color=colors,label=AGE_GROUP_NAMES[1:],title='RSV hospitalizations')
# # monthly_covid_hospitalizations.plot(ax=ax[2],color=colors,label=AGE_GROUP_NAMES[1:],title='COVID-19 hospitalizations')
# # monthly_rotavirus_hospitalizations.plot(ax=ax[3],color=colors,label=AGE_GROUP_NAMES[1:],title='Rotavirus hospitalizations')
# # ax[2].set_ylabel('Incidence of hospitalizations')
# # ax[3].set_xlabel('Date')
# # plt.tight_layout()
# # plt.savefig('Figures/KPSC_hospitalizations_incidence_by_age_w_rota.png')

# # # save to csv
# # monthly_flu_hospitalizations.to_csv('Data/Processed/KPSC_flu_hosp_incidence_by_age.csv',index=False)
# # monthly_rsv_hospitalizations.to_csv('Data/Processed/KPSC_rsv_hosp_incidence_by_age.csv',index=False)
# # monthly_covid_hospitalizations.to_csv('Data/Processed/KPSC_covid_hosp_incidence_by_age.csv',index=False)
# # monthly_rotavirus_hospitalizations.to_csv('Data/Processed/KPSC_rota_hosp_incidence_by_age.csv',index=False)

# # flu_hospitalizations["Relative"] = flu_hospitalizations["Count"]/flu_hospitalizations["Count"].max()
# # rsv_hospitalizations["Relative"] = rsv_hospitalizations["Count"]/rsv_hospitalizations["Count"].max()
# # covid_hospitalizations["Relative"] = covid_hospitalizations["Count"]/covid_hospitalizations["Count"].max()

# # fig, ax = plt.subplots(1,1,figsize=(6.5,6.5),sharex=True)
# # # ax[0].plot(flu_hospitalizations["Date"],flu_hospitalizations["Count"],color='#648FFF',label='Flu')
# # ax.plot(monthly_rotavirus_hospitalizations["Date"],monthly_rotavirus_hospitalizations["Count"],color='#DC267F',label='RSV')
# # # ax[0].plot(covid_hospitalizations["Date"],covid_hospitalizations["Count"],color='#FFB000',label='COVID-19')
# # ax.set_title('Hospitalizations by month')
# # ax.set_ylabel('Number of hospitalizations')
# # ax.legend()
# # # ax[1].plot(flu_hospitalizations["Date"],flu_hospitalizations["Relative"],color='#648FFF',label='Flu')
# # # ax[1].plot(rsv_hospitalizations["Date"],rsv_hospitalizations["Relative"],color='#DC267F',label='RSV')
# # # ax[1].plot(covid_hospitalizations["Date"],covid_hospitalizations["Relative"],color='#FFB000',label='COVID-19')
# # ax.set_xlabel('Date')
# # # ax[1].set_ylabel('Proportion of maximum hospitalizations')
# # plt.tight_layout()
# # plt.savefig('Figures/KPSC_rota_hospitalizations_by_month_relative.png')