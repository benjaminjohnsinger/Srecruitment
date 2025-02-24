import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sas7bdat import SAS7BDAT
from matplotlib import cm as colormaps
from Parameters.census_population import *
from plotting import *
import pickle

############### CDC data ###############
# ### RSV data
# rsv_pre2020 = pd.read_csv('Data/Raw/Respiratory_Syncytial_Virus_Laboratory_Data__NREVSS_.csv')
# rsv_post2020 = pd.read_csv('Data/Raw/Percent_Positivity_of_Respiratory_Syncytial_Virus_Nucleic_Acid_Amplification_Tests_by_HHS_Region__National_Respiratory_and_Enteric_Virus_Surveillance_System_20250213.csv')

# # get dates from column "Week ending Date" which are in format e.g. 22JUL2017
# rsv_pre2020["Date"] = pd.to_datetime(rsv_pre2020["Week ending Date"],format='%d%b%Y')
# # get dates from the column "mmwrweek_end" which are in fomrat e.g. 04/11/2020 12:00:00 AM
# rsv_post2020["Date"] = pd.to_datetime(rsv_post2020["mmwrweek_end"],format='%m/%d/%Y %I:%M:%S %p')

# # positivity pre 2020 is column "RSV Detections"/"RSV Tests"
# rsv_pre2020["Percent Positive"] = 100*rsv_pre2020["RSV Detections"]/rsv_pre2020["RSV Tests"]
# rsv_pre2020.fillna(0,inplace=True)
# # positivity post 2020 is column "pcr_percent_positive"
# rsv_post2020["Percent Positive"] = rsv_post2020["pcr_percent_positive"]

# # add "Region " string in formt of "HHS region " column in pre 2020 data
# rsv_pre2020["Region"] = "Region " + rsv_pre2020["HHS region "].astype(str)
# # drop "National" level from post 2020 data and rename
# rsv_post2020 = rsv_post2020[rsv_post2020["level"] != "National"]
# rsv_post2020.rename(columns={"level":"Region"},inplace=True)

# # concatenate time series
# rsv_pp = pd.concat([rsv_pre2020[["Date","Region","Percent Positive"]],rsv_post2020[["Date","Region","Percent Positive"]]])

# # average duplicates
# rsv_pp = rsv_pp.groupby(["Date","Region"]).mean().reset_index()

# # index and pivot
# rsv_pp_regional = rsv_pp.pivot(index="Date",columns="Region",values="Percent Positive")
# rsv_pp_regional = rsv_pp_regional[["Region " + str(i) for i in range(1,9)]+["Region 10"]+["Region 9"]]
# rsv_pp_regional.to_csv('Data/Processed/RSV_PercentPositive_Regions.csv')


# ### flu data
flu_pre2015 = pd.read_csv('Data/Raw/FluViewPhase2Data_States/WHO_NREVSS_Combined_prior_to_2015_16.csv')
flu_clinical = pd.read_csv('Data/Raw/FluViewPhase2Data_States/WHO_NREVSS_Clinical_Labs.csv')
flu_ph = pd.read_csv('Data/Raw/FluViewPhase2Data_States/WHO_NREVSS_Public_Health_Labs.csv')

flu_pre2015["Date"] = pd.to_datetime(flu_pre2015["YEAR"].astype(int).astype(str) + '-01-01') + pd.to_timedelta(flu_pre2015["WEEK"]*7,unit='D')
flu_clinical["Date"] = pd.to_datetime(flu_clinical["YEAR"].astype(int).astype(str) + '-01-01') + pd.to_timedelta(flu_clinical["WEEK"]*7,unit='D')

# remove Pueto Rico, Virgin Islands, District of Columbia, New York City, and Rhode Island
flu_pre2015 = flu_pre2015[~flu_pre2015["REGION"].isin(["Puerto Rico","Virgin Islands","New York City"])]
flu_clinical = flu_clinical[~flu_clinical["REGION"].isin(["Puerto Rico","Virgin Islands","New York City"])]

# replace "X" with NA
flu_pre2015.replace("X",0.0,inplace=True)
flu_clinical.replace("X",0.0,inplace=True)

# flu A columns are A (2009 H1N1),A (H1),A (H3),A (Subtyping not Performed),A (Unable to Subtype),H3N2v,A (H5)
flu_pre2015["A"] = flu_pre2015["A (2009 H1N1)"].astype(float) + flu_pre2015["A (H1)"].astype(float) + flu_pre2015["A (H3)"].astype(float) + flu_pre2015["A (Subtyping not Performed)"].astype(float) + flu_pre2015["A (Unable to Subtype)"].astype(float) + flu_pre2015["H3N2v"].astype(float) + flu_pre2015["A (H5)"].astype(float)
flu_pre2015["PERCENT A"] = 100*flu_pre2015["A"]/flu_pre2015["TOTAL SPECIMENS"].astype(float)
flu_pre2015["PERCENT B"] = 100*flu_pre2015["B"].astype(float)/flu_pre2015["TOTAL SPECIMENS"].astype(float)
# NA to 0
flu_pre2015.fillna(0,inplace=True)

# get PERCENT POSITIVE for each week by concatenating time series from pre-2015 and post-2015 clinical data
fluA_pp = pd.concat([flu_pre2015[["Date","REGION","PERCENT A"]],flu_clinical[["Date","REGION","PERCENT A"]]])
fluA_pp.rename(columns={"PERCENT A":"PERCENT POSITIVE"},inplace=True)
fluB_pp = pd.concat([flu_pre2015[["Date","REGION","PERCENT B"]],flu_clinical[["Date","REGION","PERCENT B"]]])
fluB_pp.rename(columns={"PERCENT B":"PERCENT POSITIVE"},inplace=True)
# index
fluA_pp_regional = fluA_pp.pivot(index="Date",columns="REGION",values="PERCENT POSITIVE")
# convert values type into float
fluA_pp_regional = fluA_pp_regional.astype(float)
# fluA_pp_regional = fluA_pp_regional[["Region " + str(i) for i in range(1,9)]+["Region 10"]+["Region 9"]]
fluB_pp_regional = fluB_pp.pivot(index="Date",columns="REGION",values="PERCENT POSITIVE")
fluB_pp_regional = fluB_pp_regional.astype(float)
# fluB_pp_regional = fluB_pp_regional[["Region " + str(i) for i in range(1,9)]+["Region 10"]+["Region 9"]]
fluA_pp_regional.to_csv('Data/Processed/FluView_PercentPositive_States_A.csv')
fluB_pp_regional.to_csv('Data/Processed/FluView_PercentPositive_States_B.csv')


# plots
# colors = colormaps.get_cmap('Greys',9)(np.linspace(1,0.3,9)).tolist()
# colors.append('red')
fig, ax = plt.subplots(2,1,figsize=(13.3,7.5),sharey=True,sharex=True)
fluA_pp_regional.plot(ax=ax[0],legend=False,color='k',alpha=0.3)
fluB_pp_regional.plot(ax=ax[1],legend=False,color='k',alpha=0.3)
ax[0].set_ylabel("Percent positive for flu A")
ax[1].set_ylabel("Percent positive for flu B")
# legend label is State
# ax[1].legend(title="State")
fig.suptitle("FluView Percent Positive by State")
plt.tight_layout()
plt.savefig('Figures/FluView_PercentPositive_States_ppt.png',dpi=300)

# # plots
# # plot the region between 2022-01-01 and 2022-06-01
# # colors = ["#648FFF", "#DC267F", "#FFB000", "#785EF0", "#000000", "#648FFF", "#DC267F", "#FFB000", "#785EF0", "#000000"]
# # linestyles = ['-']*5+['--']*5
# linestyles = ['-']*10
# fig, ax = plt.subplots(1,1,figsize=(13.3,7.5),sharey=True,sharex=True)
# for i,region in enumerate(rsv_pp_regional.columns):
#     rsv_pp_regional[region].plot(ax=ax, color=colors[i], linestyle=linestyles[i], label=region)
# # fluA_pp_regional.loc['2022-01-01':'2022-06-01'].plot(ax=ax[0],color=colors,legend=False)
# ax.legend(title="HHS region")
# ax.set_ylabel("Percent positive for flu A")
# plt.tight_layout()
# plt.savefig('Figures/RSV_PercentPositive_Region_Highlight_ppt.png',dpi=300)




############### Plotting KPSC data ###############
## Incidence line plots
# fig, axes = plt.subplots(3,2,figsize=(13.3,7.5),sharex=True)
# kpsc_positive_test_plot(axes[0,0],pathogen="Metapneumovirus", hospitalizations=True, incidence=True, legend=False,aggregation="Month",color='#648FFF')
# kpsc_positive_test_plot(axes[1,0],pathogen="Adenovirus", hospitalizations=True, incidence=True, legend=False,aggregation="Month",color='#648FFF')
# kpsc_positive_test_plot(axes[2,0],pathogen="Parainfluenza 3", hospitalizations=True, incidence=True, legend=False,aggregation="Month",color='#648FFF')
# kpsc_positive_test_plot(axes[0,1],pathogen="Metapneumovirus",AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, hospitalizations=True, incidence=True, legend=False,aggregation="Month")
# kpsc_positive_test_plot(axes[1,1],pathogen="Adenovirus",AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, hospitalizations=True, incidence=True, legend=False,aggregation="Month")
# kpsc_positive_test_plot(axes[2,1],pathogen="Parainfluenza 3",AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, hospitalizations=True, incidence=True, legend=True,aggregation="Month")
# plt.tight_layout()
# plt.savefig('Figures/KPSC_data_sort_of_interesting_slide.png',dpi=300)

## Cumulative seasons plot
# fig, axes = plt.subplots(6,3,sharex=True,figsize=(6.5,8.5))
# for row,pathogen in enumerate(["RSV","Influenza_A","Influenza_B","Metapneumovirus","Adenovirus","Parainfluenza 3"]):
#     for col,typ in enumerate([[False,False],[False,True],[True,True]]):
#         season_plot(axes[row,col],pathogen,typ[0],typ[1])
# axes[5,1].set_xticks(range(8),[year for year in range(2015,2023)])
# for i in range(6):
#     for j in range(1,3):
#         axes[i,j].set_yticks([],[])
# axes[0,0].set_title("Cases")
# axes[0,1].set_title("Age group share")
# axes[0,2].set_title("Adjusted for population")
# axes[0,0].set_ylabel("RSV")
# axes[1,0].set_ylabel("Influenza A")
# axes[2,0].set_ylabel("Influenza B")
# axes[3,0].set_ylabel("Metapneumovirus")
# axes[4,0].set_ylabel("Adenovirus")
# axes[5,0].set_ylabel("Parainfluenza 3")
# fig.suptitle("Cumulative cases by respiratory season")
# plt.tight_layout()
# plt.savefig("Figures/cumulative_seasons_all.png",dpi=300)

############### Processing KPSC data into time series of test-confirmed cases ###############

# # # with SAS7BDAT('Data/Raw/KPSC/testing.sas7bdat') as f:
# # #     test_data = f.to_data_frame()

# with SAS7BDAT('Data/Raw/KPSC/clinical_20241202.sas7bdat') as f:
#     clinical_data = f.to_data_frame()

# # # save random sample of clinical data
# # # clinical_data.sample(10000).to_csv('Data/Processed/KPSC_clinical_sample.csv',index=False)
# # # # load
# # # clinical_data = pd.read_csv('Data/Processed/KPSC_clinical_sample.csv')
# # # # date is 1st of October of each year (in YEAR column), plus dx_days
# clinical_data["Date"] = pd.to_datetime(clinical_data["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(clinical_data["dx_days"],unit='D')

# clinical_data["Year"] = pd.to_datetime(clinical_data["YEAR"].astype(int).astype(str) + '-10-01')
# # # sort by age in months, then translate into age groups
# # AGE_GROUPS = [range(0,3), range(3,12),range(12,5*12),range(5*12,18*12),range(18*12,40*12),range(40*12,65*12),range(65*12,120*12)]
# # AGE_GROUP_NAMES = ['<3m','3-11m','1-4y','5-17y','18-39y','40-64y','>=65y']
# # assign_age_group = lambda x: AGE_GROUP_NAMES[np.argmax([x in group for group in AGE_GROUPS])]
# # clinical_data["AGE_GROUP"] = clinical_data["age_in_mo"].apply(assign_age_group)
# # # get proportion of clinical cases with flu_vac == 1 in each month, for each age group.
# vaccination_proportion = clinical_data.groupby(["Year","age"])["flu_vac"].mean().unstack()
# # vaccination_proportion = vaccination_proportion.reindex(pd.period_range(start=vaccination_proportion.index.min(),end=vaccination_proportion.index.max(),freq='Y'))
# # vaccination_proportion = vaccination_proportion.fillna(0)
# # #reorder columns to match order in AGE_GROUP_NAMES
# # vaccination_proportion = vaccination_proportion[AGE_GROUP_NAMES]

# # # save to csv
# vaccination_proportion.to_csv('Data/Processed/KPSC_vaccinated_proportion_year_ages_by_season.csv')
# # #load
# # vaccination_proportion = pd.read_csv('Data/Processed/KPSC_vaccinated_proportion_ages_by_season.csv',index_col=0)
# vax2022 = vaccination_proportion.loc['2022-10-01']
# age_pops = pd.read_csv('Data/Raw/US_Census_population_by_age.csv',dtype=int)
# age_pop = age_pops.groupby('AGE')['POPESTIMATE2022'].sum()
# # get rid of age over 90
# age_pop = age_pop[age_pop.index < 90]
# # match vaccination proportion to age population by age
# vax2022 = vax2022.reindex(age_pop.index)
# # population weighted average of vaccination proportion
# vax2022 = (vax2022*age_pop).sum()/age_pop.sum()
# print(vax2022)

# print(vaccination_proportion[['<3m','3-11m']])
# # plot
# fig, ax = plt.subplots(figsize=(6.5,6.5))
# hsv_colors = colormaps.hsv(-0.02+np.arange(7)/7)
# hsv_colors[3] = colormaps.hsv((3/7)+0.04)
# vaccination_proportion.plot(ax=ax,color=hsv_colors)
# # # plot dashed vertical lines at october each year
# # for year in range(9):
# #     ax.axvline(12*year,color='black',alpha=0.3)
# ax.set_ylim(0,1)
# # # x labels based on years - first index is october 2015
# # ax.set_xticks(range(3,len(vaccination_proportion),12),[year for year in range(2016,2024)])
# # label seasons, e.g. 2015/16, 2016/17, etc.
# ax.set_xticks(range(8),[f"20{year}/{year+1}" for year in range(15,23)])
# ax.set_xlim(0,7)
# ax.set_title("Proportion of clinical cases with recent (<1y) flu vaccine")
# ax.set_ylabel("Proportion")
# ax.set_xlabel("Season")
# plt.savefig('Figures/KPSC_vaccinated_proportion_age_by_season.png',dpi=300)

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

# # # filter test data to only include tests with StudyID matching a value in hospitalizations
# test_data = test_data[test_data["StudyID"].isin(hospitalizations["StudyID"])]
# test_data["Date"] = pd.to_datetime(test_data["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(test_data["lab_days"],unit='D')
# # # # save test data
# test_data.to_csv('Data/Processed/KPSC_hospitalized_tests.csv',index=False)
# # # # load test data
# # test_data = pd.read_csv('Data/Processed/KPSC_hospitalized_tests.csv')
# # positive_tests = test_data[test_data["result_val"] == 'Positive']

# # # # find date of from hospitalizations for each study ID and match to positive tests
# positive_tests = hospitalizations.merge(positive_tests[["StudyID","Date","pathogen","lab_type","lab_days"]],on="StudyID",how="left")
# # # # rename columns
# positive_tests = positive_tests.rename(columns={"Date_x":"Hospitalization date","Date_y":"Test date"})
# # # # keep only rows where Hospitalization date is within 14 days of Test date
# positive_tests = positive_tests[np.abs((pd.to_datetime(positive_tests["Hospitalization date"]) - pd.to_datetime(positive_tests["Test date"])).dt.days) <= 14]

# # # save to csv
# positive_tests.to_csv('Data/Processed/KPSC_positive_matched_hospitalizations.csv',index=False)

## Separating out individual pathogen data from positive matched hospitalizations
# AGE_GROUP_NAMES = ['<3m','3-11m','1-4y','5-17y','18-39y','40-64y','>=65y']
# aggregation = None
# fig, ax = plt.subplots()
# for incidence in [False, True]:
#     for AGE_GROUPS in [None,[range(0,3), range(3,12),range(12,5*12),range(5*12,18*12),range(18*12,40*12),range(40*12,65*12),range(65*12,90*12)]]:
#         for pathogen in ["RSV","Influenza_A","Influenza_B","Metapneumovirus","Adenovirus","Parainfluenza 3"]:
#             kpsc_positive_test_plot(ax,pathogen=pathogen,AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, hospitalizations=True, incidence=incidence, legend=True,aggregation=aggregation, save_data=True)
