import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sas7bdat import SAS7BDAT
from matplotlib import cm as colormaps
from Parameters.census_population import *
from plotting import *
import pickle

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

# with SAS7BDAT('Data/Raw/KPSC/clinical_20241202.sas7bdat') as f:
#     clinical_data = f.to_data_frame()

# with SAS7BDAT('Data/Raw/KPSC/testing.sas7bdat') as f:
#     test_data = f.to_data_frame()

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
