## Brainstorming for susceptible recruitment project
## Code to explore how susceptibles recruitment affects outbreak dynamics
## BJS August 2024

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
# import itertools as it
# from plotting import *
# from math import comb
from utils import *
from sas7bdat import SAS7BDAT
import pickle
# from scipy.optimize import curve_fit
# import time
# # import corner

# with SAS7BDAT('Data/Raw/KPSC/demographics.sas7bdat') as f:
#     demography = f.to_data_frame()
# # pivot table to just give number in each age group in each year
# demography = demography.pivot_table(index='age',columns='YEAR',values='n',aggfunc='sum')
# # save to csv
# demography.to_csv('Data/Processed/ages_for_Noah.csv')

# print(date_to_t("2010-10-01"))

i=0
for pathogen in ["RSV","InfluenzaA","InfluenzaB","Metapneumovirus","Adenovirus","Parainfluenza3"]:
    for option1 in ["X","setimport"]:
        i+=1
        print("\""+pathogen,"250325","FlexStepwise",option1,"flexage",str(0.01),"15 1 0.7\"")
print(i)

# infectious_contact = np.genfromtxt("Data/Processed/infectious_contact_rsv.csv",delimiter=',',dtype=np.float64)
# import_contact = np.genfromtxt("Data/Processed/import_contact_rsv.csv",delimiter=',',dtype=np.float64)

# print(np.median(infectious_contact/import_contact,axis=0))
# print(np.sum(infectious_contact,axis=0)/np.sum(import_contact,axis=0))
# print(np.max(infectious_contact,axis=0)/np.max(import_contact,axis=0))
# print(np.mean(infectious_contact,axis=0)/np.mean(import_contact,axis=0))
# print(np.median(infectious_contact,axis=0)/np.median(import_contact,axis=0))

# files = [
# "InfluenzaB_deFBfl250304"
# ]

# for filefraction in files:
#     with open("Data/Processed/DE_opt_"+filefraction+".pickle","rb") as f:
#         opt = pickle.load(f)
#     print(filefraction,opt.x)

# with open("Data/Processed/results250303/DE_opt_InfluenzaA_deFluAmo250228.pickle","rb") as f:
#     opt = pickle.load(f)

# print(opt)

# with open("Data/Processed/mcmc_trajectory_from_FluA_DE.pickle","rb") as f:
#     mcmc = pickle.load(f)

# # corner plot
# variables = ["WANE","SEASONALITY","OFFSET","BETA","IMPORT_RATE","EXTRA_IMMUNITY","FIRST_IMMUNITY","FIRST_DIS_INF_FACTOR","P_OBS","OBS_AGE_YOUNG","OBS_AGE_OLD","OBS_AGE_YOUNG_OLD"]
# corner.corner(mcmc,labels=variables,show_titles=True)
# plt.savefig("Figures/Corner_plot_FluA_MCMC_from_DE.png")