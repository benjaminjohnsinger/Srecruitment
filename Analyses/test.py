## Brainstorming for susceptible recruitment project
## Code to explore how susceptibles recruitment affects outbreak dynamics
## BJS August 2024

import numpy as np
import matplotlib.pyplot as plt
# import scipy as sp
import pandas as pd
# import itertools as it
# from plotting import *
# from math import comb
from utils import *
# from sas7bdat import SAS7BDAT
import pickle
# from scipy.optimize import curve_fit
# import time
# # import corner

i=0
pathogen = "RSV"
for lockdown in ["Mobility","FlexStepwise"]:
    for option1 in ["X","ni","nb"]:
        if pathogen == "RSV":
            i+=1
            print("\""+pathogen,"250311",lockdown,option1,"X","15 1 0.7\"")

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