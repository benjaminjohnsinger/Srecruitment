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
# from utils import *
# from sas7bdat import SAS7BDAT
import pickle
# from scipy.optimize import curve_fit
# import time
# import corner

incidence = pd.read_csv("Data/Processed/KPSC_Influenza_A_incidence_age_daily.csv",index_col=0)
print(incidence)

# for filefraction in ["InfluenzaA_deFluAmo250228","InfluenzaA_deFluAbig250228","InfluenzaB_deFluBmo250228","RSV_deRSVmv250228","RSV_deRSVfi250228","RSV_deRSVmo250228","RSV_deRSVfl250228"]:
#     with open("Data/Processed/results250303/DE_opt_"+filefraction+".pickle","rb") as f:
#         opt = pickle.load(f)
#     print(filefraction, opt.x)

# with open("Data/Processed/results250303/DE_opt_InfluenzaA_deFluAmo250228.pickle","rb") as f:
#     opt = pickle.load(f)

# print(opt)

# with open("Data/Processed/mcmc_trajectory_from_FluA_DE.pickle","rb") as f:
#     mcmc = pickle.load(f)

# # corner plot
# variables = ["WANE","SEASONALITY","OFFSET","BETA","IMPORT_RATE","EXTRA_IMMUNITY","FIRST_IMMUNITY","FIRST_DIS_INF_FACTOR","P_OBS","OBS_AGE_YOUNG","OBS_AGE_OLD","OBS_AGE_YOUNG_OLD"]
# corner.corner(mcmc,labels=variables,show_titles=True)
# plt.savefig("Figures/Corner_plot_FluA_MCMC_from_DE.png")