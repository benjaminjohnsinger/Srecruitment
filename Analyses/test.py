## Brainstorming for susceptible recruitment project
## Code to explore how susceptibles recruitment affects outbreak dynamics
## BJS August 2024

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
# import pandas as pd
# import itertools as it
# from plotting import *
# from math import comb
from utils import *
# from sas7bdat import SAS7BDAT
import pickle
from scipy.optimize import curve_fit
import time

all_bounds = np.array([[0,1e-2], # WANE
[0,0.5], # SEASONALITY
[0,1], # OFFSET
[0,0.5], # BETA
[0,1e-10], # IMPORT_RATE
[0,1], # S_REL - immunity after second infection above minimum
[0,1], # S_REL - immunity after first infection above minimum
[0,1], # S_REL - relative infection and disease immunity after first infection
[0,0.1], # P_OBS
[0,1], # AGE_OBS - young_immunity
[0,1], # AGE_OBS - old_immunity
[0,1]]) # AGE_OBS - young_old

start = time.time()
lh_sampler = sp.stats.qmc.LatinHypercube(d=12, seed=250212)
lh_samples = lh_sampler.random(300)
lh_samples_scaled = sp.stats.qmc.scale(lh_samples, all_bounds[:,0], all_bounds[:,1])
print(time.time()-start)
print(lh_samples_scaled)