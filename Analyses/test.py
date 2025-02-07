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

x = [281.19056487083435, 924.5615150928497, 1768.5358440876007]
# fit linear model
x = np.array(x)
y = np.array([1,2,3])
m, b = np.polyfit(np.log(y),np.log(x), 1)
print(m,b)
