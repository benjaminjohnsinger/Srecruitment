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

x = np.array([1,2,3,4,5,6,7])

print(np.tile(x,8))
print(x.repeat(8))
print(x.repeat(8).reshape((-1,8)).T.flatten())