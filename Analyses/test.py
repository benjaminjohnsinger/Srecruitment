## Brainstorming for susceptible recruitment project
## Code to explore how susceptibles recruitment affects outbreak dynamics
## BJS August 2024

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import itertools as it
from plotting import *
from math import comb

w = np.array([0.   , 0.12166667, 0.   ])
print(np.mean(w))
print(np.sum(w==0.0))
print(np.mean(w)*len(w)/np.sum(w!=0))


# N=5
# x = range(N,0,-(N//4+1))
# for i in x:
#     print(i)
# z = ["x","y"]
# print([i=="x" for i in z])

# for i in it.product(range(3),repeat=3):
#     print(i)
#     for j,p in enumerate(i):
#         print(j,p)

# def f1(t):
#     return 0.5*np.sin(2*np.pi*t/52)

# census_data = pd.read_csv('Data/Raw/US_Census_population_by_age.csv', delimiter=',')
# AGE_POP = np.repeat(census_data.loc[(census_data["SEX"]==0) & (census_data["AGE"]<=100),"POPESTIMATE2022"],12)/12
# AGE_POP = np.array(AGE_POP.values)
# np.savetxt('Data/Processed/US_Census_population_by_age.csv',AGE_POP,delimiter=',')

# ## SIRS model
# T = 1300

# # Initialize
# S = np.zeros((T,3))
# I = np.zeros((T,3))
# R = np.zeros((T,3))

# # Initial conditions
# S[0] = 0.99
# I[0] = 0.01
# R[0] = 0

# # Parameters
# beta = 0.5
# gamma = 0.3
# nu = [0.005,0.01,0.03]
# seasonality = 0.05

# # Run the model
# for t in range(1, T):
#     for i in range(3):
#         S[t,i] = S[t-1,i] - beta*S[t-1,i]*I[t-1,i]*(1+seasonality*np.sin(2*np.pi*t/52)) + nu[i]*R[t-1,i]
#         I[t,i] = I[t-1,i] + beta*S[t-1,i]*I[t-1,i]*(1+seasonality*np.sin(2*np.pi*t/52)) - gamma*I[t-1,i]
#         R[t,i] = R[t-1,i] + gamma*I[t-1,i] - nu[i]*R[t-1,i]

# # Plot the results
# plt.plot(I[:,0], label='Low waning')
# plt.plot(I[:,1], label='Medium waning')
# plt.plot(I[:,2], label='High waning')
# # Plot vertical line every two years
# for i in range(0, T//52+1):
#     plt.axvline(x=52*i, color='lightgray', linewidth=0.5)
#     if i%2 == 0:
#         plt.axvline(x=52*i, color='lightgray', linewidth=1)
#     if i%5 == 0:
#         plt.axvline(x=52*i, color='gray', linewidth=1)
# plt.legend()
# plt.show()
