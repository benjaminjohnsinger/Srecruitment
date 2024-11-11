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
from utils import *
from sas7bdat import SAS7BDAT
import pickle

# with SAS7BDAT('Data/Raw/KPSC/clinical.sas7bdat') as f:
#     clinical_data = f.to_data_frame()

age_group_dict = {'Infants':range(1), 'Young children':range(1,5), 'Older children':range(5,18), 'Young adults':range(18,40), 'Middle-aged adults':range(40,65), 'Older adults':range(65,100)}

# rota_vax_by_age = np.zeros(6)
# flu_vax_by_age = np.zeros(6)

# for age_group_n,age_group in enumerate(age_group_dict.keys()):
#     rota_vax_by_age[age_group_n] = np.mean(clinical_data.loc[[age in age_group_dict[age_group] for age in clinical_data["age"]],"rota_vac"])
#     flu_vax_by_age[age_group_n] = np.mean(clinical_data.loc[[age in age_group_dict[age_group] for age in clinical_data["age"]],"flu_vac"])
rota_vax_by_age = [7.32078280e-01,9.06606591e-01,6.04187094e-01,7.92712761e-04,3.60311619e-05,1.55745789e-05]
flu_vax_by_age = [0.38035097,0.67629991,0.52047387,0.50746089,0.65703239,0.85448068]

age_names = ['<1y','1-4y','5-17y','18-39y','40-64y','>=65y']


fig, ax = plt.subplots(2,1,figsize=(6.5,6.5),sharex=True)
ax[0].bar(range(6),rota_vax_by_age,color='silver')
# ax[0].set_xticks(range(6))
# ax[0].set_xticklabels(age_group_dict.keys())
ax[0].set_title('Rotavirus vaccination by age group')
ax[0].set_ylabel('Proportion vaccinated')

ax[1].bar(range(6),flu_vax_by_age,color='silver')
ax[1].set_xticks(range(6))
ax[1].set_xticklabels(age_names)
ax[1].set_title('Influenza vaccination by age group')
ax[1].set_ylabel('Proportion vaccinated')

plt.tight_layout()
plt.savefig('Figures/KPSC_vaccination_by_age_group.png')

# with SAS7BDAT('Data/Raw/KPSC/demographics.sas7bdat') as f:
#     demographics_data = f.to_data_frame()

# age_groups =['<3mo','3-12mo','1-4y','5-17y','18-39y','40-64y','>=65y']
# AGE_GROUP_NAMES = ['Newborns','Infants','Young children','Older children','Young adults','Middle-aged adults','Older adults']

# # print population by year and age group ("age")
# pop_by_age = np.zeros((8,7))
# for year_n,year in enumerate(range(2015,2023)):
#     for age_n,age in enumerate(age_groups):
#         pop_by_age[year_n,age_n] = np.sum(demographics_data.loc[(demographics_data["age"]==age) & (demographics_data["YEAR"]==year),"n"])
# pop_by_age_df = pd.DataFrame(pop_by_age,columns=AGE_GROUP_NAMES,index=range(2015,2023))

# pop_by_age_df.to_csv('Data/Processed/KPSC_population_by_age.csv',index=False)

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
