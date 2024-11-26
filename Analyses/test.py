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
# import pickle

def rectangular_beta(x,a,b,theta):
    return theta*sp.special.gamma(a+b)/(sp.special.gamma(a)*sp.special.gamma(b))*x**(a-1)*(1-x)**(b-1) + 1-theta

def piecewise(x,young_immunity,old_immunity,young_old):
    return np.maximum(0,np.minimum(1,1+young_old)*young_immunity**x)+np.maximum(0,np.minimum(1,1-young_old)*old_immunity**(6-x))

y = np.array([0.5,0.3,0.2,0.01,0.01,0.01,0.3])/0.5
# medians = np.array([1/1080,7/1080,36/1080,140/1080,351/1080,631/1080,877/1080])
medians=np.arange(7)
def distance(params):
    # a, b, theta = params
    yi, oi, yo = params
    # function = sp.stats.beta(params[0],params[1]).pdf
    # function = lambda x: a*b*x**(a-1)*(1-x**a)**(b-1)
    # function = lambda x: sp.stats.betabinom(a,b,6).pmf(x)/np.max(sp.stats.betabinom(a,b,6).pmf(np.arange(7)))
    # function = lambda x: rectangular_beta(x,a,b,theta)
    function = lambda x: piecewise(x,yi,oi,yo)
    values = [function(median) for median in medians]
    # values = [function(i) for i in range(7)]
    return np.sqrt(np.sum((values-y)**2))

print(piecewise(medians,0.54,-0.01,0.4))

result = sp.optimize.minimize(distance,[0.5,0.5,0.4],method='Nelder-Mead')
print(result.x)
# a,b,theta = result.x
yi,oi,yo = result.x
# print(sp.stats.beta(result.x[0],result.x[1]).pdf(medians))
# ksf = lambda x: a*b*x**(a-1)*(1-x**a)**(b-1)
# function = lambda x: sp.stats.betabinom(a,b,6).pmf(x)/np.max(sp.stats.betabinom(a,b,6).pmf(np.arange(7)))
# print([function(i) for i in range(7)])
# print([rectangular_beta(i,a,b,theta) for i in medians])
print([piecewise(i,yi,oi,yo) for i in medians])

# plt.plot(np.linspace(0,1,1000),sp.stats.beta(result.x[0],result.x[1]).pdf(np.linspace(0,1,1000)),color='k')
# plt.plot(np.linspace(0,1,1000),ksf(np.linspace(0,1,1000)))
# plt.plot(np.arange(7),[function(i) for i in range(7)],color='black')
# plt.plot(np.linspace(0,1,1000),[rectangular_beta(i,a,b,theta) for i in np.linspace(0,1,1000)],color='black')
plt.plot(np.linspace(0,6,1000),[piecewise(i,yi,oi,yo) for i in np.linspace(0,6,1000)],color='black')
plt.scatter(medians,y,color='#648FFF')
# plt.scatter(medians,sp.stats.beta(result.x[0],result.x[1]).pdf(medians),color='red',marker='+',zorder=10)
# plt.scatter(medians,[rectangular_beta(i,a,b,theta) for i in medians],color='red',marker='+',zorder=10)
plt.scatter(medians,[piecewise(i,yi,oi,yo) for i in medians],color='red',marker='+',zorder=10)
# plt.xticks(np.arange(7),[f'{i+1}/1080' for i in [0,6,35,139,350,630,876]])
# plt.ylim([-0.05,1.05])
plt.savefig('Figures/discretized_piecwise_exp_fit_to_normalized_age_obs_guess_RSV.png',dpi=300)


# incubation_median_RSV = 4.4
# incubation_dispersion_RSV = 1.24
# incubation_distribution_RSV = sp.stats.lognorm(np.log(incubation_dispersion_RSV),scale=incubation_median_RSV)

# admittance_logmean_RSV = 1.85
# admittance_logsd_RSV = 0.762
# admittance_distribution_RSV = sp.stats.lognorm(admittance_logsd_RSV,scale=np.exp(admittance_logmean_RSV))

# incubation_median_fluA = 1.4
# incubation_dispersion_fluA = 1.51
# incubation_distribution_fluA = sp.stats.lognorm(np.log(incubation_dispersion_fluA),scale=incubation_median_fluA)

# admittance_mu_flu = 1.92
# admittance_sigma_flu = 0.914
# admittance_Q_flu = 0.126
# admittance_distribution_flu = sp.stats.gengamma(admittance_Q_flu,admittance_mu_flu,admittance_sigma_flu)
# print(admittance_distribution_flu.ppf([0.25,0.5,0.75]))


# # generate 100k samples of incubation plus admittance
# np.random.seed(241125)
# N = int(1e6)
# infection_to_admittance = incubation_distribution_RSV.rvs(N) + admittance_distribution_RSV.rvs(N)
# # shape, loc, scale = sp.stats.lognorm.fit(infection_to_admittance)
# # fit = sp.stats.lognorm(shape, loc=loc, scale=scale)
# # plt.hist(infection_to_admittance,bins=1000,color='silver',density=True)
# # plt.plot(np.linspace(0,50,1000),fit.pdf(np.linspace(0,50,1000)))
# # plt.xlim([0,50])
# # plt.show()

# x=np.linspace(0,30.44,1000)
# p_this_month = [np.mean(day+infection_to_admittance<=30.44) for day in x]
# p_next_month = [np.mean(day+infection_to_admittance>30.44) for day in x]
# p_two_months = [np.mean(day+infection_to_admittance>60.88) for day in x]
# plt.plot(x,p_this_month,color='black')
# plt.plot(x,p_next_month,color='black',linestyle='--')
# plt.plot(x,p_two_months,color='black',linestyle=':')
# print(np.mean(p_next_month))
# print(np.mean(p_two_months))
# # annotate with means
# plt.text(18,np.mean(p_this_month)+0.01,f'{np.mean(p_this_month):.3f}',color='black')
# plt.text(18,np.mean(p_next_month)+0.01,f'{np.mean(p_next_month):.3f}',color='black')
# plt.text(18,np.mean(p_two_months)+0.01,f'{np.mean(p_two_months):.3f}',color='black')
# plt.legend(['This month','Next month','Two months'])
# plt.xlabel('Day of infection')
# plt.ylabel('Probability of admittance within time frame')
# plt.tight_layout()
# plt.savefig('Figures/RSV_admittance_within_time_frame.png',dpi=300)

# samples = real_to_p(np.random.normal(-1,1.5,10000))
# # plot density
# fig, ax = plt.subplots(1,1,figsize=(6.5,6.5))
# ax.hist(samples,bins=50,density=True,color='silver')
# ax.set_title('Density of samples')
# ax.set_xlabel('Sample value')
# ax.set_ylabel('Density')
# plt.tight_layout()
# plt.show()



# means = np.array([0.1,0.2,0.3])
# cov = np.array([[0.01,0,0],[0,0.01,0],[0,0,0.01]])

# move = np.random.multivariate_normal(np.zeros(3),cov)
# print(move)
# print(real_to_p(p_to_real(means)+move))

# print(np.sum(np.array([0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00050052e+04,8.30484011e+03,1.87321224e+03,5.58351161e+01,4.28455975e-01,1.67069413e-03,6.20424525e-06,2.92637579e+02,2.74118090e+02,7.09693608e+01,1.37167853e+01,1.07012980e+01,1.06713644e+01,1.06579077e+01,1.84619366e+03,4.77663778e+03,5.15030817e+03,1.04305304e+04,1.78012040e+04,1.91596837e+04,1.07178628e+04,2.78841796e+01,6.68586045e+01,7.16395379e+01,2.42260553e+02,3.67618026e+02,3.28200210e+02,1.11709479e+02,1.97472715e+03,2.92368497e+04,2.42394648e+05,9.36936487e+05,1.56437910e+06,1.57971969e+06,7.44055459e+05,1.84081403e+01,2.05811181e+02,1.66926286e+03,1.14129778e+04,1.70822361e+04,1.42951212e+04,3.97385592e+03,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00])))
# print(np.sum(np.array([14164.85592336,42865.11544626,251230.04007045,959091.80710918,1599641.28439574,1613513.36343163,758869.5449999])))

# contact_matrix = (8.8/(10*15.85))*np.genfromtxt('Data/Processed/contact_matrices/Pitzer_contact_all_US_Census.csv', delimiter=',', dtype=np.float64)
# # first eigenvalue
# eigenvalues = np.linalg.eigvals(contact_matrix)
# print(np.max(eigenvalues))

# with SAS7BDAT('Data/Raw/KPSC/clinical.sas7bdat') as f:
#     clinical_data = f.to_data_frame()

# # sample 10k rows and save to csv
# clinical_data.sample(10000).to_csv('Data/Processed/KPSC_clinical_sample.csv',index=False)

# # bar chart of CODE
# code_counts = clinical_data["CODE"].value_counts()
# code_counts = code_counts.sort_values(ascending=False)
# print(code_counts)

# fig, ax = plt.subplots(1,1,figsize=(6.5,6.5))
# ax.bar(range(len(code_counts)),code_counts,color='silver')
# ax.set_xticks(range(len(code_counts)))
# ax.set_xticklabels(code_counts.index)
# ax.set_title('Number of cases by code')
# ax.set_ylabel('Number of cases')
# plt.tight_layout()
# plt.savefig('Figures/KPSC_code_counts.png')

# age_group_dict = {'Infants':range(1), 'Young children':range(1,5), 'Older children':range(5,18), 'Young adults':range(18,40), 'Middle-aged adults':range(40,65), 'Older adults':range(65,100)}

# rota_vax_by_age = np.zeros(6)
# flu_vax_by_age = np.zeros(6)

# for age_group_n,age_group in enumerate(age_group_dict.keys()):
#     rota_vax_by_age[age_group_n] = np.mean(clinical_data.loc[[age in age_group_dict[age_group] for age in clinical_data["age"]],"rota_vac"])
#     flu_vax_by_age[age_group_n] = np.mean(clinical_data.loc[[age in age_group_dict[age_group] for age in clinical_data["age"]],"flu_vac"])
# rota_vax_by_age = [7.32078280e-01,9.06606591e-01,6.04187094e-01,7.92712761e-04,3.60311619e-05,1.55745789e-05]
# flu_vax_by_age = [0.38035097,0.67629991,0.52047387,0.50746089,0.65703239,0.85448068]

# age_names = ['<1y','1-4y','5-17y','18-39y','40-64y','>=65y']


# fig, ax = plt.subplots(2,1,figsize=(6.5,6.5),sharex=True)
# ax[0].bar(range(6),rota_vax_by_age,color='silver')
# # ax[0].set_xticks(range(6))
# # ax[0].set_xticklabels(age_group_dict.keys())
# ax[0].set_title('Rotavirus vaccination by age group')
# ax[0].set_ylabel('Proportion vaccinated')

# ax[1].bar(range(6),flu_vax_by_age,color='silver')
# ax[1].set_xticks(range(6))
# ax[1].set_xticklabels(age_names)
# ax[1].set_title('Influenza vaccination by age group')
# ax[1].set_ylabel('Proportion vaccinated')

# plt.tight_layout()
# plt.savefig('Figures/KPSC_vaccination_by_age_group.png')



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
