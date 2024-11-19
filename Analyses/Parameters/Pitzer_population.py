## Population to match Pitzer paper

import numpy as np

## Population size and age distribution
POP_SIZE = 3.9e7 # population of california

AGE_POP = np.genfromtxt('Data/Processed/US_Census_population_by_age.csv', delimiter=',') # US in 2022
AGE_GROUPS = [range(0,6),range(6,12),range(12,2*12),range(2*12,3*12),range(3*12,4*12),range(4*12,5*12),range(5*12,10*12),range(10*12,15*12),range(15*12,20*12),range(20*12,25*12),range(25*12,30*12),range(30*12,35*12),range(35*12,40*12),range(40*12,45*12),range(45*12,50*12),range(50*12,55*12),range(55*12,60*12),range(60*12,65*12),range(65*12,70*12),range(70*12,75*12),range(75*12,80*12),range(80*12,85*12),range(85*12,90*12),range(90*12,95*12),range(95*12,100*12)]
MEDIAN_AGE = [np.median(AGE_GROUPS[i]) for i in range(len(AGE_GROUPS))]
AGE_GROUP_NAMES = ['0-5m','6-11m','1y','2y','3y','4y','5y','10y','15y','20y','25y','30y','35y','40y','45y','50y','55y','60y','65y','70y','75y','80y','85y','90y','95y']

# Age group proportions for California in 2022
AGE_PROPORTION = np.array([np.sum(AGE_POP[group]) for group in AGE_GROUPS])/np.sum(AGE_POP)
CENSUS_AGE_POP = AGE_PROPORTION*POP_SIZE

# Rate of aging out of each age group.
AGING_RATE = 1/np.array([len(group)/12*365 for group in AGE_GROUPS])