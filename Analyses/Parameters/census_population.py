## Test population parameters
## Rough estimates of population size and age distribution

import jax.numpy as jnp
import pandas as pd

## Population size and age distribution

# SC_COUNTY_POPS = {"Los Angeles":10081570,"Kern":887641,"Orange":3168044,"San Diego":3316073,"San Bernardino":2149031,"Riverside":2411439,"Ventura":847263}

POP_SIZE = 4024493 # KPSC insured population size in 2015

AGE_POP = jnp.asarray(pd.read_csv('Data/Processed/US_Census_population_by_age.csv', header=None).values) # US in 2022
AGE_GROUPS = [
    jnp.arange(3),
    jnp.arange(3,12),
    jnp.arange(12,5*12),
    jnp.arange(5*12,18*12),
    jnp.arange(18*12,40*12),
    jnp.arange(40*12,65*12),
    jnp.arange(65*12,90*12)]
# AGE_GROUPS = [range(0,6),range(6,12),range(12,2*12),range(2*12,3*12),range(3*12,4*12),range(4*12,5*12),range(5*12,10*12),range(10*12,15*12),range(15*12,20*12),range(20*12,25*12),range(25*12,30*12),range(30*12,35*12),range(35*12,40*12),range(40*12,45*12),range(45*12,50*12),range(50*12,55*12),range(55*12,60*12),range(60*12,65*12),range(65*12,70*12),range(70*12,75*12),range(75*12,80*12),range(80*12,85*12),range(85*12,90*12),range(90*12,95*12),range(95*12,100*12)]
MEDIAN_AGE = [jnp.median(group) for group in AGE_GROUPS]
# AGE_GROUP_NAMES = [
#     'Newborns',
#     'Infants',
#     'Young children',
#     'Older children',
#     'Young adults',
#     'Middle-aged adults',
#     'Older adults']
AGE_GROUP_NAMES = ['<3m','3-11m','1-4y','5-17y','18-39y','40-64y','>=65y']
# AGE_GROUP_NAMES = ['0y','1y','2y','3y','4y','5y','10y','15y','20y','25y','30y','35y','40y','45y','50y','55y','60y','65y','70y','75y','80y','85y','90y','95y','100y']

# Age group proportions for California in 2022
AGE_PROPORTION = jnp.array([jnp.sum(AGE_POP[group]) for group in AGE_GROUPS])/jnp.sum(AGE_POP)
# age_array = [x for xs in [[i]*AGE_POP[i] for i in group] for x in xs]
# print(MEDIAN_AGE)
# print([jnp.median(jnp.array([x for xs in [[i]*int(AGE_POP[i]) for i in group] for x in xs])) for group in AGE_GROUPS])
CENSUS_AGE_POP = AGE_PROPORTION*POP_SIZE

# Rate of aging out of each age group. Last rate informed by US life expectancy at age 65.
# AGING_RATE = 1/jnp.array([3/12*365,9/12*365,4*365,13*365,22*365,25*365,18.35*365])
# Same with last rate informed by US life expectancy at birth. This leads to a more stable age distribution and population size.
AGING_RATE = 1/jnp.array([3/12*365,9/12*365,4*365,13*365,22*365,25*365,12.43*365])
# AGING_RATE = 1/jnp.array([len(group)/12*365 for group in AGE_GROUPS])

# Birth rate (for California in 2022)
# BIRTH_RATE = 3.99e5/(POP_SIZE*365)
# AGING_RATE = BIRTH_RATE/AGE_PROPORTION # Stable population distribution