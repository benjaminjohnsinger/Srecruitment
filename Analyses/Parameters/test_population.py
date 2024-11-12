## Test population parameters
## Rough estimates of population size and age distribution

import numpy as np

## Population size and age distribution
# POP_SIZE = 3.25e7
POP_SIZE = 4e6
AGE_POP = np.genfromtxt('Data/Processed/US_Census_population_by_age.csv', delimiter=',') # US in 2022
KP_AGE_GROUPS = [
    range(3),
    range(3,12),
    range(12,5*12),
    range(5*12,18*12),
    range(18*12,40*12),
    range(40*12,65*12),
    range(65*12,100*12)]
NAG = len(KP_AGE_GROUPS)
MEDIAN_AGE = [np.median(KP_AGE_GROUPS[i]) for i in range(len(KP_AGE_GROUPS))]
AGE_GROUP_NAMES = ['Newborns','Infants','Young children','Older children','Young adults','Middle-aged adults','Older adults']

# Age group proportions for California in 2022
AGE_PROPORTION = np.array([np.sum(AGE_POP[group]) for group in KP_AGE_GROUPS])/np.sum(AGE_POP)
KP_AGE_POP = AGE_PROPORTION*POP_SIZE

# Rate of aging out of each age group. Last rate informed by US life expectancy at age 65.
# AGING_RATE = 1/np.array([3/12*365,9/12*365,4*365,13*365,22*365,25*365,18.35*365])
# Same with last rate informed by US life expectancy at birth. This leads to a more stable age distribution and population size.
AGING_RATE = 1/np.array([3/12*365,9/12*365,4*365,13*365,22*365,25*365,12.43*365])


# Birth rate (for California in 2022)
BIRTH_RATE = 3.99e5/(POP_SIZE*365)
# AGING_RATE = BIRTH_RATE/AGE_PROPORTION # Stable population distribution