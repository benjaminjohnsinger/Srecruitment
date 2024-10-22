## Test population parameters
## Rough estimates of population size and age distribution

import numpy as np

## Population size and age distribution
POP_SIZE = 3.9e7 # California population in 2022
AGE_POP = np.genfromtxt('Data/Processed/US_Census_population_by_age.csv', delimiter=',') # US in 2022
KP_AGE_GROUPS = [
    range(3),
    range(3,12),
    range(12,5*12),
    range(5*12,18*12),
    range(18*12,40*12),
    range(40*12,65*12),
    range(65*12,100*12)]
AGE_GROUP_NAMES = ['Newborns','Infants','Young children','Older children','Young adults','Middle-aged adults','Older adults']
AGE_PROPORTION = np.array([np.sum(AGE_POP[group]) for group in KP_AGE_GROUPS])/np.sum(AGE_POP)
KP_AGE_POP = AGE_PROPORTION*POP_SIZE
NAG = len(KP_AGE_GROUPS)

# Birth rate (for California in 2022)
BIRTH_RATE = 3.99e5/(POP_SIZE*12)
AGING_RATE = BIRTH_RATE/AGE_PROPORTION # Stable population distribution