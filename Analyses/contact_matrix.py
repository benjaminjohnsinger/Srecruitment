# import jax.numpy as jnp
import numpy as np
## Load data
# Load synthetic contact matrices from Prem et al. 2021
PREM_HOME = np.genfromtxt('Data/Processed/contact_matrices/Prem_contact_USA_home.csv', delimiter=',')
PREM_WORK = np.genfromtxt('Data/Processed/contact_matrices/Prem_contact_USA_work.csv', delimiter=',')
PREM_SCHOOL = np.genfromtxt('Data/Processed/contact_matrices/Prem_contact_USA_school.csv', delimiter=',')
PREM_OTHERS = np.genfromtxt('Data/Processed/contact_matrices/Prem_contact_USA_others.csv', delimiter=',')
PREM_ALL = np.genfromtxt('Data/Processed/contact_matrices/Prem_contact_USA_all.csv', delimiter=',')

# Load population by each age in months from 0 to 1199
AGE_POP = np.genfromtxt('Data/Processed/US_Census_population_by_age.csv', delimiter=',')
AGE_POP_norm = AGE_POP/np.sum(AGE_POP)

## Define matrix to map between age groups
# Define age groups
PREM_AGE_GROUPS = [range(i*5*12,(i+1)*5*12) for i in range(15)]
PREM_AGE_GROUPS.append(range(75*12,100*12))
# KP_AGE_GROUPS = [np.arange(0,3),np.arange(3,12),np.arange(12,5*12),np.arange(5*12,8*12),np.arange(8*12,40*12),np.arange(40*12,65*12),np.arange(65*12,100*12)]
from Parameters.census_population import AGE_GROUPS as KP_AGE_GROUPS
PITZER_AGE_GROUPS = [range(0,6),range(6,12),range(12,2*12),range(2*12,3*12),range(3*12,4*12),range(4*12,5*12),range(5*12,10*12),range(10*12,15*12),range(15*12,20*12),range(20*12,25*12),range(25*12,30*12),range(30*12,35*12),range(35*12,40*12),range(40*12,45*12),range(45*12,50*12),range(50*12,55*12),range(55*12,60*12),range(60*12,65*12),range(65*12,70*12),range(70*12,75*12),range(75*12,80*12),range(80*12,85*12),range(85*12,90*12),range(90*12,95*12),range(95*12,120*12)]
MIKE_AGE_GROUPS = [range(0,4*12),range(4*12,6*12),range(6*12,18*12),range(18*12,120*12)]
NAG = len(KP_AGE_GROUPS)
KP_AGE_POP = np.array([np.sum(AGE_POP[group]) for group in KP_AGE_GROUPS])
KP_AGE_POP = KP_AGE_POP/np.sum(KP_AGE_POP)
from Parameters.census_population import AGE_PROPORTION
print(AGE_PROPORTION, KP_AGE_POP)
# NAG = len(MIKE_AGE_GROUPS)

# Create matrix of population overlap between age groups
AGE_MAP = np.zeros((NAG,16))
for age in range(1200):
    AGE_MAP[np.where([age in group for group in KP_AGE_GROUPS])[0][0],np.where([age in group for group in PREM_AGE_GROUPS])[0][0]] += AGE_POP_norm[age]
# Columns sum to 1
AGE_MAP = AGE_MAP/np.sum(AGE_MAP,axis=0)
# Rows sum to 1
AGE_INC = AGE_MAP/np.sum(AGE_MAP,axis=1)[:,np.newaxis]

## Transform contact matrices to MIKE age groups
KP_HOME = np.dot(np.dot(AGE_INC,PREM_HOME),AGE_MAP.T)
KP_WORK = np.dot(np.dot(AGE_INC,PREM_WORK),AGE_MAP.T)
KP_SCHOOL = np.dot(np.dot(AGE_INC,PREM_SCHOOL),AGE_MAP.T)
KP_OTHERS = np.dot(np.dot(AGE_INC,PREM_OTHERS),AGE_MAP.T)
KP_ALL = np.dot(np.dot(AGE_INC,PREM_ALL),AGE_MAP.T)
# average number of contacts per person per day
avg_contacts = np.sum(KP_ALL.T * KP_AGE_POP)
print(f'Average number of contacts per person per day (Prem et al. 2021, USA, KPSC age groups): {avg_contacts:.2f}')


# ## Plot contact matrices
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(4,2,figsize=(20,10))
# ax[0,0].imshow(PREM_HOME)
# ax[0,0].set_title('Prem home')
# ax[0,1].imshow(KP_HOME)
# ax[0,1].set_title('KP home')
# ax[1,0].imshow(PREM_WORK)
# ax[1,0].set_title('Prem work')
# ax[1,1].imshow(KP_WORK)
# ax[1,1].set_title('KP work')
# ax[2,0].imshow(PREM_SCHOOL)
# ax[2,0].set_title('Prem school')
# ax[2,1].imshow(KP_SCHOOL)
# ax[2,1].set_title('KP school')
# ax[3,0].imshow(PREM_OTHERS)
# ax[3,0].set_title('Prem others')
# ax[3,1].imshow(KP_OTHERS)
# ax[3,1].set_title('KP others')
# plt.show()

# Save contact matrices
np.savetxt('Data/Processed/contact_matrices/KP_contact_home_US_Census.csv',KP_HOME,delimiter=',')
np.savetxt('Data/Processed/contact_matrices/KP_contact_work_US_Census.csv',KP_WORK,delimiter=',')
np.savetxt('Data/Processed/contact_matrices/KP_contact_school_US_Census.csv',KP_SCHOOL,delimiter=',')
np.savetxt('Data/Processed/contact_matrices/KP_contact_others_US_Census.csv',KP_OTHERS,delimiter=',')
np.savetxt('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv',KP_ALL,delimiter=',')