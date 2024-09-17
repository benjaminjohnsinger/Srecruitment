import numpy as np

## Load data
# Load synthetic contact matrices from Prem et al. 2021
PREM_HOME = np.genfromtxt('Data/Processed/contact_matrices/Prem_contact_USA_home.csv', delimiter=',')
PREM_WORK = np.genfromtxt('Data/Processed/contact_matrices/Prem_contact_USA_work.csv', delimiter=',')
PREM_SCHOOL = np.genfromtxt('Data/Processed/contact_matrices/Prem_contact_USA_school.csv', delimiter=',')
PREM_OTHERS = np.genfromtxt('Data/Processed/contact_matrices/Prem_contact_USA_others.csv', delimiter=',')
PREM_ALL = np.genfromtxt('Data/Processed/contact_matrices/Prem_contact_USA_all.csv', delimiter=',')

# Load population by each age in months from 0 to 1199
AGE_POP = np.genfromtxt('Data/Processed/KP_population_by_age_FAKE.csv', delimiter=',')
AGE_POP_norm = AGE_POP/np.sum(AGE_POP)

## Define matrix to map between age groups
# Define age groups
PREM_AGE_GROUPS = [range(i*5*12,(i+1)*5*12) for i in range(15)]
PREM_AGE_GROUPS.append(range(75*12,100*12))
KP_AGE_GROUPS = [range(3),range(3,12),range(12,5*12),range(5*12,18*12),range(18*12,40*12),range(40*12,65*12),range(65*12,100*12)]

# Create matrix of population overlap between age groups
AGE_MAP = np.zeros((7,16))
for age in range(1200):
    AGE_MAP[np.where([age in group for group in KP_AGE_GROUPS])[0][0],np.where([age in group for group in PREM_AGE_GROUPS])[0][0]] += AGE_POP_norm[age]
# Columns sum to 1
AGE_MAP = AGE_MAP/np.sum(AGE_MAP,axis=0)
# Rows sum to 1
AGE_INC = AGE_MAP/np.sum(AGE_MAP,axis=1)[:,np.newaxis]

## Transform contact matrices to KP age groups
KP_HOME = np.dot(np.dot(AGE_INC,PREM_HOME),AGE_MAP.T)
KP_WORK = np.dot(np.dot(AGE_INC,PREM_WORK),AGE_MAP.T)
KP_SCHOOL = np.dot(np.dot(AGE_INC,PREM_SCHOOL),AGE_MAP.T)
KP_OTHERS = np.dot(np.dot(AGE_INC,PREM_OTHERS),AGE_MAP.T)
KP_ALL = np.dot(np.dot(AGE_INC,PREM_ALL),AGE_MAP.T)

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

## Save contact matrices
np.savetxt('Data/Processed/contact_matrices/KP_contact_home_FAKE.csv',KP_HOME,delimiter=',')
np.savetxt('Data/Processed/contact_matrices/KP_contact_work_FAKE.csv',KP_WORK,delimiter=',')
np.savetxt('Data/Processed/contact_matrices/KP_contact_school_FAKE.csv',KP_SCHOOL,delimiter=',')
np.savetxt('Data/Processed/contact_matrices/KP_contact_others_FAKE.csv',KP_OTHERS,delimiter=',')
np.savetxt('Data/Processed/contact_matrices/KP_contact_all_FAKE.csv',KP_ALL,delimiter=',')