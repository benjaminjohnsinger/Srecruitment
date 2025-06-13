import jax.numpy as jnp

## Load data
# Load synthetic contact matrices from Prem et al. 2021
PREM_HOME = jnp.genfromtxt('Data/Processed/contact_matrices/Prem_contact_USA_home.csv', delimiter=',')
PREM_WORK = jnp.genfromtxt('Data/Processed/contact_matrices/Prem_contact_USA_work.csv', delimiter=',')
PREM_SCHOOL = jnp.genfromtxt('Data/Processed/contact_matrices/Prem_contact_USA_school.csv', delimiter=',')
PREM_OTHERS = jnp.genfromtxt('Data/Processed/contact_matrices/Prem_contact_USA_others.csv', delimiter=',')
PREM_ALL = jnp.genfromtxt('Data/Processed/contact_matrices/Prem_contact_USA_all.csv', delimiter=',')

# Load population by each age in months from 0 to 1199
AGE_POP = jnp.genfromtxt('Data/Processed/US_Census_population_by_age.csv', delimiter=',')
AGE_POP_norm = AGE_POP/jnp.sum(AGE_POP)

## Define matrix to map between age groups
# Define age groups
PREM_AGE_GROUPS = [range(i*5*12,(i+1)*5*12) for i in range(15)]
PREM_AGE_GROUPS.append(range(75*12,100*12))
KP_AGE_GROUPS = [range(0,3),range(3,12),range(12,5*12),range(5*12,18*12),range(18*12,40*12),range(40*12,65*12),range(65*12,100*12)]
PITZER_AGE_GROUPS = [range(0,6),range(6,12),range(12,2*12),range(2*12,3*12),range(3*12,4*12),range(4*12,5*12),range(5*12,10*12),range(10*12,15*12),range(15*12,20*12),range(20*12,25*12),range(25*12,30*12),range(30*12,35*12),range(35*12,40*12),range(40*12,45*12),range(45*12,50*12),range(50*12,55*12),range(55*12,60*12),range(60*12,65*12),range(65*12,70*12),range(70*12,75*12),range(75*12,80*12),range(80*12,85*12),range(85*12,90*12),range(90*12,95*12),range(95*12,100*12)]
MIKE_AGE_GROUPS = [range(0,4*12),range(4*12,6*12),range(6*12,18*12),range(18*12,100*12)]
# NAG = len(KP_AGE_GROUPS)
NAG = len(MIKE_AGE_GROUPS)

# Create matrix of population overlap between age groups
AGE_MAP = jnp.zeros((NAG,16))
for age in range(1200):
    AGE_MAP[jnp.where([age in group for group in MIKE_AGE_GROUPS])[0][0],jnp.where([age in group for group in PREM_AGE_GROUPS])[0][0]] += AGE_POP_norm[age]
# Columns sum to 1
AGE_MAP = AGE_MAP/jnp.sum(AGE_MAP,axis=0)
# Rows sum to 1
AGE_INC = AGE_MAP/jnp.sum(AGE_MAP,axis=1)[:,jnp.newaxis]

## Transform contact matrices to MIKE age groups
MIKE_HOME = jnp.dot(jnp.dot(AGE_INC,PREM_HOME),AGE_MAP.T)
MIKE_WORK = jnp.dot(jnp.dot(AGE_INC,PREM_WORK),AGE_MAP.T)
MIKE_SCHOOL = jnp.dot(jnp.dot(AGE_INC,PREM_SCHOOL),AGE_MAP.T)
MIKE_OTHERS = jnp.dot(jnp.dot(AGE_INC,PREM_OTHERS),AGE_MAP.T)
MIKE_ALL = jnp.dot(jnp.dot(AGE_INC,PREM_ALL),AGE_MAP.T)

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
jnp.savetxt('Data/Processed/contact_matrices/MIKE_contact_home_US_Census.csv',MIKE_HOME,delimiter=',')
jnp.savetxt('Data/Processed/contact_matrices/MIKE_contact_work_US_Census.csv',MIKE_WORK,delimiter=',')
jnp.savetxt('Data/Processed/contact_matrices/MIKE_contact_school_US_Census.csv',MIKE_SCHOOL,delimiter=',')
jnp.savetxt('Data/Processed/contact_matrices/MIKE_contact_others_US_Census.csv',MIKE_OTHERS,delimiter=',')
jnp.savetxt('Data/Processed/contact_matrices/MIKE_contact_all_US_Census.csv',MIKE_ALL,delimiter=',')