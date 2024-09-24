import numpy as np

# Contact matrix for all contact types
CONTACT = np.genfromtxt('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',')

def STATIC(t):
    return 1

def STEP(t,t_lockdown,duration,reduction):
    return 1-reduction if t < t_lockdown + duration and t > t_lockdown else 1