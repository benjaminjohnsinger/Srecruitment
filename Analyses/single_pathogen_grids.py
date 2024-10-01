## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import itertools as it
import matplotlib.pyplot as plt

from vaccination import birth_vax, all_vax
import contact_model as cm
from SIRn_ODEs import single_pathogen_deltas as deltas
from Parameters.test_population import *
from Parameters.adult_disease import *

## Period of simulation in months
T_FACTOR = 1
PERIOD = int(T_FACTOR*12*50)

# Correct parameters with T_FACTOR
WANE_UP /= T_FACTOR
WANE_SAME /= T_FACTOR
REC /= T_FACTOR
AGING_RATE /= T_FACTOR
BIRTH_RATE /= T_FACTOR
ACOV_SCALED = lambda t,T_VAX : ACOV(t,T_VAX)/T_FACTOR

## Contacts and force of infection
IMPORT = 0.01
# Contact matrix for all contact types
CONTACT = np.genfromtxt('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',')
CONTACT /= 12.99 # transform to contact proportions
# Lockdown and other mobility changes
T_LOCKDOWN = 37*12*T_FACTOR
LOCKDOWN_DURATION = 12*T_FACTOR
LOCKDOWN_REDUCTION = 0.4
shape_step = lambda t : cm.STEP(t,T_LOCKDOWN,LOCKDOWN_DURATION,LOCKDOWN_REDUCTION)
shape_static = lambda t : 1
def contact(t,shape,seasonality=SEASONALITY,offset=OFFSET,t_factor=T_FACTOR,contact=CONTACT):
    return shape(t)*(1+seasonality*np.cos(2*np.pi*(t/(12*t_factor)-offset)))*contact

## Initial conditions
STATE0 = np.zeros((1+N_S*3)*(NAG))
STATE0[NAG:2*NAG] = KP_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

## Integrate the system
POINTS = PERIOD
T_VAX = PERIOD

# Grid plots of peak incidence before and after lockdown
N=12
N_span = 10
pre_peaks = np.zeros((N,N))
post_peaks = np.zeros((N,N))
pre_space = np.zeros((N,N))
post_space = np.zeros((N,N))
# I_REL = I_REL**(1/2)
S_REL = S_REL**(2)

for beta_n in range(N):
    for wane_n in range(N):
        sim_beta = BETA*(1+(1/N_span)*(beta_n-N/2))
        sim_rec = REC*(1+(1/N_span)*(beta_n-N/2))
        sim_wane_up = WANE_UP*(1+(1/N_span)*(wane_n-N/2))
        sim_wane_same = WANE_SAME*(1+(1/N_span)*(wane_n-N/2))
        print(sim_beta,sim_rec)
        sim_rec[0] = sim_rec[0]/2
        result = sp.integrate.solve_ivp(deltas, [0,PERIOD], STATE0, method='RK45', t_eval=np.linspace(0,PERIOD,POINTS),
            args=((NAG, N_S, AGING_RATE, BIRTH_RATE, sim_wane_up, sim_wane_same, sim_rec, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV_SCALED, BCOV, T_VAX, IMPORT, sim_beta,
            lambda t : contact(t, shape_step,SEASONALITY,OFFSET,T_FACTOR,CONTACT)),))
        ## Calculate observations
        obs = np.zeros((len(result.t),NAG))
        pop_size = np.sum(result.y,axis=0)
        for i_t,t in enumerate(result.t):
            foi = sim_beta*np.dot(contact(t,shape_step),np.sum((np.array([result.y[(3*j+2)*NAG:(3*j+3)*NAG,i_t] for j in range(N_S)])*I_REL),axis=0))/pop_size[i_t]
            for i in range(N_S):
                obs[i_t,:] += OBS_AGE*P_OBS[i]*S_REL[i]*foi*result.y[(3*i+1)*NAG:(3*i+2)*NAG,i_t]*T_FACTOR
        if len(result.t) != POINTS:
            pre_peaks[beta_n,wane_n] = np.nan
            post_peaks[beta_n,wane_n] = np.nan
            pre_space[beta_n,wane_n] = np.nan
            post_space[beta_n,wane_n] = np.nan
            continue
        else:
            pre_obs = np.sum(obs[(result.t>25*12*T_FACTOR) & (result.t<T_LOCKDOWN)],axis=1)
            pre_peak = np.max(pre_obs/pop_size[(result.t>25*12*T_FACTOR) & (result.t<T_LOCKDOWN)])
            pre_peaks[beta_n,wane_n] = pre_peak
            post_peaks[beta_n,wane_n] = np.max(np.sum(obs[result.t>=T_LOCKDOWN],axis=1)/pop_size[result.t>=T_LOCKDOWN])
            # distinguish between annual, biannual, etc outbreaks pre-lockdown
            corr = np.correlate(pre_obs, pre_obs, mode='same')
            acorr = corr[len(pre_obs)//2 + 1:] / (pre_obs.var() * np.arange(len(pre_obs)-1, len(pre_obs)//2, -1))
            lag = np.abs(acorr).argmax() + 1
            pre_space[beta_n,wane_n] = lag
            # find time to first post-lockdown rebound, at least half the pre-lockdown peak
            post_peak_arg = np.argmax(np.sum(obs[result.t>=T_LOCKDOWN],axis=1)/pop_size[result.t>=T_LOCKDOWN] > pre_peak/2)
            post_space[beta_n,wane_n] = result.t[post_peak_arg]
            # post_peak_arg = np.argmin(np.abs(np.sum(obs[result.t>=T_LOCKDOWN],axis=1)/pop_size[result.t>=T_LOCKDOWN] - pre_peak))
            # post_space[beta_n,wane_n] = result.t[post_peak_arg]

## Plot the results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2,2,figsize=(6.5,6.5),sharey=True,sharex=True)

axes[0,0].imshow(pre_peaks)
# axes[0,0].set_xlabel('Waning rate')
axes[0,0].set_ylabel('Growth Rate')
axes[0,0].set_title('Pre-lockdown\npeak incidence')
axes[0,0].set_xticks(range(0,N,5),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,5)])
axes[0,0].set_yticks(range(0,N,5),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,5)])
for tick in axes[0,0].get_xticklabels():
    tick.set_rotation(45)
# add colorbar
cbar = plt.colorbar(axes[0,0].imshow(pre_peaks), ax=axes[0,0])
# cbar.set_label('Observed incidence')

axes[0,1].imshow(post_peaks)
# axes[0,1].set_xlabel('Waning rate')
# axes[0,1].set_ylabel('Growth Rate')
axes[0,1].set_title('Post-lockdown\npeak incidence')
axes[0,1].set_xticks(range(0,N,5),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,5)])
axes[0,1].set_yticks(range(0,N,5),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,5)])
for tick in axes[0,1].get_xticklabels():
    tick.set_rotation(45)
# add colorbar
cbar = plt.colorbar(axes[0,1].imshow(post_peaks), ax=axes[0,1])
cbar.set_label('Observed incidence')

axes[1,0].imshow(pre_space)
axes[1,0].set_xlabel('Waning rate')
axes[1,0].set_ylabel('Growth Rate')
axes[1,0].set_title('Pre-lockdown\nperiodicity')
axes[1,0].set_xticks(range(0,N,5),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,5)])
axes[1,0].set_yticks(range(0,N,5),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,5)])
for tick in axes[1,0].get_xticklabels():
    tick.set_rotation(45)
# add colorbar
cbar = plt.colorbar(axes[1,0].imshow(pre_space), ax=axes[1,0])
# cbar.set_label('Lag (months)')

axes[1,1].imshow(post_space)
axes[1,1].set_xlabel('Waning rate')
# axes[1,1].set_ylabel('Growth Rate')
axes[1,1].set_title('Time to post-\nlockdown rebound')
axes[1,1].set_xticks(range(0,N,5),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,5)])
axes[1,1].set_yticks(range(0,N,5),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,5)])
for tick in axes[1,1].get_xticklabels():
    tick.set_rotation(45)
# add colorbar
cbar = plt.colorbar(axes[1,1].imshow(post_space), ax=axes[1,1])
cbar.set_label('Time (months)')

plt.tight_layout()
plt.savefig('Figures/adult_beta_wane_grids_half_first_rec.png')
