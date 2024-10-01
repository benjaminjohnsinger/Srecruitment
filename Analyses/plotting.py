## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import itertools as it
import matplotlib.pyplot as plt

from SIRn_ODEs import single_pathogen_deltas as deltas


def lockdown_incidence_plot(ax,state0,params,OBS_AGE,period,points,T_LOCKDOWN,LOCKDOWN_DURATION,result=None,label='Observed cases',color='#648FFF'):
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params[0]
    if result is None:
        result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=np.linspace(0,period,points),args=params)
    ## Calculate observations
    obs = np.zeros((len(result.t),NAG))
    pop_size = np.sum(result.y,axis=0)
    for i_t,t in enumerate(result.t):
        foi = BETA*np.dot(contact(t),np.sum((np.array([result.y[(3*j+2)*NAG:(3*j+3)*NAG,i_t] for j in range(N_S)])*I_REL),axis=0))/pop_size[i_t]
        for i in range(N_S):
            obs[i_t,:] += OBS_AGE*P_OBS[i]*S_REL[i]*foi*result.y[(3*i+1)*NAG:(3*i+2)*NAG,i_t]
    ax.plot(result.t,np.sum(obs,axis=1)/pop_size, label=label,color=color)
    mx = 1.1*np.max((np.sum(obs,axis=1)/pop_size)[np.argmax(result.t>T_LOCKDOWN-5*12):np.argmax(result.t>T_LOCKDOWN+LOCKDOWN_DURATION+5*12)])
    return(mx)

def lockdown_incidence_format(ax,points,T_LOCKDOWN,LOCKDOWN_DURATION,mx):
    ax.set_xlim(T_LOCKDOWN-5*12,T_LOCKDOWN+LOCKDOWN_DURATION+5*12)
    ax.set_ylim(0,mx)
    ax.set_ylabel('Observed incidence')
    ax.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],0,mx,color='gray',alpha=0.2)
    ax.set_xticks(np.arange(T_LOCKDOWN-5*12,T_LOCKDOWN+LOCKDOWN_DURATION+5*12,12),[str(int(x)-5) for x in np.arange(0,11,1)])
    ax.set_xlabel('Time (years)')
    ax.set_title('Incidence of non-pediatric disease with 1-year lockdown')


def lockdown_susceptibility_plot(ax,state0,params,period,points,T_LOCKDOWN,LOCKDOWN_DURATION,result=None,label='Susceptible_population',color='#648FFF'):
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params[0]
    if result is None:
        result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=np.linspace(0,period,points),args=params)
    ## Calculate susceptibility
    sus = np.zeros((len(result.t),NAG))
    for i_t,t in enumerate(result.t):
        for i in range(N_S):
            sus[i_t,:] += S_REL[i]*result.y[(3*i+1)*NAG:(3*i+2)*NAG,i_t]
    total_sus = np.sum(sus,axis=1)
    rel_sus = total_sus/total_sus[np.argmax(result.t>=T_LOCKDOWN)]
    ax.plot(result.t,rel_sus, label=label,color=color)

def lockdown_susceptibility_format(ax,points,T_LOCKDOWN,LOCKDOWN_DURATION):
    ax.set_ylim(0.875,1.1)
    ax.set_xlim(T_LOCKDOWN-5*12,T_LOCKDOWN+LOCKDOWN_DURATION+5*12)
    ax.set_ylabel('Relative susceptibility')
    ax.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],0,1.1,color='gray',alpha=0.2)
    ax.set_xticks(np.arange(T_LOCKDOWN-5*12,T_LOCKDOWN+LOCKDOWN_DURATION+5*12,12),[str(int(x)-5) for x in np.arange(0,11,1)])
    ax.set_xlabel('Time (years)')
    ax.set_title('Population susceptibility with 1-year lockdown')


def grid_plot(fig,state0,params,period,points,T_LOCKDOWN,N=25,N_span=10):
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params[0]

    # Grid plots of peak incidence before and after lockdown
    pre_peaks = np.zeros((N,N))
    post_peaks = np.zeros((N,N))
    pre_space = np.zeros((N,N))
    post_space = np.zeros((N,N))

    for beta_n in range(N):
        for wane_n in range(N):
            sim_beta = BETA*(1+(1/N_span)*(beta_n-N/2))
            sim_rec = REC*(1+(1/N_span)*(beta_n-N/2))
            sim_wane_up = WANE_UP*(1+(1/N_span)*(wane_n-N/2))
            sim_wane_same = WANE_SAME*(1+(1/N_span)*(wane_n-N/2))
            args = (NAG, N_S, AGING_RATE, BIRTH_RATE, sim_wane_up, sim_wane_same, sim_rec, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, sim_beta, contact)
            result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=np.linspace(0,period,points),args=(args,))
            ## Calculate observations
            obs = np.zeros((len(result.t),NAG))
            pop_size = np.sum(result.y,axis=0)
            for i_t,t in enumerate(result.t):
                foi = sim_beta*np.dot(contact(t),np.sum((np.array([result.y[(3*j+2)*NAG:(3*j+3)*NAG,i_t] for j in range(N_S)])*I_REL),axis=0))\
                    /pop_size[i_t]
                for i in range(N_S):
                    obs[i_t,:] += OBS_AGE*P_OBS[i]*S_REL[i]*foi*result.y[(3*i+1)*NAG:(3*i+2)*NAG,i_t]
            if len(result.t) != points:
                pre_peaks[beta_n,wane_n] = np.nan
                post_peaks[beta_n,wane_n] = np.nan
                pre_space[beta_n,wane_n] = np.nan
                post_space[beta_n,wane_n] = np.nan
                continue
            else:
                pre_obs = np.sum(obs[(result.t>25*12) & (result.t<T_LOCKDOWN)],axis=1)
                pre_peak = np.max(pre_obs/pop_size[(result.t>25*12) & (result.t<T_LOCKDOWN)])
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


    axes = fig.subplots(2,2,sharey=True,sharex=True)

    axes[0,0].imshow(pre_peaks)
    axes[0,0].set_ylabel('Growth rate')
    axes[0,0].set_title('Pre-lockdown\npeak incidence')
    axes[0,0].set_xticks(range(0,N,5),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,5)])
    axes[0,0].set_yticks(range(0,N,5),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,5)])
    for tick in axes[0,0].get_xticklabels():
        tick.set_rotation(45)
    # add colorbar
    cbar = plt.colorbar(axes[0,0].imshow(pre_peaks), ax=axes[0,0])

    axes[0,1].imshow(post_peaks)
    axes[0,1].set_title('Post-lockdown\npeak incidence')
    axes[0,1].set_xticks(range(0,N,5),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,5)])
    axes[0,1].set_yticks(range(0,N,5),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,5)])
    for tick in axes[0,1].get_xticklabels():
        tick.set_rotation(45)
    # add colorbar
    cbar = plt.colorbar(axes[0,1].imshow(post_peaks), ax=axes[0,1])
    cbar.set_label('Observed incidence')

    axes[1,0].imshow(pre_space)
    axes[1,0].set_xlabel('Wane rate')
    axes[1,0].set_ylabel('Growth rate')
    axes[1,0].set_title('Pre-lockdown\nperiodicity')
    axes[1,0].set_xticks(range(0,N,5),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,5)])
    axes[1,0].set_yticks(range(0,N,5),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,5)])
    for tick in axes[1,0].get_xticklabels():
        tick.set_rotation(45)
    # add colorbar
    cbar = plt.colorbar(axes[1,0].imshow(pre_space), ax=axes[1,0])

    axes[1,1].imshow(post_space)
    axes[1,1].set_xlabel('Wane rate')
    axes[1,1].set_title('Time to post-\nlockdown rebound')
    axes[1,1].set_xticks(range(0,N,5),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,5)])
    axes[1,1].set_yticks(range(0,N,5),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,5)])
    for tick in axes[1,1].get_xticklabels():
        tick.set_rotation(45)
    # add colorbar
    cbar = plt.colorbar(axes[1,1].imshow(post_space), ax=axes[1,1])
    cbar.set_label('Time (months)')

# plt.tight_layout()
# plt.savefig('Figures/test.png')
