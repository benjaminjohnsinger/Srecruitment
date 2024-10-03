## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import itertools as it
import matplotlib.pyplot as plt

from SIRn_ODEs import single_pathogen_deltas as deltas


def lockdown_incidence_plot(ax,state0,params,OBS_AGE,period,points,T_LOCKDOWN,LOCKDOWN_DURATION,result=None,label='Observed cases',color='#648FFF',by_age=False,AGE_GROUP_NAMES=None):
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params[0]
    if result is None:
        result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=points,args=params)
    ## Calculate observations
    obs = np.zeros((len(result.t),NAG))
    pop_size = np.sum(result.y,axis=0)
    for i_t,t in enumerate(result.t):
        foi = BETA*np.dot(contact(t),np.sum((np.array([result.y[(3*j+2)*NAG:(3*j+3)*NAG,i_t] for j in range(N_S)])*I_REL),axis=0))/pop_size[i_t]
        for i in range(N_S):
            obs[i_t,:] += OBS_AGE*P_OBS[i]*S_REL[i]*foi*result.y[(3*i+1)*NAG:(3*i+2)*NAG,i_t]
    if by_age:
        cmap = plt.get_cmap('viridis')
        pop_size_by_age = np.array([np.sum(result.y[range(i_age,(3*N_S+1)*NAG,NAG),:],axis=0) for i_age in range(NAG)]).T
        for i_age in range(NAG):
            ax.plot(result.t,obs[:,i_age]/pop_size_by_age[:,i_age], label=AGE_GROUP_NAMES[i_age], color=cmap(i_age/(NAG-1)))
        mx = 1.1*np.max(np.max(obs/pop_size_by_age,axis=1)[np.argmax(result.t>T_LOCKDOWN-5*12):np.argmax(result.t>T_LOCKDOWN+LOCKDOWN_DURATION+5*12)])
    else:
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
    ax.set_title('Incidence of disease with 1-year lockdown')

def lockdown_susceptibility_plot(ax,state0,params,period,points,T_LOCKDOWN,result=None,label='Susceptible_population',color='#648FFF',relative=True,by_age=False,AGE_GROUP_NAMES=None,style='-'):
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params[0]
    if result is None:
        result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=points,args=params)
    ## Calculate susceptibility
    sus = np.zeros((len(result.t),NAG))
    for i_t,t in enumerate(result.t):
        for i in range(N_S):
            sus[i_t,:] += S_REL[i]*S_AGE*result.y[(3*i+1)*NAG:(3*i+2)*NAG,i_t]
    if by_age:
        cmap = plt.get_cmap('viridis')
        rel_sus = sus/np.sum(sus,axis=1)[:,np.newaxis]
        for i in range(NAG):
            ax.plot(result.t,rel_sus[:,i], label=AGE_GROUP_NAMES[i], color=cmap(i/(NAG-1)),linestyle=style)
    else:
        total_sus = np.sum(sus,axis=1)
        if relative:
            rel_sus = total_sus/total_sus[np.argmax(result.t>=T_LOCKDOWN)]
            ax.plot(result.t,rel_sus, label=label,color=color,linestyle=style)
        else:
            ax.plot(result.t,total_sus, label=label,color=color,linestyle=style)

def lockdown_susceptibility_format(ax,points,T_LOCKDOWN,LOCKDOWN_DURATION,ymin=0.875,ymax=1.1):
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(T_LOCKDOWN-5*12,T_LOCKDOWN+LOCKDOWN_DURATION+5*12)
    ax.set_ylabel('Relative susceptibility')
    ax.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],0,ymax,color='gray',alpha=0.2)
    ax.set_xticks(np.arange(T_LOCKDOWN-5*12,T_LOCKDOWN+LOCKDOWN_DURATION+5*12,12),[str(int(x)-5) for x in np.arange(0,11,1)])
    ax.set_xlabel('Time (years)')
    ax.set_title('Population susceptibility with 1-year lockdown')

def age_infect_plot(ax,state0,params,AGE_GROUP_NAMES,period,points,T_LOCKDOWN,LOCKDOWN_DURATION,result=None):
    cmap = plt.get_cmap('viridis')
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params[0]
    if result is None:
        result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=points,args=params)
    ## Calculate rate of infections created by each age group
    infs = np.zeros((len(result.t),NAG))
    for i_t,t in enumerate(result.t):
        infs[i_t,:] = np.sum(np.array([BETA*result.y[(3*i+2)*NAG:(3*i+3)*NAG,i_t]*I_REL[i]*np.dot(contact(t),np.sum([S_REL[j]*result.y[(3*j+1)*NAG:(3*j+2)*NAG,i_t] for j in range(N_S)],axis=0)) for i in range(N_S)]),axis=0)
    rel_infs = infs/np.sum(infs,axis=1)[:,np.newaxis]
    for i in range(NAG):
        ax.plot(result.t,rel_infs[:,i], label=AGE_GROUP_NAMES[i], color=cmap(i/(NAG-1)),zorder=1)
    ax.set_xlim(T_LOCKDOWN-5*12,T_LOCKDOWN+LOCKDOWN_DURATION+5*12)
    ax.set_title('Infections caused by each age group')
    ax.set_ylabel('Infection rate')
    ax.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],0,1,color='gray',zorder=0,alpha=0.2)
    ax.set_xticks(np.arange(T_LOCKDOWN-5*12,T_LOCKDOWN+LOCKDOWN_DURATION+5*12,12),[str(int(x)-5) for x in np.arange(0,11,1)])
    ax.set_xlabel('Time (years)')
    ax.set_ylim(0,1)
    # ax.legend()

def age_infect_grid_plot(axes,state0,params,AGE_GROUP_NAMES,period,points,T_LOCKDOWN,LOCKDOWN_DURATION,N=25,N_span=25):
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params[0]

    # Grid plots of infections caused by children, under-fives
    child_inf = np.zeros((N,N))
    under_five_inf = np.zeros((N,N))

    for beta_n in range(N):
        for wane_n in range(N):
            # Scale parameters for exploration
            sim_beta = BETA*(1+(1/N_span)*(beta_n-N/2))
            sim_rec = REC*(1+(1/N_span)*(beta_n-N/2))
            sim_wane_up = WANE_UP*(1+(1/N_span)*(wane_n-N/2))
            sim_wane_same = WANE_SAME*(1+(1/N_span)*(wane_n-N/2))
            # Run simulation
            args = (NAG, N_S, AGING_RATE, BIRTH_RATE, sim_wane_up, sim_wane_same, sim_rec, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, sim_beta, contact)
            result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=np.linspace(0,period,points),args=(args,))
            # Calculate rate of infections created by each age group
            infs = np.zeros((len(result.t),NAG))
            for i_t,t in enumerate(result.t):
                infs[i_t,:] = np.sum(np.array([sim_beta*result.y[(3*i+2)*NAG:(3*i+3)*NAG,i_t]*I_REL[i]*np.dot(contact(t),np.sum([S_REL[j]*result.y[(3*j+1)*NAG:(3*j+2)*NAG,i_t] for j in range(N_S)],axis=0)) for i in range(N_S)]),axis=0)
            # nan if simulation crashes
            if len(result.t) != points:
                child_inf[beta_n,wane_n] = np.nan
                under_five_inf[beta_n,wane_n] = np.nan
                continue
            else:
                # infections caused by children
                child_inf[beta_n,wane_n] = (np.sum(infs[:,0:4],axis=1)/np.sum(infs,axis=1))[np.argmax(result.t>=T_LOCKDOWN)]
                # infections caused by under-fives
                under_five_inf[beta_n,wane_n] = (np.sum(infs[:,0:3],axis=1)/np.sum(infs,axis=1))[np.argmax(result.t>=T_LOCKDOWN)]
    
    tick_space = N//4 + 1

    axes[0].imshow(child_inf)
    axes[0].set_ylabel('Growth rate')
    axes[0].set_xlabel('Wane rate')
    axes[0].set_title('Infections caused by\nchildren')
    axes[0].set_xticks(range(0,N,tick_space),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,tick_space)])
    axes[0].set_yticks(range(0,N,tick_space),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,tick_space)])
    cbar = plt.colorbar(axes[0].imshow(child_inf), ax=axes[0])

    axes[1].imshow(under_five_inf)
    axes[1].set_xlabel('Wane rate')
    axes[1].set_title('Infections caused by\nunder-fives')
    axes[1].set_xticks(range(0,N,tick_space),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,tick_space)])
    axes[1].set_yticks(range(0,N,tick_space),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,tick_space)])
    cbar = plt.colorbar(axes[1].imshow(under_five_inf), ax=axes[1])
    cbar.set_label('Fraction of infections')


def grid_plot(axes,state0,params,OBS_AGE,period,T_LOCKDOWN,LOCKDOWN_DURATION,N=25,N_span=25):
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params[0]

    points = np.concatenate((np.zeros(1),np.arange(T_LOCKDOWN-5*12,T_LOCKDOWN+LOCKDOWN_DURATION+5*12),np.ones(1)*period))

    # Grid plots of peak incidence before and after lockdown
    pre_peaks = np.zeros((N,N))
    post_peaks = np.zeros((N,N))
    pre_space = np.zeros((N,N))
    post_space = np.zeros((N,N))
    pre_susc = np.zeros((N,N))
    post_susc = np.zeros((N,N))

    for beta_n in range(N):
        for wane_n in range(N):
            # Scale parameters for exploration
            sim_beta = BETA*(1+(1/N_span)*(beta_n-N/2))
            sim_rec = REC*(1+(1/N_span)*(beta_n-N/2))
            sim_wane_up = WANE_UP*(1+(1/N_span)*(wane_n-N/2))
            sim_wane_same = WANE_SAME*(1+(1/N_span)*(wane_n-N/2))
            # Run simulation
            args = (NAG, N_S, AGING_RATE, BIRTH_RATE, sim_wane_up, sim_wane_same, sim_rec, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, sim_beta, contact)
            result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=points,args=(args,))
            # Calculate observations and susceptibility
            obs = np.zeros((len(result.t),NAG))
            sus = np.zeros((len(result.t),NAG))
            pop_size = np.sum(result.y,axis=0)
            for i_t,t in enumerate(result.t):
                foi = sim_beta*np.dot(contact(t),np.sum((np.array([result.y[(3*j+2)*NAG:(3*j+3)*NAG,i_t] for j in range(N_S)])*I_REL),axis=0))\
                    /pop_size[i_t]
                for i in range(N_S):
                    obs[i_t,:] += OBS_AGE*P_OBS[i]*S_REL[i]*foi*result.y[(3*i+1)*NAG:(3*i+2)*NAG,i_t]
                    sus[i_t,:] += S_REL[i]*S_AGE*result.y[(3*i+1)*NAG:(3*i+2)*NAG,i_t]
            # nan if simulation crashes
            if len(result.t) != len(points):
                pre_peaks[beta_n,wane_n] = np.nan
                post_peaks[beta_n,wane_n] = np.nan
                pre_space[beta_n,wane_n] = np.nan
                post_space[beta_n,wane_n] = np.nan
                pre_susc[beta_n,wane_n] = np.nan
                post_susc[beta_n,wane_n] = np.nan
                continue
            else:
                # pre-lockdown peak incidence
                pre_obs = np.sum(obs[(result.t>25*12) & (result.t<T_LOCKDOWN)],axis=1)
                pre_peak = np.max(pre_obs/pop_size[(result.t>25*12) & (result.t<T_LOCKDOWN)])
                pre_peaks[beta_n,wane_n] = pre_peak
                # post-lockdown peak incidence
                post_peaks[beta_n,wane_n] = np.max(np.sum(obs[result.t>=T_LOCKDOWN],axis=1)/pop_size[result.t>=T_LOCKDOWN])
                # distinguish between annual, biannual, etc outbreaks pre-lockdown
                corr = np.correlate(pre_obs, pre_obs, mode='same')
                acorr = corr[len(pre_obs)//2 + 1:] / (pre_obs.var() * np.arange(len(pre_obs)-1, len(pre_obs)//2, -1))
                lag = np.abs(acorr).argmax() + 1
                pre_space[beta_n,wane_n] = lag
                # find time to first post-lockdown rebound, first time to half the pre-lockdown peak incidence
                post_peak_arg = np.argmax(np.sum(obs[result.t>=T_LOCKDOWN],axis=1)/pop_size[result.t>=T_LOCKDOWN] > pre_peak/2)
                post_space[beta_n,wane_n] = result.t[post_peak_arg]
                # absolut susceptibility when lockdown starts
                total_sus = np.sum(sus,axis=1)
                pre_susc[beta_n,wane_n] = total_sus[np.argmax(result.t>=T_LOCKDOWN)]
                # susceptibility when lockdown ends
                # rel_sus = total_sus/total_sus[np.argmax(result.t>=T_LOCKDOWN)]
                post_susc[beta_n,wane_n] = total_sus[np.argmax(result.t>=T_LOCKDOWN+LOCKDOWN_DURATION)]
    print(post_space)
    tick_space = N//4 + 1

    axes[0,0].imshow(pre_peaks)
    axes[0,0].set_ylabel('Growth rate')
    axes[0,0].set_title('Pre-lockdown\npeak incidence')
    axes[0,0].set_xticks(range(0,N,tick_space),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,tick_space)])
    axes[0,0].set_yticks(range(0,N,tick_space),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,tick_space)])
    cbar = plt.colorbar(axes[0,0].imshow(pre_peaks), ax=axes[0,0])

    axes[0,1].imshow(post_peaks)
    axes[0,1].set_title('Post-lockdown\npeak incidence')
    axes[0,1].set_xticks(range(0,N,tick_space),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,tick_space)])
    axes[0,1].set_yticks(range(0,N,tick_space),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,tick_space)])
    cbar = plt.colorbar(axes[0,1].imshow(post_peaks), ax=axes[0,1])
    cbar.set_label('Observed incidence')

    axes[1,0].imshow(pre_space)
    axes[1,0].set_ylabel('Growth rate')
    axes[1,0].set_title('Pre-lockdown\nperiodicity')
    axes[1,0].set_xticks(range(0,N,tick_space),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,tick_space)])
    axes[1,0].set_yticks(range(0,N,tick_space),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,tick_space)])
    cbar = plt.colorbar(axes[1,0].imshow(pre_space), ax=axes[1,0])

    axes[1,1].imshow(post_space)
    axes[1,1].set_title('Time to post-\nlockdown rebound')
    axes[1,1].set_xticks(range(0,N,tick_space),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,tick_space)])
    axes[1,1].set_yticks(range(0,N,tick_space),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,tick_space)])
    cbar = plt.colorbar(axes[1,1].imshow(post_space), ax=axes[1,1])
    cbar.set_label('Time (months)')

    axes[2,0].imshow(pre_susc)
    axes[2,0].set_xlabel('Wane rate')
    axes[2,0].set_ylabel('Growth rate')
    axes[2,0].set_title('Susceptibility\nat lockdown start')
    axes[2,0].set_xticks(range(0,N,tick_space),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,tick_space)])
    axes[2,0].set_yticks(range(0,N,tick_space),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,tick_space)])
    cbar = plt.colorbar(axes[2,0].imshow(pre_susc), ax=axes[2,0])

    axes[2,1].imshow(post_susc)
    axes[2,1].set_xlabel('Wane rate')
    axes[2,1].set_title('Susceptibility\nat lockdown end')
    axes[2,1].set_xticks(range(0,N,tick_space),[f'{1.5*np.mean(WANE_UP)*(1+(1/N_span)*(wane_n-N/2)):.2f}' for wane_n in range(0,N,tick_space)])
    axes[2,1].set_yticks(range(0,N,tick_space),[f'{(BETA-np.mean(REC))*(1+(1/N_span)*(beta_n-N/2)):.0f}' for beta_n in range(0,N,tick_space)])
    cbar = plt.colorbar(axes[2,1].imshow(post_susc), ax=axes[2,1])
    cbar.set_label('Effective susceptible population')
