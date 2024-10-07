## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import itertools as it
import matplotlib.pyplot as plt

from SIRn_ODEs import single_pathogen_deltas as deltas

N_C = 3

def lockdown_incidence_plot(ax,state0,params,OBS_AGE,period,points,T_LOCKDOWN,LOCKDOWN_DURATION,result=None,label='Observed cases',color='#648FFF',by_age=False,AGE_GROUP_NAMES=None):
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params.values()
    if result is None:
        result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=points,args=(params,))
    ## Calculate observations
    obs = np.zeros((len(result.t),NAG))
    pop_size = np.sum(result.y,axis=0)
    for i_t,t in enumerate(result.t):
        foi = BETA*np.dot(contact(t),np.sum((np.array([result.y[(N_C*j+2)*NAG:(N_C*j+3)*NAG,i_t] for j in range(N_S)])*I_REL),axis=0))/pop_size[i_t]
        for i in range(N_S):
            obs[i_t,:] += OBS_AGE*P_OBS[i]*S_REL[i]*foi*result.y[(N_C*i+1)*NAG:(N_C*i+2)*NAG,i_t]
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
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params.values()
    if result is None:
        result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=points,args=(params,))
    ## Calculate susceptibility
    sus = np.zeros((len(result.t),NAG))
    for i_t,t in enumerate(result.t):
        for i in range(N_S):
            sus[i_t,:] += S_REL[i]*S_AGE*result.y[(N_C*i+1)*NAG:(N_C*i+2)*NAG,i_t]
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
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params.values()
    if result is None:
        result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=points,args=(params,))
    ## Calculate rate of infections created by each age group
    infs = np.zeros((len(result.t),NAG))
    for i_t,t in enumerate(result.t):
        infs[i_t,:] = np.sum(np.array([BETA*result.y[(N_C*i+2)*NAG:(N_C*i+3)*NAG,i_t]*I_REL[i]*np.dot(contact(t),np.sum([S_REL[j]*result.y[(N_C*j+1)*NAG:(N_C*j+2)*NAG,i_t] for j in range(N_S)],axis=0)) for i in range(N_S)]),axis=0)
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

def sim_grid(state0,params,period,points,T_LOCKDOWN,LOCKDOWN_DURATION,
            grid_params=(("BETA","REC"),("WANE_UP","WANE_SAME")),grid_mode=("scale","scale"),N=10,factors=(1,1)):
    N_params = len(grid_params)
    results = {}

    for p_n in it.product(range(N),repeat=N_params):
        # Scale parameters for exploration
        params_n = params.copy()
        for i,p in enumerate(p_n):
            for pname in grid_params[i]:
                if grid_mode[i] == "scale":
                    params_n[pname] = params[pname]*(1+factors[i]*(p/N-1/2))
                elif grid_mode[i] == "fade_vec":
                    N_S = params["N_S"]
                    params_n[pname] = np.array([1-i*factors[i]*(p/(N_span*N_S)) for i in range(N_S)])
        # Run simulation
        result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=points, args=(params_n,))
        results[p_n] = result
    return(results)

def age_infect_grid_plot(axes,results,params,T_LOCKDOWN,LOCKDOWN_DURATION,
grid_params=(("BETA","REC"),("WANE_UP","WANE_SAME")),grid_mode=("scale","scale"),label_mode=("diff_mean","nz_mean"),factors=(1,1)):
    N_params = len(grid_params)
    N = max([max(p) for p in results.keys()])+1
    # Grid plots of infections caused by children, under-fives
    child_inf = np.zeros((N,N))
    under_five_inf = np.zeros((N,N))
    parameter_values = np.zeros((N,N_params))

    for p_n,result in results.items():
        N_S = params["N_S"]
        params_n = params.copy()
        for i,p in enumerate(p_n):
            for pname in grid_params[i]:
                if grid_mode[i] == "scale":
                    params_n[pname] = params[pname]*(1+factors[i]*(p/N-1/2))
                elif grid_mode[i] == "fade_vec":
                    params_n[pname] = np.array([1-i*factors[i]*(p/(N_span*N_S)) for i in range(N_S)])
            if label_mode[i]=="diff_mean":
                parameter_values[p_n[i],i] = np.mean(params_n[grid_params[i][0]]) - np.mean(params_n[grid_params[i][1]])
            elif label_mode[i]=="mean":
                parameter_values[p_n[i],i] = np.mean([np.mean(params_n[pname]) for pname in grid_params[i]])
            elif label_mode[i]=="nz_mean":
                parameter_values[p_n[i],i] = np.mean([np.mean(params_n[pname])*(len(params_n[pname])/np.sum(params_n[pname]==0)) for pname in grid_params[i]])
        # Calculate infections caused by each age group
        NAG, BETA, REC, S_REL, I_REL, contact = params_n["NAG"], params_n["BETA"], params_n["REC"], params_n["S_REL"], params_n["I_REL"], params_n["contact"]
        infs = np.zeros((len(result.t),NAG))
        for i_t,t in enumerate(result.t):
            infs[i_t,:] = np.sum(np.array([BETA*result.y[(N_C*i+2)*NAG:(N_C*i+3)*NAG,i_t]*I_REL[i]*np.dot(contact(t),np.sum([S_REL[j]*result.y[(N_C*j+1)*NAG:(N_C*j+2)*NAG,i_t] for j in range(N_S)],axis=0)) for i in range(N_S)]),axis=0)
        # infections caused by children
        child_inf[p_n] = (np.sum(infs[:,0:4],axis=1)/np.sum(infs,axis=1))[np.argmax(result.t>=T_LOCKDOWN)]
        # infections caused by under-fives
        under_five_inf[p_n] = (np.sum(infs[:,0:3],axis=1)/np.sum(infs,axis=1))[np.argmax(result.t>=T_LOCKDOWN)]

    tick_space = N//4 + 1

    im0 = axes[0].imshow(child_inf)
    axes[0].set_title('Infections caused by\nchildren')
    axes[0].set_yticks(range(0,N,tick_space),[f'{parameter_values[y_n,0]:.0f}' for y_n in range(0,N,tick_space)])
    axes[0].set_xticks(range(0,N,tick_space),[f'{parameter_values[x_n,1]:.2f}' for x_n in range(0,N,tick_space)])
    cbar = plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(under_five_inf)
    axes[1].set_title('Infections caused by\nunder-fives')
    axes[1].set_yticks(range(0,N,tick_space),[f'{parameter_values[y_n,0]:.0f}' for y_n in range(0,N,tick_space)])
    axes[1].set_xticks(range(0,N,tick_space),[f'{parameter_values[x_n,1]:.2f}' for x_n in range(0,N,tick_space)])
    cbar = plt.colorbar(im1, ax=axes[1])
    cbar.set_label('Fraction of infections')


def obs_grid_plot(axes,results,params,OBS_AGE,T_LOCKDOWN,LOCKDOWN_DURATION,
grid_params=(("BETA","REC"),("WANE_UP","WANE_SAME")),grid_mode=("scale","scale"),factors=(1,1),
label_mode=("diff_mean","nz_mean")):
    N_params = len(grid_params)
    N = max([max(p) for p in results.keys()])+1
    # Grid plots of peak incidence before and after lockdown
    pre_peaks = np.zeros((N,N))
    post_peaks = np.zeros((N,N))
    pre_space = np.zeros((N,N))
    post_space = np.zeros((N,N))
    parameter_values = np.zeros((N,N_params))

    for p_n,result in results.items():
        N_S = params["N_S"]
        params_n = params.copy()
        for i,p in enumerate(p_n):
            for pname in grid_params[i]:
                if grid_mode[i] == "scale":
                    params_n[pname] = params[pname]*(1+factors[i]*(p/N-1/2))
                elif grid_mode[i] == "fade_vec":
                    params_n[pname] = np.array([1-i*factors[i]*(p/(N_span*N_S)) for i in range(N_S)])
            if label_mode[i]=="diff_mean":
                parameter_values[p_n[i],i] = np.mean(params_n[grid_params[i][0]]) - np.mean(params_n[grid_params[i][1]])
            elif label_mode[i]=="mean":
                parameter_values[p_n[i],i] = np.mean([np.mean(params_n[pname]) for pname in grid_params[i]])
            elif label_mode[i]=="nz_mean":
                parameter_values[p_n[i],i] = np.mean([np.mean(params_n[pname])*(len(params_n[pname])/np.sum(params_n[pname]==0)) for pname in grid_params[i]])
        # Calculate observations
        NAG, BETA, I_REL, P_OBS, S_REL, contact = params_n["NAG"], params_n["BETA"], params_n["I_REL"], params_n["P_OBS"], params_n["S_REL"], params_n["contact"]
        obs = np.zeros((len(result.t),NAG))
        pop_size = np.sum(result.y,axis=0)
        for i_t,t in enumerate(result.t):
            foi = BETA*np.dot(contact(t),np.sum((np.array([result.y[(N_C*j+2)*NAG:(N_C*j+3)*NAG,i_t] for j in range(N_S)])*I_REL),axis=0))/pop_size[i_t]
            for i in range(N_S):
                obs[i_t,:] += OBS_AGE*P_OBS[i]*S_REL[i]*foi*result.y[(N_C*i+1)*NAG:(N_C*i+2)*NAG,i_t]
        # pre-lockdown peak incidence
        pre_obs = np.sum(obs[(result.t>25*12) & (result.t<T_LOCKDOWN)],axis=1)
        pre_peak = np.max(pre_obs/pop_size[(result.t>25*12) & (result.t<T_LOCKDOWN)])
        pre_peaks[p_n] = pre_peak
        # post-lockdown peak incidence
        post_peaks[p_n] = np.max(np.sum(obs[result.t>=T_LOCKDOWN],axis=1)/pop_size[result.t>=T_LOCKDOWN])
        # distinguish between annual, biannual, etc outbreaks pre-lockdown
        corr = np.correlate(pre_obs, pre_obs, mode='same')
        acorr = corr[len(pre_obs)//2 + 1:] / (pre_obs.var() * np.arange(len(pre_obs)-1, len(pre_obs)//2, -1))
        lag = np.abs(acorr).argmax() + 1
        pre_space[p_n] = lag
        # find time to first post-lockdown rebound, first time to half the pre-lockdown peak incidence
        post_peak_arg = np.argmax(np.sum(obs[result.t>=T_LOCKDOWN],axis=1)/pop_size[result.t>=T_LOCKDOWN] > pre_peak/2)
        post_space[p_n] = result.t[np.argmax(result.t>=T_LOCKDOWN)+post_peak_arg]

    tick_space = N//4 + 1

    im00 = axes[0,0].imshow(pre_peaks)
    axes[0,0].set_title('Pre-lockdown\npeak incidence')
    axes[0,0].set_yticks(range(0,N,tick_space),[f'{parameter_values[y_n,0]:.0f}' for y_n in range(0,N,tick_space)])
    axes[0,0].set_xticks(range(0,N,tick_space),[f'{parameter_values[x_n,1]:.2f}' for x_n in range(0,N,tick_space)])
    cbar = plt.colorbar(im00, ax=axes[0,0])

    im01 = axes[0,1].imshow(post_peaks)
    axes[0,1].set_title('Post-lockdown\npeak incidence')
    axes[0,1].set_yticks(range(0,N,tick_space),[f'{parameter_values[y_n,0]:.0f}' for y_n in range(0,N,tick_space)])
    axes[0,1].set_xticks(range(0,N,tick_space),[f'{parameter_values[x_n,1]:.2f}' for x_n in range(0,N,tick_space)])
    cbar = plt.colorbar(im01, ax=axes[0,1])
    cbar.set_label('Observed incidence')

    im10 = axes[1,0].imshow(pre_space)
    axes[1,0].set_title('Pre-lockdown\nperiodicity')
    axes[1,0].set_yticks(range(0,N,tick_space),[f'{parameter_values[y_n,0]:.0f}' for y_n in range(0,N,tick_space)])
    axes[1,0].set_xticks(range(0,N,tick_space),[f'{parameter_values[x_n,1]:.2f}' for x_n in range(0,N,tick_space)])
    cbar = plt.colorbar(im10, ax=axes[1,0])

    im11 = axes[1,1].imshow(post_space)
    axes[1,1].set_title('Time to post-\nlockdown rebound')
    axes[1,1].set_yticks(range(0,N,tick_space),[f'{parameter_values[y_n,0]:.0f}' for y_n in range(0,N,tick_space)])
    axes[1,1].set_xticks(range(0,N,tick_space),[f'{parameter_values[x_n,1]:.2f}' for x_n in range(0,N,tick_space)])
    cbar = plt.colorbar(im11, ax=axes[1,1])
    cbar.set_label('Time (months)')

def susc_grid_plot(axes,results,params,T_LOCKDOWN,LOCKDOWN_DURATION,
grid_params=(("BETA","REC"),("WANE_UP","WANE_SAME")),grid_mode=("scale","scale"),factors=(1,1),
label_mode=("diff_mean","nz_mean")):
    N_params = len(grid_params)
    N = max([max(p) for p in results.keys()])+1
    # Grid plots of susceptibility
    pre_susc = np.zeros((N,N))
    post_susc = np.zeros((N,N))
    parameter_values = np.zeros((N,N_params))

    for p_n,result in results.items():
        N_S = params["N_S"]
        params_n = params.copy()
        for i,p in enumerate(p_n):
            for pname in grid_params[i]:
                if grid_mode[i] == "scale":
                    params_n[pname] = params[pname]*(1+factors[i]*(p/N-1/2))
                elif grid_mode[i] == "fade_vec":
                    N_S = params["N_S"]
                    params_n[pname] = np.array([1-i*factors[i]*(p/(N_span*N_S)) for i in range(N_S)])
            if label_mode[i]=="diff_mean":
                parameter_values[p_n[i],i] = np.mean(params_n[grid_params[i][0]]) - np.mean(params_n[grid_params[i][1]])
            elif label_mode[i]=="mean":
                parameter_values[p_n[i],i] = np.mean([np.mean(params_n[pname]) for pname in grid_params[i]])
            elif label_mode[i]=="nz_mean":
                parameter_values[p_n[i],i] = np.mean([np.mean(params_n[pname])*(len(params_n[pname])/np.sum(params_n[pname]==0)) for pname in grid_params[i]])
        NAG, S_REL, S_AGE = params_n["NAG"], params_n["S_REL"], params_n["S_AGE"]
        # Calculate susceptibility
        sus = np.zeros((len(result.t),NAG))
        pop_size = np.sum(result.y,axis=0)
        for i_t,t in enumerate(result.t):
            for i in range(N_S):
                sus[i_t,:] += S_REL[i]*S_AGE*result.y[(N_C*i+1)*NAG:(N_C*i+2)*NAG,i_t]
        # absolute susceptibility when lockdown starts
        total_sus = np.sum(sus,axis=1)
        pre_susc[p_n] = total_sus[np.argmax(result.t>=T_LOCKDOWN)]
        # susceptibility when lockdown ends
        # rel_sus = total_sus/total_sus[np.argmax(result.t>=T_LOCKDOWN)]
        post_susc[p_n] = total_sus[np.argmax(result.t>=T_LOCKDOWN+LOCKDOWN_DURATION)]

    tick_space = N//4 + 1

    im0 = axes[0].imshow(pre_susc)
    axes[0].set_title('Susceptibility\nat lockdown start')
    axes[0].set_yticks(range(0,N,tick_space),[f'{parameter_values[y_n,0]:.0f}' for y_n in range(0,N,tick_space)])
    axes[0].set_xticks(range(0,N,tick_space),[f'{parameter_values[x_n,1]:.2f}' for x_n in range(0,N,tick_space)])
    cbar = plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(post_susc)
    axes[1].set_title('Susceptibility\nat lockdown end')
    axes[1].set_yticks(range(0,N,tick_space),[f'{parameter_values[y_n,0]:.0f}' for y_n in range(0,N,tick_space)])
    axes[1].set_xticks(range(0,N,tick_space),[f'{parameter_values[x_n,1]:.2f}' for x_n in range(0,N,tick_space)])
    cbar = plt.colorbar(im1, ax=axes[1])
    cbar.set_label('Effective susceptible population')
