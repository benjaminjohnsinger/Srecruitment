## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import itertools as it
import matplotlib.pyplot as plt

from SISn_ODEs import single_pathogen_deltas as deltas_SIS

def observations(result,params,OBS_AGE,incidence=False,N_C=2):
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params.values()
    obs = np.zeros((len(result.t),NAG))
    pop_size = np.sum(result.y,axis=0)
    for i_t,t in enumerate(result.t):
        foi = BETA*np.dot(contact(t),np.sum((np.array([result.y[(N_C*j+2)*NAG:(N_C*j+3)*NAG,i_t] for j in range(N_S)])*I_REL),axis=0))/pop_size[i_t]
        for i in range(N_S):
            obs[i_t,:] += OBS_AGE*P_OBS[i]*S_REL[i]*foi*result.y[(N_C*i+1)*NAG:(N_C*i+2)*NAG,i_t]
    if incidence:
        obs = np.sum(obs,axis=1)/pop_size
    return(obs)

def infections_by_age(result,params,N_C=2):
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params.values()
    infs = np.zeros((len(result.t),NAG))
    for i_t,t in enumerate(result.t):
        infs[i_t,:] = np.sum(np.array([BETA*result.y[(N_C*i+2)*NAG:(N_C*i+3)*NAG,i_t]*I_REL[i]*np.dot(contact(t),np.sum([S_REL[j]*result.y[(N_C*j+1)*NAG:(N_C*j+2)*NAG,i_t] for j in range(N_S)],axis=0)) for i in range(N_S)]),axis=0)
    return(infs)

def susceptibility(result,params,N_C=2):
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params.values()
    sus = np.zeros((len(result.t),NAG))
    for i_t,t in enumerate(result.t):
        for i in range(N_S):
            sus[i_t,:] += S_REL[i]*S_AGE*result.y[(N_C*i+1)*NAG:(N_C*i+2)*NAG,i_t]
    return(sus)
            

def lockdown_incidence_plot(ax,state0,params,OBS_AGE,period,points,T_LOCKDOWN,LOCKDOWN_DURATION,result=None,label='Observed cases',color='#648FFF',by_age=False,AGE_GROUP_NAMES=None,deltas=deltas_SIS):
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params.values()
    if result is None:
        result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=points,args=(params,))
    ## Calculate observations
    if by_age:
        obs = observations(result,params,OBS_AGE,incidence=Frue)
        cmap = plt.get_cmap('viridis')
        pop_size_by_age = np.array([np.sum(result.y[range(i_age,(3*N_S+1)*NAG,NAG),:],axis=0) for i_age in range(NAG)]).T
        for i_age in range(NAG):
            ax.plot(result.t,obs[:,i_age]/pop_size_by_age[:,i_age], label=AGE_GROUP_NAMES[i_age], color=cmap(i_age/(NAG-1)))
        mx = 1.1*np.max(np.max(obs/pop_size_by_age,axis=1)[np.argmax(result.t>T_LOCKDOWN-5*12):np.argmax(result.t>T_LOCKDOWN+LOCKDOWN_DURATION+5*12)])
    else:
        obs = observations(result,params,OBS_AGE,incidence=True)
        ax.plot(result.t, obs, label=label,color=color)
        mx = 1.1*np.max(obs[np.argmax(result.t>T_LOCKDOWN-5*12):np.argmax(result.t>T_LOCKDOWN+LOCKDOWN_DURATION+5*12)])
    return(mx)

def lockdown_incidence_format(ax,points,T_LOCKDOWN,LOCKDOWN_DURATION,mx,year_window=5):
    ax.set_xlim(T_LOCKDOWN-year_window*12,T_LOCKDOWN+LOCKDOWN_DURATION+year_window*12)
    ax.set_ylim(0,mx)
    ax.set_ylabel('Observed incidence')
    ax.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],0,mx,color='gray',alpha=0.2)
    ax.set_xticks(np.arange(T_LOCKDOWN-year_window*12,T_LOCKDOWN+LOCKDOWN_DURATION+year_window*12,12),[str(int(x)-year_window) for x in np.arange(0,year_window*2+1,1)])
    ax.set_xlabel('Time (years)')
    ax.set_title('Incidence of disease with 1-year lockdown')

def lockdown_susceptibility_plot(ax,state0,params,period,points,T_LOCKDOWN,result=None,label='Susceptible_population',color='#648FFF',relative=True,by_age=False,AGE_GROUP_NAMES=None,style='-',delta=deltas_SIS):
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params.values()
    if result is None:
        result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=points,args=(params,))
    ## Calculate susceptibility
    sus = susceptibility(result,params)
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

def age_infect_plot(ax,state0,params,AGE_GROUP_NAMES,period,points,T_LOCKDOWN,LOCKDOWN_DURATION,result=None,delts=deltas_SIS):
    cmap = plt.get_cmap('viridis')
    NAG, N_S, AGING_RATE, BIRTH_RATE, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, IMPORT, BETA, contact = params.values()
    if result is None:
        result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=points,args=(params,))
    ## Calculate rate of infections created by each age group
    infs = infections_by_age(result,params)
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
            grid_params=(("BETA","REC"),("WANE_UP","WANE_SAME")),grid_mode=("scale","scale"),N=10,factors=(1,1),deltas=deltas_SIS):
    N_params = len(grid_params)
    N_S = params["N_S"]
    results = {}
    for p_n in it.product(range(N),repeat=N_params):
        # Scale parameters for exploration
        params_n = params.copy()
        for i,p in enumerate(p_n):
            for pname in grid_params[i]:
                if grid_mode[i] == "scale":
                    params_n[pname] = params[pname]*(1+(p/N-1/2))**factors[i]
                elif grid_mode[i] == "fade_vec":
                    vec = np.array([(1-j*(p/(N*(N_S-1))))**factors[i] for j in range(N_S)])
                    vec = vec.reshape(params[pname].shape)
                    params_n[pname] = vec
        # Run simulation
        result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=points, args=(params_n,))
        results[p_n] = result
    return(results)

def param_line_plot(ax,results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE=None,
grid_params=("WANE_UP","WANE_SAME"),grid_mode="scale",factor=1,label_mode="mean",
y_values=("peak incidence","time to rebound"),y_labels=("Peak incidence","Time to rebound (years)"),x_label="Waning",
colors=("#648FFF","#DC267F")):
    N = max([max(p) for p in results.keys()])+1
    # Line plots of chosen value before and after lockdown
    values = np.zeros((N,len(y_values)))
    parameter_values = np.zeros(N)

    for p_n,result in results.items():
        N_S = params["N_S"]
        params_n = params.copy()
        print(p_n)
        for p in p_n:
            for pname in grid_params:
                if grid_mode == "scale":
                    params_n[pname] = params[pname]*(1+(p/N-1/2))**factor
                elif grid_mode == "fade_vec":
                    N_S = params["N_S"]
                    vec = np.array([(1-j*(p/(N*N_S)))**factor for j in range(N_S)])
                    vec = vec.reshape(params[pname].shape)
                    params_n[pname] = vec
            # Express summary of parameters as a single value
            if grid_mode == "fade_vec":
                parameter_values[p] = factor*p/(N*N_S)
            elif label_mode=="diff_mean":
                parameter_values[p] = np.mean(params_n[grid_params[0]]) - np.mean(params_n[grid_params[1]])
            elif label_mode=="mean":
                parameter_values[p] = np.mean([np.mean(params_n[pname]) for pname in grid_params])
            elif label_mode=="nz_mean":
                parameter_values[p] = np.mean([np.mean(params_n[pname])*(len(params_n[pname])/np.sum(params_n[pname]==0)) for pname in grid_params])
        # Calculate observations
        if ("peak incidence" in y_values) or ("time to rebound" in y_values) or ("rebound peak incidence"in y_values) or ("periodicity" in y_values):
            obs = observations(result,params_n,OBS_AGE,incidence=True)
            if "peak incidence" in y_values:
                values[p_n,[idx for idx in range(len(y_values)) if y_values[idx]=="peak incidence"]] = np.max(obs[(result.t>25*12) & (result.t<T_LOCKDOWN)])
            if "time to rebound" in y_values:
                post_peak_arg = np.argmax(obs[result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION)] > np.max(obs[(result.t>25*12) & (result.t<T_LOCKDOWN)])/2)
                values[p_n,[idx for idx in range(len(y_values)) if y_values[idx]=="time to rebound"]] = result.t[np.argmax(result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION))+post_peak_arg]-(T_LOCKDOWN+LOCKDOWN_DURATION)
            if "rebound peak incidence" in y_values:
                values[p_n,[idx for idx in range(len(y_values)) if y_values[idx]=="rebound peak incidence"]] = np.max(obs[result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION)])
            if "periodicity" in y_values:
                pre_obs = obs[(result.t>25*12) & (result.t<T_LOCKDOWN)]
                corr = np.correlate(pre_obs, pre_obs, mode='same')
                acorr = corr[len(pre_obs)//2 + 1:] / (pre_obs.var() * np.arange(len(pre_obs)-1, len(pre_obs)//2, -1))
                acorr = acorr + np.linspace(0.1, 0, len(acorr))
                lag = np.abs(acorr).argmax() + 1
                values[p_n,[idx for idx in range(len(y_values)) if y_values[idx]=="periodicity"]] = lag/12
        if "child infections" in y_values or "under-five infections" in y_values:
            infs = infections_by_age(result,params_n)
            if "child infections" in y_values:
                values[p_n,[idx for idx in range(len(y_values)) if y_values[idx]=="child infections"]] = (np.sum(infs[:,0:4],axis=1)/np.sum(infs,axis=1))[np.argmax(result.t>=T_LOCKDOWN)]
            if "under-five infections" in y_values:
                values[p_n,[idx for idx in range(len(y_values)) if y_values[idx]=="under-five infections"]] = (np.sum(infs[:,0:3],axis=1)/np.sum(infs,axis=1))[np.argmax(result.t>=T_LOCKDOWN)]
        if "pre-lockdown susceptibility" in y_values or "post-lockdown susceptibility" in y_values:
            sus = susceptibility(result,params_n)
            total_sus = np.sum(sus,axis=1)
            if "pre-lockdown susceptibility" in y_values:
                values[p_n,[idx for idx in range(len(y_values)) if y_values[idx]=="pre-lockdown susceptibility"]] = total_sus[np.argmax(result.t<=T_LOCKDOWN)]
            if "post-lockdown susceptibility" in y_values:
                values[p_n,[idx for idx in range(len(y_values)) if y_values[idx]=="post-lockdown susceptibility"]] = total_sus[np.argmax(result.t>=T_LOCKDOWN+LOCKDOWN_DURATION)]
    for i in range(len(y_values)):
        ax.plot(parameter_values,values[:,i],label=y_labels[i],color=colors[i])
    ax.set_xlabel(x_label)

def grid_plot(ax,results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE=None,
grid_params=(("BETA","REC"),("WANE_UP","WANE_SAME")),grid_mode=("scale","scale"),factors=(1,1),label_mode=("diff_mean","nz_mean"),
z_value="peak incidence",z_label="Observed incidence"):
    N_params = len(grid_params)
    N = max([max(p) for p in results.keys()])+1
    z_values = np.zeros((N,N))
    parameter_values = np.zeros((N,N_params))

    for p_n,result in results.items():
        N_S = params["N_S"]
        params_n = params.copy()
        for i,p in enumerate(p_n):
            for pname in grid_params[i]:
                if grid_mode[i] == "scale":
                    params_n[pname] = params[pname]*(1+(p/N-1/2))**factors[i]
                elif grid_mode[i] == "fade_vec":
                    vec = np.linspace(1,1-p, N_S)**factors[i]
                    vec = np.array([(1-j*(p/N_S))**factors[i] for j in range(N_S)])
                    vec = vec.reshape(params[pname].shape)
                    params_n[pname] = vec
            if grid_mode == "fade_vec":
                parameter_values[p,i] = factors[i]*p/(N*N_S)
            if label_mode[i]=="diff_mean":
                parameter_values[p,i] = np.mean(params_n[grid_params[i][0]]) - np.mean(params_n[grid_params[i][1]])
            elif label_mode[i]=="mean":
                parameter_values[p,i] = np.mean([np.mean(params_n[pname]) for pname in grid_params[i]])
            elif label_mode[i]=="nz_mean":
                parameter_values[p,i] = np.mean([np.mean(params_n[pname])*(len(params_n[pname])/np.sum(params_n[pname]==0)) for pname in grid_params[i]])
        # Calculate observations
        if (z_value == "peak incidence") or (z_value == "min incidence") or (z_value == "oscillation size") or (z_value == "time to rebound") or (z_value == "rebound peak incidence") or (z_value == "periodicity"):
            obs = observations(result,params_n,OBS_AGE,incidence=True)
            if z_value == "peak incidence":
                z_values[p_n] = np.max(obs[(result.t>25*12) & (result.t<T_LOCKDOWN)])
            if z_value == "min incidence":
                z_values[p_n] = np.min(obs[(result.t>25*12) & (result.t<T_LOCKDOWN)])
            if z_value == "oscillation size":
                pre_obs = obs[(result.t>25*12) & (result.t<T_LOCKDOWN)]
                z_values[p_n] = (np.max(pre_obs) - np.min(pre_obs))/np.mean(pre_obs)
            if z_value == "time to rebound":
                post_peak_arg = np.argmax(obs[result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION)] > np.max(obs[(result.t>25*12) & (result.t<T_LOCKDOWN)])/2)
                z_values[p_n] = result.t[np.argmax(result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION))+post_peak_arg]-(T_LOCKDOWN+LOCKDOWN_DURATION)
            if z_value == "rebound peak incidence":
                z_values[p_n] = np.max(obs[result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION)])
            if z_value == "periodicity":
                pre_obs = obs[(result.t>25*12) & (result.t<T_LOCKDOWN)]
                corr = np.correlate(pre_obs, pre_obs, mode='same')
                acorr = corr[len(pre_obs)//2 + 1:] / (pre_obs.var() * np.arange(len(pre_obs)-1, len(pre_obs)//2, -1))
                acorr = acorr + np.linspace(0.1, 0, len(acorr))
                lag = np.abs(acorr).argmax() + 1
                z_values[p_n] = lag/12
        if (z_value == "child infections") or (z_value == "under-five infections"):
            infs = infections_by_age(result,params_n)
            if z_value == "child infections":
                z_values[p_n] = (np.sum(infs[:,0:4],axis=1)/np.sum(infs,axis=1))[np.argmax(result.t>=T_LOCKDOWN)]
            if z_value == "under-five infections":
                z_values[p_n] = (np.sum(infs[:,0:3],axis=1)/np.sum(infs,axis=1))[np.argmax(result.t>=T_LOCKDOWN)]
        if (z_value == "pre-lockdown susceptibility") or (z_value == "post-lockdown susceptibility"):
            sus = susceptibility(result,params_n)
            total_sus = np.sum(sus,axis=1)
            if z_value == "pre-lockdown susceptibility":
                z_values[p_n] = total_sus[np.argmax(result.t>=T_LOCKDOWN)]
            if z_value == "post-lockdown susceptibility":
                z_values[p_n] = total_sus[np.argmax(result.t>=T_LOCKDOWN+LOCKDOWN_DURATION)]

    im = ax.imshow(z_values)
    ax.set_yticks(range(0,N,N//4+1),[f'{parameter_values[y_n,0]:.2f}' for y_n in range(0,N,N//4+1)])
    ax.set_xticks(range(0,N,N//4+1),[f'{parameter_values[x_n,1]:.2f}' for x_n in range(0,N,N//4+1)])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(z_label)