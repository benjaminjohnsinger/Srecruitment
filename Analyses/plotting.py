## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
from matplotlib import cm as colormaps
import matplotlib.patheffects as pe
from math import comb
import corner
import pickle
import colorsys
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController

from utils import date_to_t, t_to_date, calculate_population_size, susceptibility, infections_by_age, parameters_from_DE, observations

N_C = 2
N_S = 3
NAG = 7
from JAX_ODEs import deltas

##### General plotting parameters #####
hsv_colors = colormaps.hsv(-0.02+np.arange(7)/7)
hsv_colors[3] = colormaps.hsv((3/7)+0.04)
# '#ff0000', '#ffb700', '#6cff00', '#00ffc0', '#00bbff', '#1900ff', '#f300ff'

##### Simple line plots #####
def lockdown_incidence_plot(ax,state0,params,points,T_LOCKDOWN,solution=None,label='Observed cases',color='#648FFF',linewidth=1,alpha=1,by_age=False,AGE_GROUP_NAMES=None,relative=False,deltas=deltas,obs=None,times=None,start_t=date_to_t('2015-10-01'),end_t=date_to_t('2025-05-01'),factor=1,p_time_to_obs=[1]):
    if solution is None:
        term = ODETerm(deltas)
        solver = Dopri5()
        saveat = SaveAt(ts=points)
        step_controller = PIDController(rtol=1e-5, atol=1e-5)
        solution = diffeqsolve(
                            term, solver,
                            t0=0, t1=int(points[-1]), dt0=None, stepsize_controller=step_controller,
                            saveat=saveat, y0=state0.flatten(), args=params, 
                            max_steps=None,  
                            )
    if times is None:
        times = solution.ts
    values = solution.ys.T
    dates = [t_to_date(t) for t in times]
    start_index = np.argmin(times<=start_t)
    end_index = np.argmin(times<=end_t)
    if end_index <= start_index:
        end_index = len(times)

    trajectory = np.diff(values[-NAG:,:],axis=1).T
    expected_obs = np.sum([np.roll(trajectory,i,axis=0)*p_time_to_obs[i] for i in range(len(p_time_to_obs))],axis=0)

    if by_age:
        pop_size_by_age = calculate_population_size(values)[1:]
        obs = factor*expected_obs
        for i_age in range(NAG):
            ax.plot(dates[(start_index+1):end_index],obs[start_index:end_index,i_age]/pop_size_by_age[start_index:end_index,i_age], label=AGE_GROUP_NAMES[i_age], color=hsv_colors[i_age],linewidth=linewidth,alpha=alpha)
        mx = 1.1*np.max(np.max(obs/pop_size_by_age,axis=1)[start_index:end_index])
    else:
        obs = factor*np.sum(expected_obs,axis=1)/np.sum(values[:-NAG,:],axis=0)[1:]
        # obs = factor*np.sum(expected_obs,axis=1)
        if relative:
            pre_mx = np.max(obs[start_index:np.argmin(times<=T_LOCKDOWN)])
            ax.plot(dates[(start_index+1):end_index], obs[start_index:end_index]/pre_mx, label=label,color=color,linewidth=linewidth,alpha=alpha)
            mx = 1.1*np.max(obs[start_index:end_index])/pre_mx
        else:
            ax.plot(dates[(start_index+1):end_index], obs[start_index:end_index], label=label,color=color,linewidth=linewidth,alpha=alpha)
            mx = 1.1*np.max(np.array(obs[start_index:end_index]))
    return(mx)

def lockdown_incidence_format(ax,T_LOCKDOWN,LOCKDOWN_DURATION,mx,year_window=5,year_skip=1,title='Incidence of disease'):
    # ax.set_xlim(T_LOCKDOWN-year_window*365,T_LOCKDOWN+LOCKDOWN_DURATION+year_window*365)
    # ax.set_ylim(0,mx)
    ax.set_ylabel('Incidence per 10k')
    # ax.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],0,mx,color='gray',alpha=0.2)
    # ax.set_xticks(np.arange(T_LOCKDOWN-year_window*365,T_LOCKDOWN+LOCKDOWN_DURATION+year_window*365+365,365*year_skip),[str(int(x)-year_window-1) for x in np.arange(0,year_window*2+2,year_skip)])
    ax.set_xlabel('Time (years)')
    ax.set_title(title)

def prevalence_plot(ax,state0,params,points,obs_age=None,solution=None,label='Observed cases',color='#648FFF',linewidth=1,alpha=1,by_age=False,AGE_GROUP_NAMES=None,deltas=deltas,times=None,start_t=date_to_t('2015-10-01'),end_t=date_to_t('2025-05-01')):
    if solution is None:
        term = ODETerm(deltas)
        solver = Dopri5()
        saveat = SaveAt(ts=points)
        step_controller = PIDController(rtol=1e-5, atol=1e-5)
        solution = diffeqsolve(
                            term, solver,
                            t0=0, t1=int(points[-1]), dt0=None, stepsize_controller=step_controller,
                            saveat=saveat, y0=state0.flatten(), args=params, 
                            max_steps=None,  
                            )
    if times is None:
        times = solution.ts
    values = solution.ys.T
    dates = [t_to_date(t) for t in times]
    start_index = np.argmin(times<=start_t)
    end_index = np.argmin(times<=end_t)
    if end_index <= start_index:
        end_index = len(times)
    
    if obs_age is None:
        obs_age = np.ones(NAG)

    population_size = calculate_population_size(values, N_S=N_S, NAG=NAG)
    infectious = jnp.sum(values[1:].reshape((2*N_S+1, NAG, -1))[1:2*N_S:2], axis=0).T
    if by_age:
        expected_infectious = jax.nn.softplus(infectious[start_index:end_index]*100)/100
        expected_prevalence = jnp.divide(expected_infectious, population_size[start_index:end_index] * obs_age)
        for i_age in range(NAG):
            ax.plot(dates[start_index:end_index],expected_prevalence[:,i_age], label=AGE_GROUP_NAMES[i_age], color=hsv_colors[i_age],linewidth=linewidth,alpha=alpha)
        mx = 1.1*np.max(np.max(expected_prevalence,axis=1))
    else:
        expected_infectious = jax.nn.softplus((infectious[start_index:end_index] / obs_age).sum(axis=1)*100)/100
        expected_prevalence = jnp.divide(expected_infectious, population_size[start_index:end_index].sum(axis=1))
        ax.plot(dates[start_index:end_index], expected_prevalence, label=label,color=color,linewidth=linewidth,alpha=alpha)
        mx = 1.1*np.max(expected_prevalence)
    return(mx)


def lockdown_susceptibility_plot(ax,state0,params,period,points,T_LOCKDOWN,solution=None,label='Susceptible_population',color='#648FFF',relative=True,proportion=False,by_age=False,AGE_GROUP_NAMES=None,style='-',delta=deltas):
    NAG = 7
    N_S = 3
    if solution is None:
        term = ODETerm(deltas)
        solver = Dopri5()
        saveat = SaveAt(ts=POINTS)
        step_controller = PIDController(rtol=1e-5, atol=1e-5)
        solution = diffeqsolve(
                            term, solver,
                            t0=0, t1=int(POINTS[-1]), dt0=None, stepsize_controller=step_controller,
                            saveat=saveat, y0=STATE0.flatten(), args=params, 
                            max_steps=None,  
                            )
    times = solution.ts
    values = solution.ys.T
    dates = [t_to_date(t) for t in times]
    ## Calculate susceptibility
    sus = susceptibility(solution,params)
    if by_age:
        if proportion:
            pop_by_age = calculate_population_size(values)
            sus = sus/pop_by_age
        for i in range(NAG):
            ax.plot(dates,sus[:,i], label=AGE_GROUP_NAMES[i], color=hsv_colors[i],linestyle=style)
    else:
        total_sus = np.sum(sus,axis=1)
        if relative:
            pre_mx_sus = np.mean(total_sus[np.argmax(times>T_LOCKDOWN-5*365):np.argmax(times>T_LOCKDOWN)])
            rel_sus = total_sus/pre_mx_sus
            ax.plot(dates,rel_sus, label=label,color=color,linestyle=style)
        else:
            ax.plot(dates,total_sus, label=label,color=color,linestyle=style)

def lockdown_susceptibility_format(ax,T_LOCKDOWN,LOCKDOWN_DURATION,ymin=0.875,ymax=1.1,year_window=5):
    # ax.set_xlim(T_LOCKDOWN-year_window*365,T_LOCKDOWN+LOCKDOWN_DURATION+year_window*365)
    # ax.set_xticks(np.arange(T_LOCKDOWN-year_window*365,T_LOCKDOWN+LOCKDOWN_DURATION+year_window*365,365),[str(int(x)-year_window) for x in np.arange(0,2*year_window+1,1)])
    ax.set_xlabel('Time (years)')
    ax.set_title('Population susceptibility')
    yin, yax = ax.get_ylim()
    ax.set_ylabel('Effective susceptibles')
    if not (ymax is None or ymin is None):
        ax.set_ylim(ymin,ymax)
        ax.set_ylabel('Relative susceptibility')
    #     ax.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],ymin,ymax,color='gray',alpha=0.2)
    # else:
    #     ax.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],yin,yax,color='gray',alpha=0.2)

def age_infect_plot(ax,state0,params,AGE_GROUP_NAMES,period,points,T_LOCKDOWN,LOCKDOWN_DURATION,result=None,delts=deltas):
    cmap = plt.get_cmap('viridis')
    NAG, N_S, AGING_RATE, births, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, arrivals, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact = params.values()
    if result is None:
        result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=points,args=(params,))
    ## Calculate rate of infections created by each age group
    infs = infections_by_age(result,params)
    rel_infs = infs/np.sum(infs,axis=1)[:,np.newaxis]
    for i in range(NAG):
        ax.plot(result.t,rel_infs[:,i], label=AGE_GROUP_NAMES[i], color=cmap(i/(NAG-1)),zorder=1)
    ax.set_xlim(T_LOCKDOWN-5*365,T_LOCKDOWN+LOCKDOWN_DURATION+5*365)
    ax.set_title('Infections caused by each age group')
    ax.set_ylabel('Infection rate')
    ax.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],0,1,color='gray',zorder=0,alpha=0.2)
    ax.set_xticks(np.arange(T_LOCKDOWN-5*365,T_LOCKDOWN+LOCKDOWN_DURATION+5*365,365),[str(int(x)-5) for x in np.arange(0,11,1)])
    ax.set_xlabel('Time (years)')
    ax.set_ylim(0,1)
    # ax.legend()

##### Sim grid based plots #####
def param_line_plot(ax,results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE=None,
    grid_params=("WANE_UP","WANE_SAME"),grid_mode="scale",factor=1,label_mode="mean",
    y_values=("peak incidence","time to rebound"),y_labels=("Peak incidence","Time to rebound (years)"),x_label="Waning",
    colors=("#648FFF","#DC267F")):
    N = max([max(p) for p in results.keys()])+1
    # Line plots of chosen value before and after lockdown
    values = np.zeros((N,len(y_values)))
    parameter_values = np.zeros(N)

    for p_n,result in results.items():
        params_n = params.copy()
        for p in p_n:
            for pname in grid_params:
                if grid_mode == "scale":
                    params_n[pname] = params[pname]*(1+(p/N-1/2))**factor
                elif grid_mode == "fade_vec":
                    vec_len = len(params[pname])
                    vec = np.array([(1-j*(p/(N*(vec_len-1))))**factor for j in range(vec_len)])
                    vec = vec.reshape(params[pname].shape)
                    params_n[pname] = vec
            # Express summary of parameters as a single value
            if grid_mode == "fade_vec":
                parameter_values[p] = factor*p/(N*(vec_len-1))
            elif label_mode=="diff_mean":
                parameter_values[p] = np.mean(params_n[grid_params[0]]) - np.mean(params_n[grid_params[1]])
            elif label_mode=="mean":
                parameter_values[p] = np.mean([np.mean(params_n[pname]) for pname in grid_params])
            elif label_mode=="nz_mean":
                parameter_values[p] = np.mean([np.mean(params_n[pname])*(len(params_n[pname])/np.sum(params_n[pname]!=0)) for pname in grid_params])
        # Calculate observations
        if ("peak incidence" in y_values) or ("time to rebound" in y_values) or ("rebound peak incidence"in y_values) or ("periodicity" in y_values):
            obs = observations(result,params_n,OBS_AGE,incidence=True)
            if "peak incidence" in y_values:
                values[p_n,[idx for idx in range(len(y_values)) if y_values[idx]=="peak incidence"]] = np.max(obs[(result.t>T_LOCKDOWN-12*365) & (result.t<T_LOCKDOWN)])
            if "time to rebound" in y_values:
                post_peak_arg = np.argmax(obs[result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION)] > np.max(obs[(result.t>T_LOCKDOWN-12*365) & (result.t<T_LOCKDOWN)])/2)
                val = result.t[np.argmax(result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION))+post_peak_arg]-(T_LOCKDOWN+LOCKDOWN_DURATION)
                values[p_n,[idx for idx in range(len(y_values)) if y_values[idx]=="time to rebound"]] = val/365
            if "rebound peak incidence" in y_values:
                values[p_n,[idx for idx in range(len(y_values)) if y_values[idx]=="rebound peak incidence"]] = np.max(obs[result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION)])
            if "periodicity" in y_values:
                pre_obs = obs[(result.t>T_LOCKDOWN-12*365) & (result.t<T_LOCKDOWN)]
                corr = np.correlate(pre_obs, pre_obs, mode='same')
                acorr = corr[len(pre_obs)//2 + 1:] / (pre_obs.var() * np.arange(len(pre_obs)-1, len(pre_obs)//2, -1))
                acorr = acorr + np.linspace(0.1, 0, len(acorr))
                lag = np.abs(acorr).argmax() + 1
                values[p_n,[idx for idx in range(len(y_values)) if y_values[idx]=="periodicity"]] = lag/365
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
    z_value="peak incidence",z_label="Observed incidence",x_labels=("Growth","Waning"),
    save=False,file=None,fix=False,vmin=None,vmax=None):
    N_params = len(grid_params)
    N = max([max(p) for p in results.keys()])+1
    z_values = np.zeros(np.repeat(N,N_params))
    parameter_values = np.zeros((N,N_params))
    if save or (file is None):
        for p_n,result in results.items():
            if all([p==0 for p in p_n[1:]]):
                print(p_n)
            params_n = params.copy()
            for i,p in enumerate(p_n):
                for pname in grid_params[i]:
                    if grid_mode[i] == "scale":
                        params_n[pname] = params[pname]*(1+(p/N-1/2))**factors[i]
                    elif grid_mode[i] == "fade_vec":
                        vec_len = len(params[pname])
                        vec = np.array([(1-j*(p/(N*(vec_len-1))))**factors[i] for j in range(vec_len)])
                        vec = vec.reshape(params[pname].shape)
                        params_n[pname] = vec
                if grid_mode[i] == "fade_vec":
                    parameter_values[p,i] = (p/(N*(vec_len-1)))**factors[i]
                if label_mode[i]=="diff_mean":
                    parameter_values[p,i] = np.mean(params_n[grid_params[i][0]]) - np.mean(params_n[grid_params[i][1]])
                elif label_mode[i]=="mean":
                    parameter_values[p,i] = np.mean([np.mean(params_n[pname]) for pname in grid_params[i]])
                elif label_mode[i]=="nz_mean":
                    parameter_values[p,i] = np.mean([np.mean(params_n[pname])*(len(params_n[pname])/np.sum(params_n[pname]!=0)) for pname in grid_params[i]])
            # Calculate observations
            if (z_value == "peak incidence") or (z_value == "min incidence") or (z_value == "oscillation size") or (z_value == "time to rebound") or (z_value == "rebound peak incidence") or (z_value == "periodicity"):
                obs = observations(result,params_n,OBS_AGE,incidence=True)
                if z_value == "peak incidence":
                    z_values[p_n] = np.max(obs[(result.t>T_LOCKDOWN-12*365) & (result.t<T_LOCKDOWN)])
                if z_value == "min incidence":
                    z_values[p_n] = np.min(obs[(result.t>T_LOCKDOWN-12*365) & (result.t<T_LOCKDOWN)])
                if z_value == "oscillation size":
                    pre_obs = obs[(result.t>T_LOCKDOWN-12*365) & (result.t<T_LOCKDOWN)]
                    z_values[p_n] = (np.max(pre_obs) - np.min(pre_obs))/np.mean(pre_obs)
                if z_value == "time to rebound":
                    post_peak_arg = np.argmax(obs[result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION)] > np.max(obs[(result.t>T_LOCKDOWN-12*365) & (result.t<T_LOCKDOWN)])/2)
                    z_values[p_n] = (result.t[np.argmax(result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION))+post_peak_arg]-(T_LOCKDOWN+LOCKDOWN_DURATION))/365
                if z_value == "rebound peak incidence":
                    z_values[p_n] = np.max(obs[result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION)])
                if z_value == "periodicity":
                    pre_obs = obs[(result.t>T_LOCKDOWN-12*365) & (result.t<T_LOCKDOWN)]
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
    if save:
        with open(file,'wb') as f:
            pickle.dump((parameter_values,z_values),f)
    if file is not None:
        with open(file,'rb') as f:
            parameter_values,z_values = pickle.load(f)
    if N_params == 2:
        im = ax.imshow(np.flipud(z_values))
        ax.set_yticks(range(0,N,N//4+1),[f'{parameter_values[y_n,0]:.2g}' for y_n in range(N-1,-1,-(N//4+1))])
        ax.set_xticks(range(0,N,N//4+1),[f'{parameter_values[x_n,1]:.2g}' for x_n in range(0,N,N//4+1)])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(z_label)
    else:
        ax_width = ax.shape[1]
        for i in range(N_params):
            for j in range(i+1,N_params):
                axn = sum((N_params-k) for k in range(i+1))+j-i-N_params-1
                axis = ax[axn//ax_width,axn%ax_width]
                if fix:
                    # get z_values with all indixes equal to N//2 except for i and j
                    z_values_plot = np.zeros((N,N))
                    for i_n in range(N):
                        for j_n in range(N):
                            indices = [N//2]*N_params
                            indices[i] = i_n
                            indices[j] = j_n
                            z_values_plot[i_n,j_n] = z_values[tuple(indices)]
                else:  
                    z_values_plot = np.mean(z_values,axis=tuple([k for k in range(N_params) if k not in [i,j]]))
                if (vmax is not None) and (vmin is not None):
                    im = axis.imshow(np.flipud(z_values_plot),vmin=vmin,vmax=vmax)
                else:
                    im = axis.imshow(np.flipud(z_values_plot))
                    cbar = plt.colorbar(im, ax=axis)
                axis.set_yticks(range(0,N,N//4+1),[f'{parameter_values[y_n,i]:.2g}' for y_n in range(N-1,-1,-(N//4+1))])
                axis.set_xticks(range(0,N,N//4+1),[f'{parameter_values[x_n,j]:.2g}' for x_n in range(0,N,N//4+1)])
                axis.set_xlabel(x_labels[j])
                axis.set_ylabel(x_labels[i])
        return(im)

##### Clustering plots #####
def cluster_plot(axes,results,obses,n_clusters,labels,cluster_centers,relative=False,color=False,line=True,clusters=None, color_values_all=None,
    parameters=["BETA","WANE","S_REL"],param_labels=["Infectiousness","Waning","Acquired\nimmunity"],
    grid_mode=["scale","scale","based_vec"],base_values=[40,1/30,1/4],factors=[0.7,3,1],N=25,
    y_value=("time to rebound"),y_label="Time to rebound",
    t_lockdown="2014-01-01",LOCKDOWN_DURATION=365):
    if clusters is None:
        clusters = set(labels)
    T_LOCKDOWN = date_to_t(pd.to_datetime(t_lockdown))
    times = np.array(date_to_t(pd.date_range(start=pd.to_datetime(t_lockdown)-pd.Timedelta(5*365,unit='D'),end=pd.to_datetime(t_lockdown),freq='MS')))[:-1]
    param_values_all = np.zeros((len(results.keys()),len(parameters)))
    for i,p_n in enumerate(results.keys()):
        for n_p,param in enumerate(parameters):
            if grid_mode[n_p] == "scale":
                param_values_all[i,n_p] = base_values[n_p]*(1+(p_n[n_p]/N-1/2))**factors[n_p]
            elif grid_mode[n_p] == "fade_vec":
                param_values_all[i,n_p] = p_n[n_p]/(2*N)
            elif grid_mode[n_p] == "based_vec":
                param_values_all[i,n_p] = base_values[n_p]+(1/2-base_values[n_p])*p_n[n_p]/N
    for i,cluster in enumerate(clusters):
        idx = np.where(labels==cluster)[0]
        values = np.zeros(len(idx))
        mx = 0
        param_values = param_values_all[idx,:]
        if color:
            if color_values_all is None:
                color_values = 0.95*(param_values-np.min(param_values_all,axis=0))/(np.max(param_values_all,axis=0)-np.min(param_values_all,axis=0))
            else:
                color_values = color_values_all[idx]
        for n_j,j in enumerate(idx):
            result = results[list(results.keys())[j]]
            obs = 100*obses[list(obses.keys())[j]]
            p_n = list(results.keys())[j]
            if y_value == "rebound peak incidence":
                values[n_j] = np.max(obs[result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION)])
            elif y_value == "time to rebound":
                post_peak_arg = np.argmax(obs[result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION)] > np.max(obs[(result.t>T_LOCKDOWN-5*365) & (result.t<T_LOCKDOWN)])/2)
                val = result.t[np.argmax(result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION))+post_peak_arg]-(T_LOCKDOWN+LOCKDOWN_DURATION)
                values[n_j] = min(val/365,5)
            elif y_value == "child infections":
                infs = infections_by_age(result,params)
                values[n_j] = (np.sum(infs[:,0:4],axis=1)/np.sum(infs,axis=1))[np.argmax(result.t>=T_LOCKDOWN)]
            if color:
                if np.random.rand() < 500/len(idx):
                    mxs = lockdown_incidence_plot(axes[i,0],None,None,None,None,None,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,obs=obs,relative=relative,
                    start_t=date_to_t('2009-01-01'),end_t=date_to_t('2020-01-01'),
                    color=color_values[n_j],alpha=1)
                    mx = max(mx,mxs)
            else:
                mxs = lockdown_incidence_plot(axes[i,0],None,None,None,None,None,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,obs=obs,relative=relative,
                start_t=date_to_t('2009-01-01'),end_t=date_to_t('2020-01-01'),
                color='black',alpha=0.01)
                mx = max(mx,mxs)
        if cluster_centers is not None:
            if color:
                axes[i,0].plot(times,100*cluster_centers[cluster],color='black',label='Cluster center')
            else:
                axes[i,0].plot(times,100*cluster_centers[cluster],color='red',label='Cluster center')
        lockdown_incidence_format(axes[i,0],T_LOCKDOWN,LOCKDOWN_DURATION,mx,title='',year_skip=2)
        axes[i,0].set_xlabel("")
        for n_p,param in enumerate(parameters):
            if color and not line:
                jitter_param = np.random.normal(-1,1,len(param_values[:,n_p]))*base_values[n_p]/(2*N)
                jitter_values = np.random.normal(-1,1,len(param_values[:,n_p]))*(1/24)
                axes[i,n_p+1].scatter(param_values[:,n_p]+jitter_param,values+jitter_values,c=color_values,alpha=1,s=15/np.sqrt(len(idx)),linewidths=0)
            elif not line:
                axes[i,n_p+1].scatter(param_values[:,n_p],values,color="black",alpha=0.3,s=10)
            else:
                pf = param_values[:,n_p]
                Qs = np.zeros((len(pf),3))
                for pidx in range(len(pf)):
                    Qs[pidx,:] = np.percentile(values[pf==pf[pidx]],[25,50,75])
                sort_args = np.argsort(pf)
                param_sorted = pf[sort_args]
                Qs_sorted = Qs[sort_args,:]
                if color:
                    axes[i,n_p+1].plot(param_sorted,Qs_sorted[:,1],color=color_values[n_j])
                    axes[i,n_p+1].fill_between(param_sorted,Qs_sorted[:,0],Qs_sorted[:,2],alpha=0.3,color=color_values[n_j])
                else:
                    axes[i,n_p+1].plot(param_sorted,Qs_sorted[:,1],color="black")
                    axes[i,n_p+1].fill_between(param_sorted,Qs_sorted[:,0],Qs_sorted[:,2],alpha=0.3,color="black")
            if n_p > 0:
                axes[i,n_p+1].set_yticklabels([])
        if n_clusters > 1:
            axes[i,0].set_ylabel(f"Cluster {cluster+1}\n\nIncidence")
        else:
            axes[i,0].set_ylabel(f"All simulations\n\nIncidence")
        axes[i,1].set_ylabel("\n"+y_label)
    if n_clusters > 1:
        axes[len(clusters)-1,0].set_xlabel("Time (years)")
        for n_p,label in enumerate(param_labels):
            axes[len(clusters)-1,n_p+1].set_xlabel(f"{label}")

##### Fitting plots #####
def mcmc_trajectory_plot(axes,trajectory,param_names):
    for i in range(trajectory.shape[1]):
        axes[i].plot(trajectory[:,i])
        axes[i].set_ylabel(param_names[i])

def mcmc_corner_plot(trajectory,param_names,burn_in):
    fig = corner.corner(trajectory[burn_in:,:],labels=param_names,quantiles=[0.16,0.5,0.84],show_titles=True)

##### KPSC data plots #####
pathogen_names = {"RSV": ["RESPIRATORY SYNCYTIAL VIRUS","RESPIRATORY SYNCYTIAL VIRUS SUBTYPE A","RESPIRATORY SYNCYTIAL VIRUS SUBTYPE B"],
"InfluenzaA": ["INFLUENZA A","INFLUENZA A H1N1 2009","INFLUENZA A VIRUS","INFLUENZA A VIRUS SUBTYPE H1","INFLUENZA A VIRUS SUBTYPE/HEMAGGLUTININ H3","INFLUENZA VIRUS A","INFLUENZA VIRUS A+B"],
# "InfluenzaAH1": ["INFLUENZA A H1N1 2009","INFLUENZA A VIRUS SUBTYPE H1"],
# "InfluenzaAH3": ["INFLUENZA A VIRUS SUBTYPE/HEMAGGLUTININ H3"],
"InfluenzaB": ["INFLUENZA B","INFLUENZA VIRUS B","INFLUENZA VIRUS A+B"],
# "Influenza": ["INFLUENZA A","INFLUENZA A H1N1 2009","INFLUENZA A VIRUS","INFLUENZA A VIRUS SUBTYPE H1","INFLUENZA A VIRUS SUBTYPE/HEMAGGLUTININ H3","INFLUENZA VIRUS A","INFLUENZA VIRUS A+B","INFLUENZA B","INFLUENZA VIRUS B"],
"Metapneumovirus": ["HUMAN METAPNEUMOVIRUS VIRUS",],
# "HMPV" : ["HUMAN METAPNEUMOVIRUS VIRUS",],
"Adenovirus": ["ADENOVIRUS",],
# "Parainfluenza": ["PARAINFLUENZA VIRUS 1","PARAINFLUENZA VIRUS 2","PARAINFLUENZA VIRUS 3","PARAINFLUENZA VIRUS 4"],
"Parainfluenza3": ["PARAINFLUENZA VIRUS 3"],
"Rhinovirus": ["ENTEROVIRUS/RHINOVIRUS"],
"Pertussis": ["BORDETELLA PERTUSSIS"],
"M.pneumoniae": ["MYCOPLASMA PNEUMONIAE"],
"C.pneumoniae": ["CHLAMYDOPHILA PNEUMONIAE"],
"SARS-CoV-2": ["SARS-COV-2 (COVID-19)"],
"Enterovirus": ["ENTEROVIRUS/RHINOVIRUS"],
}
reverse_names = [{v:k for v in values} for k,values in pathogen_names.items()]
reverse_names =  {k:v for d in reverse_names for k,v in d.items()}
import numpy as np
def kpsc_positive_test_plot(ax, hospitalizations=True, pathogen="RSV", AGE_GROUPS=None, AGE_GROUP_NAMES=None, NDI_GROUPS=None, NDI_GROUP_NAMES=None, incidence=False, color=hsv_colors, title=None, legend=True, aggregation=None, save_data=False, load_data=False):
    print(pathogen)
    if load_data:
        if pathogen == "test":
            cases = pd.read_csv(f'Data/Processed/KPSC_ARI_test_{["cases","incidence"][incidence]}_{["all","age"][AGE_GROUPS is not None]}_{["daily","weekly","monthly"][[None,"Week","Month"].index(aggregation)]}.csv',index_col=0,parse_dates=True)
        else:
            cases = pd.read_csv(f'Data/Processed/KPSC_ARI_{pathogen}_{["cases","incidence"][incidence]}_{["all","age"][AGE_GROUPS is not None]}_{["daily","weekly","monthly"][[None,"Week","Month"].index(aggregation)]}.csv',index_col=0,parse_dates=True)
    else:
        if incidence:
            # load age population data
            age_by_month = pd.read_csv("Data/Processed/KPSC_population_by_age_group_monthly.csv")
            age_by_month['month_start'] = pd.to_datetime(age_by_month['month_start'])
            age_by_month = age_by_month.set_index("month_start")
            
            # # load ndi population data
            # ndi_by_year = pd.read_csv("Data/Processed/KPSC_population_by_ndi.csv")
            # ndi_by_year.columns = NDI_GROUP_NAMES
            # ndi_by_year.loc[:,"Year"] = np.arange(2015,2023)
            # ndi_by_year = ndi_by_year.set_index("Year")
            # for year in range(2023,2026):
            #     ndi_by_year.loc[year] = ndi_by_year.loc[2022]
        
        if hospitalizations:
            positive_tests = pd.read_csv('Data/Processed/KPSC_positive_matched_hospitalizations.csv')
        else:
            positive_tests = pd.read_csv('Data/Processed/KPSC_positive_matched_all_clinical_cleaned.csv')

        positive_tests.loc[:,"pathogen_class"] = positive_tests["pathogen"].map(reverse_names)
        cases = positive_tests[(positive_tests['pathogen_class'] == pathogen) & (positive_tests['dxgroup']=="ARI")]
        cases = cases.drop_duplicates(subset=cases.columns.difference(['CODE','dxgroup','pathogen']))
        if hospitalizations:
            cases.loc[:,"Date"] = pd.to_datetime(cases["Hospitalization date"])
        else:
            cases.loc[:,"Date"] = pd.to_datetime(cases["Clinical date"])
        print(cases.columns)
        print(cases["Date"].max())
        print(cases["ndi"].min(),cases["ndi"].max())
        print(cases["NDI"].min(),cases["NDI"].max())
        if aggregation is not None:
            cases.loc[:,"Year"] = cases["Date"].dt.year
            if aggregation == "Month":
                cases.loc[:,"Month"] = cases["Date"].dt.month
            elif aggregation == "Week":
                cases.loc[:,"Week"] = cases["Date"].dt.isocalendar().week
        if AGE_GROUPS is not None:
            for i in range(len(AGE_GROUPS)):
                cases.loc[cases["age_in_mo"].isin(AGE_GROUPS[i]),"age_group"] = AGE_GROUP_NAMES[i]
            if aggregation is None:
                cases = cases.groupby(["Date","age_group"]).size().reset_index(name='Count')
            else:
                cases = cases.groupby(["Year",aggregation,"age_group"]).size().reset_index(name='Count')
            print(cases["Count"].sum())
        elif NDI_GROUPS is not None:
            for i in range(len(NDI_GROUPS)):
                cases.loc[(cases["ndi"] >= NDI_GROUPS[i][0]) & (cases["ndi"] < NDI_GROUPS[i][-1]),"NDI_group"] = NDI_GROUP_NAMES[i]
            if aggregation is None:
                cases = cases.groupby(["Date","NDI_group"]).size().reset_index(name='Count')
            else:
                cases = cases.groupby(["Year",aggregation,"NDI_group"]).size().reset_index(name='Count')
        else:
            if aggregation is None:
                cases = cases.groupby(["Date"]).size().reset_index(name='Count')
            else:
                cases = cases.groupby(["Year",aggregation]).size().reset_index(name='Count')
        if aggregation is None:
            frequency = "D"
        elif aggregation == "Month":
            frequency = "MS"
        elif aggregation == "Week":
            frequency = "W-MON"

        for date in pd.date_range(start='2015-10-01',end='2025-05-01',freq=frequency):
            year = date.year
            if aggregation=="Month":
                agg = date.month
            elif aggregation=="Week":
                agg = date.isocalendar().week
            if AGE_GROUPS is not None:
                for age_group in AGE_GROUP_NAMES:
                    if aggregation is None:
                        if not ((cases["Date"]==date) & (cases["age_group"]==age_group)).any():
                            cases = pd.concat([cases,pd.DataFrame({"Date":[date],"age_group":[age_group],"Count":[0]})])
                    else:
                        if not (((cases["Year"]==year) & (cases[aggregation]==agg)) & (cases["age_group"]==age_group)).any():
                            cases = pd.concat([cases,pd.DataFrame({"Year":[year],aggregation:[agg],"age_group":[age_group],"Count":[0]})])
            elif NDI_GROUPS is not None:
                for ndi_group in NDI_GROUP_NAMES:
                    if aggregation is None:
                        if not ((cases["Date"]==date) & (cases["NDI_group"]==ndi_group)).any():
                            cases = pd.concat([cases,pd.DataFrame({"Date":[date],"NDI_group":[ndi_group],"Count":[0]})])
                    else:
                        if not (((cases["Year"]==year) & (cases[aggregation]==agg)) & (cases["NDI_group"]==ndi_group)).any():
                            cases = pd.concat([cases,pd.DataFrame({"Year":[year],aggregation:[agg],"NDI_group":[ndi_group],"Count":[0]})])
            if aggregation is None:
                if not (cases["Date"]==date).any():
                    cases = pd.concat([cases,pd.DataFrame({"Date":[date],"Count":[0]})])
            else:
                if not ((cases["Year"]==year) & (cases[aggregation]==agg)).any():
                    cases = pd.concat([cases,pd.DataFrame({"Year":[year],aggregation:[agg],"Count":[0]})])
        if aggregation is not None:
            if aggregation == "Month":
                cases.loc[:,"Date"] = pd.to_datetime(cases["Year"].astype(str) + '-' + cases["Month"].astype(str) + '-01')
            elif aggregation == "Week":
                cases.loc[:,"Date"] = cases["Year"].astype(str) + '-' + cases["Week"].astype(str)
                cases.loc[:,"Date"] = pd.to_datetime(cases["Date"].add('-1').astype(str),format='%Y-%W-%w')
        cases = cases.sort_values(by="Date")
        cases = cases.set_index("Date")

        if aggregation is not None:
            cases = cases.drop(columns=["Year",aggregation])
        if AGE_GROUPS is not None:
            cases = cases.pivot(columns="age_group",values="Count")
            cases = cases[AGE_GROUP_NAMES]
            if incidence:
                # Align population data with case dates
                cases_with_pop = cases.copy()
                for date_idx in cases.index:
                    # Find the closest month_start date in age_by_month
                    closest_month = age_by_month.index[age_by_month.index <= date_idx].max()
                    if pd.notna(closest_month):
                        cases_with_pop.loc[date_idx] = cases.loc[date_idx] / age_by_month.loc[closest_month].values
                cases = cases_with_pop
        elif NDI_GROUPS is not None:
            cases = cases.pivot(columns="NDI_group",values="Count")
            cases = cases[NDI_GROUP_NAMES]
            if incidence:
                cases = cases.div(ndi_by_year.loc[cases.index.year].values)
        elif incidence:
            # For non-age-group cases, sum all age groups from monthly data
            cases_with_pop = cases.copy()
            for date_idx in cases.index:
                closest_month = age_by_month.index[age_by_month.index <= date_idx].max()
                if pd.notna(closest_month):
                    total_pop = age_by_month.loc[closest_month].sum()
                    cases_with_pop.loc[date_idx, "Count"] = cases.loc[date_idx, "Count"] / total_pop
            cases = cases_with_pop

        if save_data:
            filename = f'Data/Processed/KPSC_ARI_{pathogen}_{["cases","incidence"][incidence]}_{["all","age"][AGE_GROUPS is not None]}_{["daily","weekly","monthly"][["D","W-MON","MS"].index(frequency)]}.csv'
            cases.to_csv(filename)

    if incidence:
        cases *= 10000
    if title is None:
        if incidence:
            title = f"{pathogen} incidence"
        else:
            title = f"{pathogen} positive cases"
    if AGE_GROUPS is not None:
        # cases.plot(ax=ax,legend=legend,color=color,label=AGE_GROUP_NAMES,title=f"{pathogen} positive tests")
        for i in range(len(AGE_GROUP_NAMES)):
            ax.plot(cases.index, cases[AGE_GROUP_NAMES[i]], label=AGE_GROUP_NAMES[i], color=color[i])
            # ax.set_xticks(cases.index)
            # ax.set_xticklabels([year if year % 2 == 0 else '' for year in cases.index.year], rotation=45)
    elif NDI_GROUPS is not None:
        for i in range(len(NDI_GROUP_NAMES)):
            ax.plot(cases.index, cases[NDI_GROUP_NAMES[i]], label=NDI_GROUP_NAMES[i], color=color[i])
    else:
        # cases.plot(ax=ax,legend=False,color=color,title=f"{pathogen} positive tests")
        ax.plot(cases.index,cases["Count"],color=color)
    ax.set_title(title)
    if incidence:
        ax.set_ylabel("Incidence per 10k members")
    else:
        ax.set_ylabel("Cases")
    if legend:
        # two column legend
        if AGE_GROUPS is not None:
            ax.legend(ncol=2,title="Age groups")
        elif NDI_GROUPS is not None:
            ax.legend(ncol=2,title="NDI groups")
    # ax.set_xlabel("Date")

nice_names = {"RSV": "RSV", "InfluenzaA": "Influenza A", "InfluenzaB": "Influenza B", "Metapneumovirus": "Metapneumovirus", "Adenovirus": "Adenovirus", "Parainfluenza3": "Parainfluenza 3", "Rhinovirus": "Rhinovirus", "Pertussis": "Pertussis", "M.pneumoniae": "M. pneumoniae", "C.pneumoniae": "C. pneumoniae", "SARS-CoV-2": "SARS-CoV-2", "Enterovirus": "Enterovirus"}
from data_processing import calculate_proportion_positive_incidence
def kpsc_proportion_positive_incidence_plot(ax, pathogen="RSV", AGE_GROUPS=None, AGE_GROUP_NAMES=None, title=None, color=hsv_colors, legend=True, aggregation="D", window_size=28, weighting_factor=0.5, label=None, factor=1000000, annotations=False, pp_only=False, hosp=False):
    print(pathogen)
    incidence = calculate_proportion_positive_incidence(pathogen, aggregation=aggregation, window_size=window_size, weighting_factor=weighting_factor, sum_age_groups=AGE_GROUPS is None, save_counts=False, pp_only=pp_only, hosp=hosp)
    incidence *= factor
    if aggregation == "MS":
        # add 14 days to the index to get the middle of the month
        incidence.index = incidence.index + pd.Timedelta(days=14)
    if AGE_GROUPS is not None:
        for i in range(len(AGE_GROUP_NAMES)):
            ax.plot(incidence.index, incidence[AGE_GROUP_NAMES[i]], label=AGE_GROUP_NAMES[i], color=color[i])
    else:
        if label is None:
            label = nice_names.get(pathogen, pathogen)
        ax.plot(incidence.index, incidence["Total"], color=color, label=label)
    if title is None:
        title = f"{nice_names.get(pathogen, pathogen)} estimated incidence of hospitalizations"
    ax.set_title(title)
    if factor >= 1000000:
        factor_label = f"{factor // 1000000}M"
    elif factor >= 1000:
        factor_label = f"{factor // 1000}k"
    else:
        factor_label = str(factor)
    ax.set_ylabel(f"Estimated incidence of hospitalizations per {factor_label} members")
    if legend:
        if AGE_GROUPS is not None:
            ax.legend(ncol=2,title="Age group")
    if annotations:
        if aggregation != "D":
            incidence = incidence.resample("D").interpolate()
        # take only first 9*365 values of incidence
        obs_per_season = calculate_observations_per_season(incidence)
        print(obs_per_season)
        peak_times = incidence["Total"].groupby(incidence.index.map(get_season_start)).idxmax()
        last_peak_time = peak_times[peak_times.index < pd.to_datetime("2020-03-19")].max()
        # find the rebound season - the first season after 2020-03-19 to read 50% of the total number of observations in the median season before 2020-03-19
        rebound_season = calculate_rebound_season(obs_per_season, definition="40% median")
        print(rebound_season)
        if rebound_season is None:
            # plot a line at 2020-03-19 and annotate that there was no rebound season by the end of the data, then quit
            ax.plot([last_peak_time, pd.to_datetime("2025-05-01")], [incidence.loc[last_peak_time, "Total"], incidence.loc[last_peak_time, "Total"]], color="black", linestyle="--")
            time_diff = pd.to_datetime("2025-05-01") - last_peak_time
            ax.annotate(f"{time_diff.days}+ days", xy=(last_peak_time + time_diff/2, incidence.loc[last_peak_time, "Total"]), xytext=(0,2), textcoords='offset points', ha='center', va="bottom", color="black",
                    path_effects=[pe.Stroke(linewidth=1, foreground='white'), pe.Normal()])
            return
        rebound_peak_time = peak_times.iloc[rebound_season]
        print(rebound_peak_time)
        # draw a line from the last peak before 2020-03-19 to the peak of the rebound season, and annotate the time between them
        time_diff = rebound_peak_time - last_peak_time
        ax.plot([last_peak_time, rebound_peak_time], [incidence.loc[rebound_peak_time, "Total"], incidence.loc[rebound_peak_time, "Total"]], color="black")
        ax.annotate(f"{time_diff.days} days", xy=(last_peak_time + time_diff/2, incidence.loc[rebound_peak_time, "Total"]), xytext=(0,2), textcoords='offset points', ha='center', va="bottom", color="black",
                    path_effects=[pe.Stroke(linewidth=1, foreground='white'), pe.Normal()])
        # # annotate the relative size of the rebound season compared to the median pre-covid season
        # rebound_size = obs_per_season[5:].max()
        # size_rebound = obs_per_season[5:].argmax() + 5
        # size_rebound_time = peak_times.iloc[size_rebound]
        # ax.annotate(f"{rebound_size/repr_pre_covid_obs:.0%}", xy=(size_rebound_time, incidence.loc[size_rebound_time, "Total"]), xytext=(0,0), textcoords='offset points', ha='left', va='top', color="black", 
        #             path_effects=[pe.Stroke(linewidth=1, foreground='white'), pe.Normal()])
    return np.max(incidence)

def get_season_start(date):
    # two weeks buffer because of strange behaviour in rebound
    if isinstance(date, pd.Timestamp):
        if date >= pd.to_datetime(f'{date.year}-09-17'):
            return pd.to_datetime(f'{date.year}-09-17')
        else:
            return pd.to_datetime(f'{date.year-1}-09-17')

def get_season_start_jax(date):
    threshold = (date//365)*365 + 259
    return threshold + jnp.where(date >= threshold, 0, -365)

def calculate_observations_per_season(incidence, age_groups=False):
    # pad with two weeks's worth of zeros at the start
    incidence = pd.concat([pd.DataFrame(0, index=pd.date_range(end=incidence.index[0]-pd.Timedelta(days=1), periods=14, freq='D'), columns=incidence.columns), incidence])
    if age_groups:
        obs_per_season = incidence.values[:9*365].reshape((9,365,-1)).sum(axis=1)
        # include last incomplete season (2024/25) up to 2025-05-01
        last_season_start = pd.to_datetime('2024-09-17')
        last_season_data = incidence[last_season_start:].values.sum(axis=0)
        obs_per_season = np.vstack([obs_per_season, last_season_data])
    else:
        obs_per_season = incidence.values[:9*365].reshape((9,365,-1)).sum(axis=1).sum(axis=1)
        # include last incomplete season (2024/25) up to 2025-05-01
        last_season_start = pd.to_datetime('2024-09-17')
        last_season_data = incidence[last_season_start:].values.sum()
        obs_per_season = np.append(obs_per_season, last_season_data)
    return obs_per_season


def calculate_observations_per_season_jax(incidence, age_groups=False, pad_days=14, n_full_seasons=9, season_len=365):
    """
    JAX-compatible version for traced arrays.
    Expects daily incidence starting at 2015-10-01 (or equivalent offset).
    Uses fixed-size seasonal blocks and appends one final incomplete season.
    """
    x = jnp.asarray(incidence)

    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError("incidence must have shape (time,) or (time, n_age_groups).")

    # pad = jnp.zeros((pad_days, n_age), dtype=x.dtype)
    # x = jnp.concatenate([pad, x], axis=0)

    full_days = n_full_seasons * season_len
    full = x[:full_days].reshape(n_full_seasons, season_len, 7).sum(axis=1)   # (n_full_seasons, n_age)
    last = x[full_days:].sum(axis=0, keepdims=True)                                # (1, n_age)
    out = jnp.concatenate([full, last], axis=0)                                    # (n_full_seasons+1, n_age)

    if age_groups:
        return out
    return out.sum(axis=1)

def calculate_rebound_season(obs_per_season, definition = "40% median"):
    if definition == "40% median":
        repr_pre_covid_obs = np.median(obs_per_season[:5]) * 0.4
    elif definition == "50% median":
        repr_pre_covid_obs = np.median(obs_per_season[:5]) * 0.5
    elif definition == "third median":
        repr_pre_covid_obs = np.median(obs_per_season[:5]) * (1/3)
    elif definition == "min":
        repr_pre_covid_obs = np.min(obs_per_season[:5])
    elif definition == "second min":
        repr_pre_covid_obs = np.partition(obs_per_season[:5], 1)[1]
    elif definition == "mean minus std":
        repr_pre_covid_obs = np.mean(obs_per_season[:5]) - np.std(obs_per_season[:5])
    rebound_season = np.where(obs_per_season[5:] >= repr_pre_covid_obs)[0]
    if len(rebound_season) == 0:
        return None
    rebound_season = rebound_season[0] + 5
    return rebound_season

def age_group_incidence_plot(ax,pathogen,color="k",season=0, AGE_GROUPS=None, AGE_GROUP_NAMES=None, factor=100000, label=None):
    incidence = calculate_proportion_positive_incidence(pathogen, aggregation="D", window_size=1, weighting_factor=0, sum_age_groups=False, save_counts=False)
    incidence = incidence[AGE_GROUP_NAMES]
    pop_by_age_group_month = pd.read_csv('Data/Processed/KPSC_population_by_age_group_monthly.csv', index_col=0, parse_dates=['month_start'])
    pop_by_age_group_day = pop_by_age_group_month.resample("D").ffill()[:incidence.shape[0]]
    pop_by_age_group_year = pop_by_age_group_month.resample("YE").ffill()[:10]
    incidence = incidence.multiply(pop_by_age_group_day[AGE_GROUP_NAMES].values, axis=0)

    # are any incidence values nan?
    incidence = incidence.fillna(0)
    # take only first 9*365 values of incidence
    obs_per_season = calculate_observations_per_season(incidence, age_groups=True)
    if season == "pre_median":
        season_obs = obs_per_season.sum(axis=1)[:5]
        # index of median season
        season_idx = np.argsort(season_obs)[2]
    elif season == "rebound":
        season_idx = calculate_rebound_season(obs_per_season.sum(axis=1))
        if season_idx is None:
            return
    else:
        season_idx = season
    obs_per_season = obs_per_season / pop_by_age_group_year[AGE_GROUP_NAMES].values * factor
    # plot simple line of incidence in each age group in the season_idx season
    ax.plot(np.arange(7), obs_per_season[season_idx,:], color=color, label=label)
    # label with age group names
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(AGE_GROUP_NAMES, fontsize=6)

## plot cumulative cases in each age group in each season
def season_sizes(cases):
    seasons = [pd.to_datetime('20'+str(x)+'-10-01') for x in range(15,26)]
    season_cumulative = np.zeros((len(seasons)-1,7))
    for seas in range(len(seasons)-1):
        season_cumulative[seas,:] = cases[(cases.index>seasons[seas]) & (cases.index<seasons[seas+1])].sum(axis=0)
    return(pd.DataFrame(season_cumulative))

def season_plot(ax,pathogen,incidence=True,relative=False):
    cases = pd.read_csv('Data/Processed/KPSC_ARI_'+pathogen+'_cases_age_daily.csv',index_col=0)
    cases.index = pd.to_datetime(cases.index)
    season_cumulative = season_sizes(cases)
    print(pathogen,np.sum(season_cumulative,axis=1))
    if relative and not incidence:
        season_relative = season_cumulative.div(season_cumulative.sum(axis=1),axis=0)
    if relative and incidence:
        incidence_data = pd.read_csv('Data/Processed/KPSC_ARI_'+pathogen+'_incidence_age_daily.csv',index_col=0)
        incidence_data.index = pd.to_datetime(incidence_data.index)
        season_incidence = season_sizes(incidence_data)
        season_relative = season_incidence.div(season_incidence.sum(axis=1),axis=0)
    if incidence or relative:
        for i in range(season_cumulative.shape[0]):
            if np.sum(season_cumulative.iloc[i,:])<=50:
                season_relative.iloc[i,:] = 0
        season_relative.plot(ax=ax,kind="bar",stacked=True,color=hsv_colors,legend=False)
    else:
        season_cumulative.plot(ax=ax,kind="bar",stacked=True,color=hsv_colors,legend=False)

if __name__ == "__main__":
    plt.rcParams.update({'font.size':8})
    # text type is palatino
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    from Parameters.census_population import AGE_GROUPS, AGE_GROUP_NAMES

    fig = plt.figure(layout="constrained", figsize=(7,4))

    pathogens = ["Metapneumovirus","Parainfluenza3","Adenovirus","RSV","InfluenzaA","InfluenzaB",]

    # subfigs = fig.subfigures(1, 2, wspace=0.05, width_ratios=[7, 3])
    # axA = subfigs[1].subplots(len(pathogens), 1, sharex = True)
    # axB = subfigs[0].subplots((len(pathogens) + 1)//2, 2, sharex = True)

    
    # # figA, axA = plt.subplots(6, 1, figsize=(2.5,4), sharex = True)
    # for pi,pathogen in enumerate(pathogens):
    #     print(pathogen)
    #     age_group_incidence_plot(axA[pi],pathogen,color="k",season="pre_median", AGE_GROUPS=AGE_GROUPS, AGE_GROUP_NAMES=AGE_GROUP_NAMES, label="Pre-COVID-19")
    #     age_group_incidence_plot(axA[pi],pathogen,color="silver",season="rebound", AGE_GROUPS=AGE_GROUPS, AGE_GROUP_NAMES=AGE_GROUP_NAMES, label="Re-emergence")
    #     axA[pi].set_title(nice_names.get(pathogen, pathogen))
    # # set singe x label for all subplots
    # axA[-1].set_xlabel("Age group")
    # axA[0].legend(loc="upper right", fontsize=6)
    # # set single y label for all subplots
    # subfigs[1].text(-0.05, 0.5, 'Incidence per 100k members', va='center', rotation='vertical')
    # # plt.tight_layout()
    # # plt.savefig("Figures/KPSC_age_group_incidence_pre_median_rebound.png",dpi=300)

    figB, axB = plt.subplots(3, 2, figsize=(6.5,4), sharex = True)
    data_color = "#648FFF"
    aggregation = "W-MON"
    agg_factor = {"D":1, "W-MON":7, "MS":30.44}[aggregation]
    factor = 100000
    if factor >= 1000000:
        factor_label = f"{factor // 1000000}M"
    elif factor >= 1000:
        factor_label = f"{factor // 1000}k"
    else:
        factor_label = str(factor)
    for pi, pathogen in enumerate(pathogens):
        kpsc_proportion_positive_incidence_plot(
            axB[pi//2, pi%2], pathogen=pathogen, title=nice_names.get(pathogen, pathogen),
            color=data_color, aggregation=aggregation, factor=factor,
            annotations=False, label="Data", hosp=False)
    # suppress all y labels and replace with single label on left
    for i in range(len(pathogens)//2):
        for j in range(2):
            axB[i,j].set_ylabel("")
    axB[len(pathogens)//4,0].set_ylabel(f"Incidence per {factor_label} members")
    # only icnlude every other year label
    for axB_i in axB.flatten():
        axB_i.set_xticks(pd.date_range(start='2016-01-01',end='2025-01-01',freq='2YS'))
        axB_i.set_xticklabels([str(year) for year in range(2016,2026,2)])
        axB_i.set_xticks(pd.date_range(start='2016-01-01',end='2025-01-01',freq='YS'), minor=True)

    # plt.tight_layout()
    # plt.savefig("Figures/KPSC_proportion_positive_incidence_weekly_annotated.png",dpi=300)

    # now include plots of simulations on top of data
    
    lockdown = "Exponential"
    option1 = "NA"
    option2 = "flexagep01"
    seeds = [2603172,]*6
    ## Initial conditions
    from Parameters.census_population import CENSUS_AGE_POP
    STATE0 = jnp.zeros((2*N_S+1,NAG))
    STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
    STATE0 = STATE0.at[1,:].set(1)
    # # flatten initial state and add maternal immunity compartment
    STATE0 = STATE0.flatten()
    STATE0 = jnp.concatenate((jnp.array([0]), STATE0))
    from Parameters.times_and_contacts import PERIOD
    POINTS = jnp.array(date_to_t(PERIOD))
    T_LOCKDOWN = date_to_t(pd.to_datetime("2020-03-20"))
    for pi, pathogen in enumerate(pathogens):
        print(pathogen)
        print(f"Plotting simulations for {pathogen}...")
        try:
            params, _, _, _, p_time_to_obs = parameters_from_DE(pathogen, lockdown, option1, option2, seeds[pi])
        except Exception as e:
            print(f"Pathogen {pathogen} not found")
            continue
        lockdown_incidence_plot(axB[pi//2,pi%2], STATE0, params, POINTS, T_LOCKDOWN,
                                p_time_to_obs=p_time_to_obs,
                                color="#DC267F", factor=factor*agg_factor, label="Simulation")
    axB[0,1].legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig("Figures/ReportOverallIncidenceWeeklyExponential2604172.png",dpi=300)

    # subfigs[0].suptitle("A", x=0.01, fontweight='bold')
    # subfigs[1].suptitle("B", x=0.01, fontweight='bold')

    # plt.savefig("Figures/Figure1_thirdmedian.png",dpi=300)