## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
from matplotlib import cm as colormaps
from math import comb
import corner
import pickle

from utils import *

N_C=2
from SISn_ODEs import single_pathogen_deltas as deltas_SIS

##### General plotting parameters #####
hsv_colors = colormaps.hsv(-0.02+np.arange(7)/7)
hsv_colors[3] = colormaps.hsv((3/7)+0.04)

##### Simple line plots #####
def lockdown_incidence_plot(ax,state0,params,OBS_AGE,period,points,T_LOCKDOWN,LOCKDOWN_DURATION,result=None,label='Observed cases',color='#648FFF',linewidth=1,alpha=1,by_age=False,AGE_GROUP_NAMES=None,relative=False,deltas=deltas_SIS,obs=None,times=None,start_t=date_to_t(pd.to_datetime('2015-10-01')),end_t=date_to_t(pd.to_datetime('2023-09-30')),factor=1,p_time_to_obs=[1]):
    if params is not None:
        NAG, N_S, AGING_RATE, births, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, arrivals, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact = params.values()
    if result is None:
        result = sp.integrate.solve_ivp(deltas, [0,period], state0, method='RK45', t_eval=points,args=(params,))
    if times is None:
        times = result.t
    dates = [t_to_date(t) for t in times]
    start_index = np.argmin(times<=start_t)
    end_index = np.argmin(times<=end_t)
    if by_age:
        pop_size_by_age = np.array([np.sum(result.y[range(i_age,(N_C*N_S+1)*NAG,NAG),:],axis=0) for i_age in range(NAG)]).T
        if obs is None:
            obs = factor*observations(result,params,OBS_AGE,incidence=False)
        else:
            obs = factor*obs
        # using roll to account for detection delays, note that frist len(p_time_to_obs) days should not be used due to wrap-around
        obs = np.sum([np.roll(obs,i,axis=0)*p_time_to_obs[i] for i in range(len(p_time_to_obs))],axis=0)
        for i_age in range(NAG):
            ax.plot(dates[start_index:end_index],obs[start_index:end_index,i_age]/pop_size_by_age[start_index:end_index,i_age], label=AGE_GROUP_NAMES[i_age], color=hsv_colors[i_age],linewidth=linewidth,alpha=alpha)
        mx = 1.1*np.max(np.max(obs/pop_size_by_age,axis=1)[start_index:end_index])
    else:
        if obs is None:
            obs = factor*observations(result,params,OBS_AGE,incidence=True)
        else:
            obs = factor*obs
        obs = np.sum([np.roll(obs,i)*p_time_to_obs[i] for i in range(len(p_time_to_obs))],axis=0)
        if relative:
            pre_mx = np.max(obs[start_index:np.argmin(times<=T_LOCKDOWN)])
            ax.plot(dates[start_index:end_index], obs[start_index:end_index]/pre_mx[start_index:end_index], label=label,color=color,linewidth=linewidth,alpha=alpha)
            mx = 1.1*np.max(obs[start_index:end_index])/pre_mx
        else:
            ax.plot(dates[start_index:end_index], obs[start_index:end_index], label=label,color=color,linewidth=linewidth,alpha=alpha)
            mx = 1.1*np.max(obs[start_index:end_index])
    return(mx)

def lockdown_incidence_format(ax,T_LOCKDOWN,LOCKDOWN_DURATION,mx,year_window=5,year_skip=1,title='Incidence of disease with 1-year lockdown'):
    # ax.set_xlim(T_LOCKDOWN-year_window*365,T_LOCKDOWN+LOCKDOWN_DURATION+year_window*365)
    # ax.set_ylim(0,mx)
    ax.set_ylabel('Observed incidence')
    # ax.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],0,mx,color='gray',alpha=0.2)
    # ax.set_xticks(np.arange(T_LOCKDOWN-year_window*365,T_LOCKDOWN+LOCKDOWN_DURATION+year_window*365+365,365*year_skip),[str(int(x)-year_window-1) for x in np.arange(0,year_window*2+2,year_skip)])
    ax.set_xlabel('Time (years)')
    ax.set_title(title)

def lockdown_susceptibility_plot(ax,state0,params,period,points,T_LOCKDOWN,result=None,label='Susceptible_population',color='#648FFF',relative=True,by_age=False,AGE_GROUP_NAMES=None,style='-',delta=deltas_SIS):
    NAG, N_S, AGING_RATE, births, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, T_VAX, arrivals, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact = params.values()
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
            pre_mx_sus = np.mean(total_sus[np.argmax(result.t>T_LOCKDOWN-5*365):np.argmax(result.t>T_LOCKDOWN)])
            rel_sus = total_sus/pre_mx_sus
            ax.plot(result.t,rel_sus, label=label,color=color,linestyle=style)
        else:
            ax.plot(result.t,total_sus, label=label,color=color,linestyle=style)

def lockdown_susceptibility_format(ax,T_LOCKDOWN,LOCKDOWN_DURATION,ymin=0.875,ymax=1.1,year_window=5):
    ax.set_xlim(T_LOCKDOWN-year_window*365,T_LOCKDOWN+LOCKDOWN_DURATION+year_window*365)
    ax.set_ylabel('Relative susceptibility')
    ax.set_xticks(np.arange(T_LOCKDOWN-year_window*365,T_LOCKDOWN+LOCKDOWN_DURATION+year_window*365,365),[str(int(x)-year_window) for x in np.arange(0,2*year_window+1,1)])
    ax.set_xlabel('Time (years)')
    ax.set_title('Population susceptibility with 1-year lockdown')
    yin, yax = ax.get_ylim()
    if not (ymax is None or ymin is None):
        ax.set_ylim(ymin,ymax)
    #     ax.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],ymin,ymax,color='gray',alpha=0.2)
    # else:
    #     ax.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],yin,yax,color='gray',alpha=0.2)

def age_infect_plot(ax,state0,params,AGE_GROUP_NAMES,period,points,T_LOCKDOWN,LOCKDOWN_DURATION,result=None,delts=deltas_SIS):
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
    t_lockdown="2007-01-01",LOCKDOWN_DURATION=365):
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
                    color=color_values[n_j],alpha=1)
                    mx = max(mx,mxs)
            else:
                mxs = lockdown_incidence_plot(axes[i,0],None,None,None,None,None,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,obs=obs,relative=relative,
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
"Influenza A": ["INFLUENZA A","INFLUENZA A H1N1 2009","INFLUENZA A VIRUS","INFLUENZA A VIRUS SUBTYPE H1","INFLUENZA A VIRUS SUBTYPE/HEMAGGLUTININ H3","INFLUENZA VIRUS A","INFLUENZA VIRUS A+B"],
"Influenza A H1": ["INFLUENZA A H1N1 2009","INFLUENZA A VIRUS SUBTYPE H1"],
"Influenza A H3": ["INFLUENZA A VIRUS SUBTYPE/HEMAGGLUTININ H3"],
"Influenza B": ["INFLUENZA B","INFLUENZA VIRUS B","INFLUENZA VIRUS A+B"],
"Influenza": ["INFLUENZA A","INFLUENZA A H1N1 2009","INFLUENZA A VIRUS","INFLUENZA A VIRUS SUBTYPE H1","INFLUENZA A VIRUS SUBTYPE/HEMAGGLUTININ H3","INFLUENZA VIRUS A","INFLUENZA VIRUS A+B","INFLUENZA B","INFLUENZA VIRUS B"],
"Metapneumovirus": ["HUMAN METAPNEUMOVIRUS VIRUS",],
"Adenovirus": ["ADENOVIRUS",],
"Parainfuenza": ["PARAINFLUENZA VIRUS 1","PARAINFLUENZA VIRUS 2","PARAINFLUENZA VIRUS 3","PARAINFLUENZA VIRUS 4"],
"Parainfluenza 3": ["PARAINFLUENZA VIRUS 3"]}
def kpsc_positive_test_plot(ax,hospitalizations=True,pathogen="RSV",AGE_GROUPS=None,AGE_GROUP_NAMES=None,incidence=False, color=hsv_colors, title=None, legend=True, aggregation=None, save_data=False):
    print(pathogen)
    respiratory_codes = pd.read_csv('Data/Processed/respiratory_codes.csv')
    if incidence:
        # load age population data
        age_by_year = pd.read_csv("Data/Processed/KPSC_population_by_age.csv")
        # # add first two columns - remove this once year added
        # age_by_year["Infants"] = age_by_year["Infants"] + age_by_year["Newborns"]
        # age_by_year.drop(columns=["Newborns"],inplace=True)
        if AGE_GROUPS is not None:
            age_by_year.columns = AGE_GROUP_NAMES
        age_by_year["Year"] = np.arange(2015,2023)
        age_by_year = age_by_year.set_index("Year")
        # repeat last row for 2023
        age_by_year.loc[2023] = age_by_year.loc[2022]
    
    if hospitalizations:
        positive_tests = pd.read_csv('Data/Processed/KPSC_positive_matched_hospitalizations.csv')
    else:
        positive_tests = pd.read_csv('Data/Processed/KPSC_positive_matched_all_clinical.csv')
    names = pathogen_names[pathogen]
    cases = positive_tests[positive_tests['pathogen'].isin(names) & positive_tests['CODE'].isin(respiratory_codes)]
    cases = cases.drop_duplicates(subset=cases.columns.difference(['CODE','dxgroup']))
    if hospitalizations:
        cases["Date"] = pd.to_datetime(cases["Hospitalization date"])
    else:
        cases["Date"] = pd.to_datetime(cases["Clinical date"])
    if aggregation is not None:
        cases["Year"] = cases["Date"].dt.year
        if aggregation == "Month":
            cases["Month"] = cases["Date"].dt.month
        elif aggregation == "Week":
            cases["Week"] = cases["Date"].dt.isocalendar().week
    if AGE_GROUPS is not None:
        for i in range(len(AGE_GROUPS)):
            cases.loc[cases["age_in_mo"].isin(AGE_GROUPS[i]),"age_group"] = AGE_GROUP_NAMES[i]
        if aggregation is None:
            cases = cases.groupby(["Date","age_group"]).size().reset_index(name='Count')
        else:
            cases = cases.groupby(["Year",aggregation,"age_group"]).size().reset_index(name='Count')
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

    for date in pd.date_range(start='2015-10-01',end='2023-10-01',freq=frequency):
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
        if aggregation is None:
            if not (cases["Date"]==date).any():
                cases = pd.concat([cases,pd.DataFrame({"Date":[date],"Count":[0]})])
        else:
            if not ((cases["Year"]==year) & (cases[aggregation]==agg)).any():
                cases = pd.concat([cases,pd.DataFrame({"Year":[year],aggregation:[agg],"Count":[0]})])
    if aggregation is not None:
        if aggregation == "Month":
            cases["Date"] = pd.to_datetime(cases["Year"].astype(str) + '-' + cases["Month"].astype(str) + '-01')
        elif aggregation == "Week":
            cases["Date"] = cases["Year"].astype(str) + '-' + cases["Week"].astype(str)
            cases["Date"] = pd.to_datetime(cases["Date"].add('-1').astype(str),format='%Y-%W-%w')
    cases = cases.sort_values(by="Date")
    cases = cases.set_index("Date")

    if aggregation is not None:
        cases = cases.drop(columns=["Year",aggregation])
    if AGE_GROUPS is not None:
        cases = cases.pivot(columns="age_group",values="Count")
        cases = cases[AGE_GROUP_NAMES]
        if incidence:
            cases = cases.div(age_by_year.loc[cases.index.year].values)
    elif incidence:
        cases["Count"] = cases["Count"]/np.sum(age_by_year.loc[cases.index.year].values,axis=1)

    if save_data:
        filename = f'Data/Processed/KPSC_{pathogen}_{["cases","incidence"][incidence]}_{["all","age"][AGE_GROUPS is not None]}_{["daily","weekly","monthly"][["D","W-MON","MS"].index(frequency)]}.csv'
        cases.to_csv(filename)

    if incidence:
        cases *= 10000
    if title is None:
        if incidence:
            title = f"{pathogen} incidence"
        else:
            title = f"{pathogen} positive cases"
    if AGE_GROUPS is not None:
        cases.plot(ax=ax,legend=legend,color=color,label=AGE_GROUP_NAMES,title=f"{pathogen} positive tests")
    else:
        cases.plot(ax=ax,legend=False,color=color,title=f"{pathogen} positive tests")
    if incidence:
        ax.set_ylabel("Incidence per 10k")
    else:
        ax.set_ylabel("Cases")
    if legend:
        # set legend title to "Age groups"
        ax.legend(title="Age groups")
    ax.set_xlabel("Date")

## plot cumulative cases in each age group in each season
def season_sizes(cases):
    seasons = [pd.to_datetime('20'+str(x)+'-10-01') for x in range(15,24)]
    season_cumulative = np.zeros((len(seasons)-1,7))
    season_relative = np.zeros((len(seasons)-1,7))
    for seas in range(len(seasons)-1):
        season_cumulative[seas,:] = cases[(cases.index>seasons[seas]) & (cases.index<seasons[seas+1])].sum(axis=0)
    return(pd.DataFrame(season_cumulative))

def season_plot(ax,pathogen,incidence=False,relative=False):
    cases = pd.read_csv('Data/Processed/KPSC_'+pathogen+'_cases_age_daily.csv',index_col=0)
    cases.index = pd.to_datetime(cases.index)
    season_cumulative = season_sizes(cases)
    print(pathogen,np.sum(season_cumulative,axis=1))
    if relative and not incidence:
        season_relative = season_cumulative.div(season_cumulative.sum(axis=1),axis=0)
    if relative and incidence:
        incidence_data = pd.read_csv('Data/Processed/KPSC_'+pathogen+'_incidence_age_daily.csv',index_col=0)
        incidence_data.index = pd.to_datetime(incidence_data.index)
        season_incidence = season_sizes(incidence_data)
        season_relative = season_incidence.div(season_incidence.sum(axis=1),axis=0)
    if incidence or relative:
        for i in range(season_cumulative.shape[0]):
            if np.sum(season_cumulative.iloc[i,:])<=37:
                season_relative.iloc[i,:] = 0
        season_relative.plot(ax=ax,kind="bar",stacked=True,color=hsv_colors,legend=False)
    else:
        season_cumulative.plot(ax=ax,kind="bar",stacked=True,color=hsv_colors,legend=False)
    