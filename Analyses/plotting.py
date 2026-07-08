## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

import os
import re
import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
from matplotlib import cm as colormaps, ticker
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from math import comb
import corner
import pickle
import colorsys
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController
import time

from utils import date_to_t, t_to_date, calculate_population_size, susceptibility, infections_by_age, observations, load_optimization_results, x_to_params, sum_age_to, pathogen_parameters, load_mcmc_chain, load_random_mcmc_result, parameters_names_bounds, constrained_immunity, calculate_R0_from_values

N_C = 2
N_S = 3
# NAG = 7
from JAX_ODEs import deltas

##### General plotting parameters #####
hsv_colors = colormaps.hsv(-0.02+np.arange(7)/7)
hsv_colors[3] = colormaps.hsv((3/7)+0.095)
# '#ff0000', '#ffb700', '#6cff00', '#00ffc0', '#00bbff', '#1900ff', '#f300ff'

def lockdown_incidence_plot(
    ax,state0,params,points,T_LOCKDOWN,solution=None,label='Observed cases',color='#648FFF',linewidth=1,alpha=1,
    by_age=False,AGE_GROUP_NAMES=None,select_age_group=None,relative=False,deltas=deltas,obs=None,times=None,
    start_t=date_to_t('2015-10-01'),end_t=date_to_t('2025-05-01'),factor=1,p_time_to_obs=[1],
    NAG=7, AGE_GROUPS=None, max_month=None,
    test_data=None,daily_hospitalization_rates=None,aggregation=None, uncertainty="confidence"
):
    if solution is None:
        term = ODETerm(deltas)
        solver = Dopri5()
        saveat = SaveAt(ts=points)
        step_controller = PIDController(rtol=1e-5, atol=1e-5)
        params_sim = params + (NAG,)
        solution = diffeqsolve(
            term,
            solver,
            t0=0,
            t1=int(points[-1]),
            dt0=None,
            stepsize_controller=step_controller,
            saveat=saveat,
            y0=state0.flatten(),
            args=params_sim,
            max_steps=None,
        )

    if times is None:
        times = solution.ts

    values = solution.ys.T
    dates = [t_to_date(t) for t in times]
    start_index = np.argmin(times <= start_t)
    end_index = np.argmin(times <= end_t)
    if end_index <= start_index:
        end_index = len(times)
    
    trajectory = np.diff(values[-NAG:, :], axis=1).T
    expected_obs = np.sum([np.roll(trajectory, i, axis=0) * p_time_to_obs[i] for i in range(len(p_time_to_obs))], axis=0)
    if (AGE_GROUPS is not None) and (max_month is not None):
        expected_obs = sum_age_to(expected_obs, max_month, AGE_GROUPS)

    # Resolve optional single-age-group selection
    selected_age_idx = None
    if select_age_group is not None:
        by_age = True
        if isinstance(select_age_group, str):
            if AGE_GROUP_NAMES is None:
                raise ValueError("AGE_GROUP_NAMES must be provided when select_age_group is a string.")
            selected_age_idx = AGE_GROUP_NAMES.index(select_age_group)
        else:
            selected_age_idx = int(select_age_group)
        if selected_age_idx < 0 or selected_age_idx >= NAG:
            raise ValueError(f"select_age_group index must be between 0 and {NAG-1}.")

    if by_age:
        hsv_colors = colormaps.hsv(-0.02+np.arange(NAG)/NAG)
        hsv_colors[3] = colormaps.hsv((3/NAG)+0.28/NAG)
        pop_size_by_age = calculate_population_size(values, NAG=NAG)[1:]
        if (AGE_GROUPS is not None) and (max_month is not None):
            pop_size_by_age = sum_age_to(pop_size_by_age, max_month, AGE_GROUPS)
        obs = factor * expected_obs

        if relative:
            divisor = np.max(obs[start_index:np.argmin(times <= T_LOCKDOWN)], axis=0)
        else:
            divisor = pop_size_by_age[start_index:end_index].T

        if selected_age_idx is None:
            for i_age in range(NAG):
                if uncertainty != "draw":
                    ax.plot(
                        dates[(start_index + 1):end_index],
                        obs[start_index:end_index, i_age] / divisor[i_age],
                        label=AGE_GROUP_NAMES[i_age] if AGE_GROUP_NAMES is not None else f"Age {i_age}",
                        color=hsv_colors[i_age],
                        linewidth=linewidth,
                        alpha=alpha,
                    )
            mx = 1.1 * np.max(np.max(obs / pop_size_by_age, axis=1)[start_index:end_index])
        else:
            series = obs[start_index:end_index, selected_age_idx] / divisor[selected_age_idx]
            age_label = (
                AGE_GROUP_NAMES[selected_age_idx]
                if AGE_GROUP_NAMES is not None
                else f"Age {selected_age_idx}"
            )
            if uncertainty != "draw":
                ax.plot(
                    dates[(start_index + 1):end_index],
                    series,
                    label=label if label != 'Observed cases' else age_label,
                    color=color if color is not None else hsv_colors[selected_age_idx],
                    linewidth=linewidth,
                    alpha=alpha,
                )
            mx = 1.1 * np.max(series)
    else:
        obs = factor * np.sum(expected_obs, axis=1) / np.sum(values[:-NAG, :], axis=0)[1:]
        if relative:
            pre_mx = np.max(obs[start_index:np.argmin(times <= T_LOCKDOWN)])
            if uncertainty != "draw":
                ax.plot(
                    dates[(start_index + 1):end_index],
                    obs[start_index:end_index] / pre_mx,
                    label=label,
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                )
            mx = 1.1 * np.max(obs[start_index:end_index]) / pre_mx
        else:
            if uncertainty != "draw":
                ax.plot(
                    dates[(start_index+1):end_index],
                    obs[start_index:end_index],
                    label=label,
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                )
            mx = 1.1 * np.max(np.array(obs[start_index:end_index]))

    # Plot 95% CI or drawn trajectory from binomial distribution if test_data provided
    if test_data is not None:
        if by_age and selected_age_idx is not None:
            pop_size_by_age = calculate_population_size(values, NAG=NAG)
            overall_hosp = daily_hospitalization_rates[:, selected_age_idx] * pop_size_by_age[start_index+1:end_index, selected_age_idx]
            expected_obs_col = expected_obs[start_index:end_index, selected_age_idx]
            with np.errstate(divide='ignore', invalid='ignore'):
                expected_prop = expected_obs_col / overall_hosp
            n_tests = test_data[1:, selected_age_idx, 0]
            # Aggregate if needed
            if aggregation is not None:
                dates_agg = dates[(start_index + 1):end_index]
                agg_freq_map = {'D': 'D', 'W': 'W', 'M': 'MS'}
                agg_freq = agg_freq_map.get(aggregation[0], 'D')
                
                pop_size_col = pop_size_by_age[start_index+1:end_index, selected_age_idx]
                df_agg = pd.DataFrame({
                    'date': dates_agg,
                    'expected_obs': expected_obs_col,
                    'overall_hosp': overall_hosp,
                    'expected_prop': expected_prop,
                    'pop_size': pop_size_col,
                    'n_tests': n_tests
                })
                df_agg.loc[:, 'date'] = pd.to_datetime(df_agg['date'])
                df_agg = df_agg.set_index('date')
                
                expected_obs_col = df_agg['expected_obs'].resample(agg_freq).sum().values
                overall_hosp = df_agg['overall_hosp'].resample(agg_freq).sum().values
                pop_size_col = df_agg['pop_size'].resample(agg_freq).first().values
                with np.errstate(divide='ignore', invalid='ignore'):
                    expected_prop = expected_obs_col / overall_hosp
                dates_agg = df_agg['expected_obs'].resample(agg_freq).sum().index.to_list()
                n_tests = df_agg['n_tests'].resample(agg_freq).sum().values
            else:
                dates_agg = dates[(start_index + 1):end_index]
                pop_size_col = pop_size_by_age[start_index:end_index, selected_age_idx]
                overall_hosp = overall_hosp[start_index:end_index]
            
            ci_lower = np.zeros_like(expected_prop)
            ci_upper = np.zeros_like(expected_prop)
            drawn_path = np.zeros_like(expected_prop)
            
            for t_idx in range(len(expected_prop)):
                if n_tests[t_idx] > 0:
                    p = np.clip(expected_prop[t_idx], 0, 1)
                    if uncertainty == "confidence":
                        ci_lower[t_idx] = overall_hosp[t_idx] * sp.stats.binom.ppf(0.025, n_tests[t_idx], p) / (n_tests[t_idx] * pop_size_col[t_idx])
                        ci_upper[t_idx] = overall_hosp[t_idx] * sp.stats.binom.ppf(0.975, n_tests[t_idx], p) / (n_tests[t_idx] * pop_size_col[t_idx])
                    elif uncertainty == "draw":
                        draw_k = np.random.binomial(int(n_tests[t_idx]), p)
                        drawn_path[t_idx] = overall_hosp[t_idx] * draw_k / (n_tests[t_idx] * pop_size_col[t_idx])
            
            agg_factor_map = {'D': 1, 'W': 7, 'M': 30.44}
            agg_factor = agg_factor_map.get(aggregation[0], 1) if aggregation is not None else 1
            
            if uncertainty == "confidence":
                ax.fill_between(
                    dates_agg,
                    factor * ci_lower / agg_factor,
                    factor * ci_upper / agg_factor,
                    alpha=0.2,
                    color=color,
                )
            elif uncertainty == "draw":
                age_label = (
                    AGE_GROUP_NAMES[selected_age_idx]
                    if AGE_GROUP_NAMES is not None
                    else f"Age {selected_age_idx}"
                )
                plot_label = label if label != 'Observed cases' else age_label
                plot_color = color if color is not None else hsv_colors[selected_age_idx]
                
                ax.plot(
                    dates_agg,
                    factor * drawn_path / agg_factor,
                    label=plot_label,
                    color=plot_color,
                    linewidth=linewidth,
                    alpha=alpha
                )

        elif not by_age:
            pop_size_by_age = calculate_population_size(values, NAG=NAG)
            overall_hosp = np.sum((daily_hospitalization_rates * pop_size_by_age[start_index+1:end_index, :]), axis=1)
            expected_obs_agg = np.sum(expected_obs[start_index:end_index], axis=1)
            expected_prop = expected_obs_agg / overall_hosp
            n_tests = np.sum(test_data[1:, :, 0], axis=1) if test_data.ndim == 3 else test_data[:, 0]
            # Aggregate if needed
            if aggregation is not None:
                dates_agg = dates[(start_index + 1):end_index]
                agg_freq_map = {'D': 'D', 'W': 'W', 'M': 'MS'}
                agg_freq = agg_freq_map.get(aggregation[0], 'D')
                
                pop_size_agg = np.sum(pop_size_by_age[start_index+1:end_index, :], axis=1)
                df_agg = pd.DataFrame({
                    'date': dates_agg,
                    'expected_obs': expected_obs_agg,
                    'overall_hosp': overall_hosp,
                    'expected_prop': expected_prop,
                    'pop_size': pop_size_agg,
                    'n_tests': n_tests
                })
                df_agg.loc[:, 'date'] = pd.to_datetime(df_agg['date'])
                df_agg = df_agg.set_index('date')
                
                expected_obs_agg = df_agg['expected_obs'].resample(agg_freq).sum().values
                overall_hosp = df_agg['overall_hosp'].resample(agg_freq).sum().values
                pop_size_agg = df_agg['pop_size'].resample(agg_freq).first().values
                expected_prop = expected_obs_agg / overall_hosp
                n_tests = df_agg['n_tests'].resample(agg_freq).sum().values
                dates_agg = df_agg['expected_obs'].resample(agg_freq).sum().index.to_list()
            else:
                dates_agg = dates[(start_index + 1):end_index]
                pop_size_agg = np.sum(pop_size_by_age[start_index:end_index, :], axis=1)
            
            ci_lower = np.zeros_like(expected_prop)
            ci_upper = np.zeros_like(expected_prop)
            drawn_path = np.zeros_like(expected_prop)
            
            for t_idx in range(len(expected_prop)):
                if n_tests[t_idx] > 0:
                    p = np.clip(expected_prop[t_idx], 0, 1)
                    if uncertainty == "confidence":
                        ci_lower[t_idx] = overall_hosp[t_idx] * sp.stats.binom.ppf(0.025, n_tests[t_idx], p) / (n_tests[t_idx] * pop_size_agg[t_idx])
                        ci_upper[t_idx] = overall_hosp[t_idx] * sp.stats.binom.ppf(0.975, n_tests[t_idx], p) / (n_tests[t_idx] * pop_size_agg[t_idx])
                    elif uncertainty == "draw":
                        draw_k = np.random.binomial(int(n_tests[t_idx]), p)
                        drawn_path[t_idx] = overall_hosp[t_idx] * draw_k / (n_tests[t_idx] * pop_size_agg[t_idx])
            
            agg_factor_map = {'D': 1, 'W': 7, 'M': 30.44}
            agg_factor = agg_factor_map.get(aggregation[0], 1) if aggregation is not None else 1
            
            if uncertainty == "confidence":
                ax.fill_between(
                    dates_agg,
                    factor * ci_lower / agg_factor,
                    factor * ci_upper / agg_factor,
                    alpha=0.2,
                    color=color,
                )
            elif uncertainty == "draw":
                ax.plot(
                    dates_agg,
                    factor * drawn_path / agg_factor,
                    label=label,
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha
                )

    return mx

def lockdown_incidence_format(ax,T_LOCKDOWN,LOCKDOWN_DURATION,mx,year_window=5,year_skip=1,title='Incidence of disease'):
    # ax.set_xlim(T_LOCKDOWN-year_window*365,T_LOCKDOWN+LOCKDOWN_DURATION+year_window*365)
    # ax.set_ylim(0,mx)
    ax.set_ylabel('Incidence per 10k')
    # ax.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],0,mx,color='gray',alpha=0.2)
    # ax.set_xticks(np.arange(T_LOCKDOWN-year_window*365,T_LOCKDOWN+LOCKDOWN_DURATION+year_window*365+365,365*year_skip),[str(int(x)-year_window-1) for x in np.arange(0,year_window*2+2,year_skip)])
    ax.set_xlabel('Time (years)')
    ax.set_title(title)

def prevalence_plot(ax,state0,params,points,obs_age=None,solution=None,label='Observed cases',color='#648FFF',linewidth=1,alpha=1,by_age=False,AGE_GROUP_NAMES=None,deltas=deltas,times=None,start_t=date_to_t('2015-10-01'),end_t=date_to_t('2025-05-01'), NAG=7):
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


def lockdown_susceptibility_plot(ax,state0,params,period,points,T_LOCKDOWN,solution=None,label='Susceptible_population',color='#648FFF',relative=True,proportion=False,by_age=False,AGE_GROUP_NAMES=None,style='-',delta=deltas,NAG=7,NAG_eff=7,AGE_GROUPS=None, max_month=None, linewidth=1, alpha=1):
    N_S = 3
    if solution is None:
        params = params + (NAG,)
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
    sus = susceptibility(solution,params,NAG=NAG)
    if (AGE_GROUPS is not None) and (max_month is not None):
        sus = sum_age_to(sus, max_month, AGE_GROUPS)
    if by_age:
        hsv_colors = colormaps.hsv(-0.02+np.arange(NAG_eff)/NAG_eff)
        hsv_colors[3] = colormaps.hsv((3/NAG_eff)+0.28/NAG_eff)
        if proportion:
            pop_by_age = calculate_population_size(values, NAG=NAG)
            if (AGE_GROUPS is not None) and (max_month is not None):
                pop_by_age = sum_age_to(pop_by_age, max_month, AGE_GROUPS)
            sus = sus/pop_by_age
        if relative:
            pre_mx_sus = np.max(sus[np.argmax(times>T_LOCKDOWN-5*365):np.argmax(times>T_LOCKDOWN)], axis=0)
            sus = sus/pre_mx_sus
        for i in range(NAG_eff):
            ax.plot(dates,sus[:,i], label=AGE_GROUP_NAMES[i], color=hsv_colors[i],linestyle=style, linewidth=linewidth, alpha=alpha)
    else:
        total_sus = np.sum(sus,axis=1)
        if relative:
            pre_mx_sus = np.mean(total_sus[np.argmax(times>T_LOCKDOWN-5*365):np.argmax(times>T_LOCKDOWN)])
            rel_sus = total_sus/pre_mx_sus
            ax.plot(dates,rel_sus, label=label,color=color,linestyle=style, linewidth=linewidth, alpha=alpha)
        else:
            ax.plot(dates,total_sus, label=label,color=color,linestyle=style, linewidth=linewidth, alpha=alpha)

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
import numpy as np
def kpsc_positive_test_plot(ax, pathogen="RSV", AGE_GROUPS=None, AGE_GROUP_NAMES=None, select_age_group=None, color=hsv_colors, legend=True, aggregation=None, factor=10000, label=None, linewidth=1, orig=False):
    pop_by_age_group_month = pd.read_csv('Data/Processed/KPSC_population_by_age_group_monthly.csv', index_col=0, parse_dates=['month_start'])
    pop_by_age_group_month = pop_by_age_group_month.reindex(columns=AGE_GROUP_NAMES)
    pop_by_age_group_daily = pop_by_age_group_month.resample('D').ffill()
    
    if orig:
        pathogen_data = pd.read_csv(f"Data/Processed/KPSC_ARI_{pathogen}_cases_age_daily.csv", parse_dates=["Date"])
        datename = "Date"
    else:
        pathogen_data = pd.read_csv(f'Data/Processed/KPSC_unsalvage_panel_positive_{pathogen}_matched_noncovid_ARI_hospitalizations.csv', parse_dates=["Hospitalization date"])
        datename = "Hospitalization date"
    pathogen_data.loc[:, datename] = pd.to_datetime(pathogen_data[datename])
    pathogen_data = pathogen_data.set_index(datename)
    
    # Calculate daily incidence per population
    daily_pop = pop_by_age_group_daily.reindex(pathogen_data.index).ffill()
    
    # Set aggregation frequency
    if aggregation is None:
        aggregation = 'D'
    agg_freq_map = {'D': 'D', 'W': 'W', 'M': 'MS'}
    agg_freq = agg_freq_map.get(aggregation[0], 'D')
    
    if AGE_GROUPS is not None:
        # Plot separately by age group
        mx = 0
        if select_age_group is not None:
            age_group = AGE_GROUP_NAMES[select_age_group]
            if label is None:
                label = age_group
            daily_incidence = (pathogen_data[age_group] / daily_pop[age_group]) * factor
            agg_incidence = daily_incidence.resample(agg_freq).sum()
            ax.plot(agg_incidence.index, agg_incidence.values, color=color, label=label, linewidth=linewidth)
            mx = max(mx, np.max(agg_incidence.values))
        else:
            for i, age_group in enumerate(AGE_GROUP_NAMES):
                if age_group in pathogen_data.columns:
                    daily_incidence = (pathogen_data[age_group] / daily_pop[age_group]) * factor
                    agg_incidence = daily_incidence.resample(agg_freq).sum()
                    ax.plot(agg_incidence.index, agg_incidence.values, color=color[i], label=age_group, linewidth=linewidth)
                    mx = max(mx, np.max(agg_incidence.values))
        if legend:
            ax.legend()
        return mx
    else:
        # Plot total across all age groups
        daily_incidence = (pathogen_data[AGE_GROUP_NAMES].sum(axis=1) / daily_pop[AGE_GROUP_NAMES].sum(axis=1)) * factor
        agg_incidence = daily_incidence.resample(agg_freq).sum()
        if label is None:
            label = "Total"
        ax.plot(agg_incidence.index, agg_incidence.values, color=color, label=label)
        return np.max(agg_incidence.values)

nice_names = {"RSV": "RSV", "InfluenzaA": "Influenza A", "InfluenzaB": "Influenza B", "Metapneumovirus": "Metapneumovirus", "Adenovirus": "Adenovirus", "Parainfluenza3": "Parainfluenza 3", "Rhinovirus": "Rhinovirus", "Pertussis": "Pertussis", "M.pneumoniae": "M. pneumoniae", "C.pneumoniae": "C. pneumoniae", "SARS-CoV-2": "SARS-CoV-2", "Enterovirus": "Enterovirus"}
short_names = {"RSV": "RSV", "InfluenzaA": "Flu A", "InfluenzaB": "Flu B", "Metapneumovirus": "hMPV", "Adenovirus": "AdV", "Parainfluenza": "PIV", "Parainfluenza3": "PIV3", "Rhinovirus": "RhV", "Pertussis": "Pertussis", "M.pneumoniae": "M. pneumo", "C.pneumoniae": "C. pneumo", "SARS-CoV-2": "COVID-19", "Enterovirus": "EV"}
from data_processing import calculate_proportion_positive_incidence
def kpsc_proportion_positive_incidence_plot(ax, pathogen="RSV", AGE_GROUPS=None, AGE_GROUP_NAMES=None, select_age_group=None, title=None, color=hsv_colors, linewidth=1, legend=True, aggregation="D", window_size=28, weighting_factor=0.5, label=None, factor=1000000, annotations=False, definition="50% median", pp_only=False, hosp=False, detrend=False, dedup=False, sac=True, mask=[3135,3288]):
    NAG = len(AGE_GROUP_NAMES) if AGE_GROUP_NAMES is not None else 7
    incidence = calculate_proportion_positive_incidence(pathogen, aggregation=aggregation, window_size=window_size, weighting_factor=weighting_factor, sum_age_groups=AGE_GROUPS is None, save_counts=False, pp_only=pp_only, hosp=hosp, NAG=NAG, detrend=detrend, dedup=dedup, sac=sac)
    incidence *= factor

    # mask interval as NA to create a visible gap in the plot
    if mask is not None and len(mask) == 2:
        gap_start = t_to_date(date_to_t('2015-07-04') + mask[0] + 89)
        gap_end = t_to_date(date_to_t('2015-07-04') + mask[1] + 89)
        incidence.loc[(incidence.index >= gap_start) & (incidence.index < gap_end), :] = np.nan

    if AGE_GROUPS is not None:
        if select_age_group is None:
            color = colormaps.hsv(-0.02+np.arange(NAG)/NAG)
            color[3] = colormaps.hsv((3/NAG)+0.28/NAG)
            for i in range(len(AGE_GROUP_NAMES)):
                ax.plot(incidence.index, incidence[AGE_GROUP_NAMES[i]], label=AGE_GROUP_NAMES[i], color=color[i], linewidth=linewidth)
        else:
            # print(AGE_GROUP_NAMES[select_age_group], "incidence:")
            # print(list(incidence[AGE_GROUP_NAMES[select_age_group]]))
            if label is None:
                label = AGE_GROUP_NAMES[select_age_group]
            ax.plot(incidence.index, incidence[AGE_GROUP_NAMES[select_age_group]], label=label, color=color, linewidth=linewidth)
    else:
        if label is None:
            label = nice_names.get(pathogen, pathogen)
        ax.plot(incidence.index, incidence["Total"], color=color, label=label, linewidth=linewidth)

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
        # if aggregation != "D":
        #     incidence = incidence.resample("D").interpolate()
        # # take only first 9*365 values of incidence
        # obs_per_season = calculate_observations_per_season(incidence)
        # print(obs_per_season)
        # peak_times = incidence["Total"].groupby(incidence.index.map(get_season_start)).idxmax()
        # last_peak_time = peak_times[peak_times.index < pd.to_datetime("2020-03-19")].max()
        # # find the rebound season - the first season after 2020-03-19 to read 50% of the total number of observations in the median season before 2020-03-19
        # rebound_season = calculate_rebound_season(obs_per_season, definition=definition)
        # print(rebound_season)
        # if rebound_season is None:
        #     # plot a line at 2020-03-19 and annotate that there was no rebound season by the end of the data, then quit
        #     ax.plot([last_peak_time, pd.to_datetime("2025-05-01")], [incidence.loc[last_peak_time, "Total"], incidence.loc[last_peak_time, "Total"]], color="black", linestyle="--")
        #     time_diff = pd.to_datetime("2025-05-01") - last_peak_time
        #     ax.annotate(f"{time_diff.days}+ days", xy=(last_peak_time + time_diff/2, incidence.loc[last_peak_time, "Total"]), xytext=(0,2), textcoords='offset points', ha='center', va="bottom", color="black",
        #             path_effects=[pe.Stroke(linewidth=1, foreground='white'), pe.Normal()])
        #     return
        # rebound_peak_time = peak_times.iloc[rebound_season]
        # print(rebound_peak_time)
        threshold = incidence[incidence.index < pd.to_datetime("2020-01-01")]["Total"].max() / 20
        dip_time = incidence[(incidence.index > pd.to_datetime("2020-01-01")) & (incidence["Total"] < threshold)].index.min()
        dip_idx = incidence.index.get_loc(dip_time)
        # Use the previous observed time point so this works for daily, weekly, or monthly aggregation.
        last_pre_time = incidence.index[max(dip_idx - 1, 0)]
        rebound_time = incidence[(incidence.index > last_pre_time) & (incidence["Total"] > threshold)].index.min()
        time_diff = rebound_time - last_pre_time
        if aggregation[0] == "D":
            time_amount = int(time_diff.days)
            time_label = f"{time_diff.days} days"
        elif aggregation[0] == "W":
            time_amount = int(np.floor(time_diff.days / 7))
            time_label = f"{time_amount} weeks"
        else:
            time_amount = int(np.floor(time_diff.days / 30.44))
            time_label = f"{time_amount} months"
        if color == 'black' or color == 'k':
            annotatecolor = 'red'
        else:
            annotatecolor = 'black'
        ax.plot([last_pre_time, rebound_time], [threshold, threshold], color=annotatecolor)
        if annotations=="simple":
            ax.annotate(time_amount, xy=(last_pre_time + time_diff/2, threshold), xytext=(0,2), textcoords='offset points', ha='center', va="bottom", color=annotatecolor)
        else:
            ax.annotate(time_label, xy=(last_pre_time + time_diff/2, threshold), xytext=(0,2), textcoords='offset points', ha='center', va="bottom", color=annotatecolor,
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

def calculate_observations_per_season(incidence, age_groups=False, aggregation="D"):
    if aggregation != "D":
        # resample to daily and interpolate
        scale = 1 if aggregation == "D" else (7 if aggregation == "W" else 30.44)
        incidence = incidence.resample("D").interpolate() / scale
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


def calculate_observations_per_season_jax(incidence, age_groups=False, pad_days=14, n_full_seasons=9, season_len=365, NAG=7):
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

    pad = jnp.zeros((pad_days, NAG), dtype=x.dtype)
    x = jnp.concatenate([pad, x], axis=0)

    full_days = n_full_seasons * season_len
    full = x[:full_days].reshape(n_full_seasons, season_len, NAG).sum(axis=1)   # (n_full_seasons, n_age)
    last = x[full_days:].sum(axis=0, keepdims=True)                                # (1, n_age)
    out = jnp.concatenate([full, last], axis=0)                                    # (n_full_seasons+1, n_age)

    if age_groups:
        return out
    return out.sum(axis=1)

def calculate_rebound_season(obs_per_season, definition = "40% median"):
    if isinstance(definition, str):
        definition = definition.strip().lower()

    if isinstance(definition, str) and definition.endswith("median") and "%" in definition:
        pct_str = definition.split("%", 1)[0].strip()
        try:
            pct = float(pct_str) / 100.0
        except ValueError as exc:
            raise ValueError(
                f"Invalid definition '{definition}'. Use format like '40% median'."
            ) from exc
        repr_pre_covid_obs = np.median(obs_per_season[:5]) * pct
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

def age_group_incidence_plot(ax,pathogen,color="k",season=0, AGE_GROUPS=None, AGE_GROUP_NAMES=None, factor=100000, label=None, NAG=7, sac=True):
    incidence = calculate_proportion_positive_incidence(pathogen, aggregation="D", window_size=1, weighting_factor=0, sum_age_groups=False, save_counts=False, hosp=True, dedup=True, sac=sac, NAG=NAG)
    incidence = incidence[AGE_GROUP_NAMES]
    pop_by_age_group_month = pd.read_csv('Data/Processed/KPSC_population_by_age_group_monthly'+["","_split"][NAG==8]+["","_sac"][sac]+'.csv', index_col=0, parse_dates=['month_start'])
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
    ax.bar(np.arange(NAG), obs_per_season[season_idx,:], color=color, label=label)
    # label with age group names
    ax.set_xticks(np.arange(NAG))
    ax.set_xticklabels(AGE_GROUP_NAMES, fontsize=6)

## plot cumulative cases in each age group in each season
def season_sizes(cases):
    seasons = [pd.to_datetime('20'+str(x)+'-10-01') for x in range(15,26)]
    season_cumulative = np.zeros((len(seasons)-1,7))
    for seas in range(len(seasons)-1):
        season_cumulative[seas,:] = cases[(cases.index>seasons[seas]) & (cases.index<seasons[seas+1])].sum(axis=0)
    return(pd.DataFrame(season_cumulative))

def season_plot(ax,pathogen,incidence=True,relative=False,filter=True):
    proportional_incidence = calculate_proportion_positive_incidence(pathogen, aggregation="YS-OCT", window_size=1, weighting_factor=0, sum_age_groups=False, save_counts=False, pp_only=False, hosp=True, NAG=NAG, detrend=False, dedup=True, sac=True)
    pop_by_age_group_month = pd.read_csv('Data/Processed/KPSC_population_by_age_group_monthly_sac.csv', index_col=0, parse_dates=['month_start'])
    pop_size = pop_by_age_group_month.resample("YS-OCT").ffill()[:proportional_incidence.shape[0]]
    pop_size.index = pd.to_datetime(pop_size.index)
    cases = proportional_incidence.multiply(pop_size.values, axis=0)
    if relative and not incidence:
        season_relative = cases.div(cases.sum(axis=1),axis=0)
    if relative and incidence:
        incidence_data = pd.read_csv('Data/Processed/KPSC_ARI_'+pathogen+'_incidence_age_daily.csv',index_col=0)
        incidence_data.index = pd.to_datetime(incidence_data.index)
        season_incidence = season_sizes(incidence_data)
        season_relative = season_incidence.div(season_incidence.sum(axis=1),axis=0)
    if incidence or relative:
        filtered_mask = np.ones(cases.shape[0], dtype=bool)
        for i in range(cases.shape[0]):
            if filter and np.sum(cases.iloc[i,:])<=np.max(np.sum(cases.iloc[:5,:].values, axis=1)*0.1):
                season_relative.iloc[i,:] = 0
                filtered_mask[i] = False
        season_relative.plot(ax=ax,kind="bar",stacked=True,color=hsv_colors,legend=False,width=1.0)
        # plot grey hash bar for filtered seasons
        for i in range(cases.shape[0]):
            if not filtered_mask[i]:
                ax.bar(i,1,color="grey",alpha=0.5,width=1.0)
    else:
        cases.plot(ax=ax,kind="bar",stacked=True,color=hsv_colors,legend=False,width=1.0)
    ax.set_xlabel("")
    ax.set_yticks([])
    ax.set_xlim(-0.5,cases.shape[0]-0.5)
    ax.set_ylim(0,1)

def calculate_infection_matrices_from_solution(solution, params, NAG=7, hospitalizations=False):
    # find total observed infections each season in each age group
    values = solution.ys.T
    shaped_values = values[1:, :].reshape((1+2*N_S, NAG, -1))
    infectious = shaped_values[1:2*N_S:2, :, :]
    susceptible = shaped_values[0:2*N_S:2, :, :]
    all_infectious = (params[7][:, :, None] * infectious).sum(axis=0)
    population_size = calculate_population_size(values, NAG=NAG)
    foi_matrix = params[4] * params[3][:, :, None] * all_infectious[None, :, :] / jnp.sum(population_size, axis=1)[None, None, :]
    infections_matrix = params[6][:, None, None, None] * foi_matrix[None, :, :, :] * susceptible[:, :, None, :]
    hospital_infections_matrix = infections_matrix * params[8][:, None, None, None] * params[9][None, :, None, None]
    age_hospital_matrix = hospital_infections_matrix.sum(axis=0)[:1553].mean(axis=-1)
    age_infections_matrix = infections_matrix.sum(axis=0)[:1553].mean(axis=-1)
    return jnp.stack([age_infections_matrix, age_hospital_matrix], axis=-1)

def get_infection_matrix(pathogen, seed, lockdown, option1, option2, NAG, census_age_pop, samples=None, hospitalizations=False, prefix=""):
    from fit_MCMC import run_simulation
    PERIOD = pd.date_range(start=pd.to_datetime('2015-10-01'), end=pd.to_datetime('2025-10-01'), freq='D')
    POINTS = np.array(date_to_t(PERIOD))
    ## Initial conditions
    STATE0 = jnp.zeros((2*N_S+1,NAG))
    STATE0 = STATE0.at[0,:].set(census_age_pop-1)
    STATE0 = STATE0.at[1,:].set(1)
    # # flatten initial state and add maternal immunity compartment
    STATE0 = STATE0.flatten()
    STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

    if samples is None:
        _, x, _ = load_optimization_results(prefix, pathogen, seed, lockdown, option1, option2)
        params = x_to_params(x, pathogen, lockdown, option1, option2, NAG=NAG)
        solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
        age_matrices = calculate_infection_matrices_from_solution(solution, params, NAG=NAG, hospitalizations=hospitalizations)
        return age_matrices[...,0], age_matrices[...,1]
    else:
        pruner = 0
        _, chain, _ = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, prune=pruner, prefix=prefix)
        random_indices = np.random.choice(chain.shape[0], size=samples, replace=False)
        random_samples = chain[random_indices, :]
        def get_matrix(x):
            params = x_to_params(x, pathogen, lockdown, option1, option2, NAG=NAG)
            solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
            return calculate_infection_matrices_from_solution(solution, params, NAG=NAG, hospitalizations=hospitalizations)
        age_matrices = jax.vmap(get_matrix)(random_samples)
        return age_matrices[...,0], age_matrices[...,1]

def plot_infection_matrix(ax, pathogen=None, seed=None, lockdown=None, option1=None, option2=None, NAG=None, age_group_names=None, census_age_pop=None, hospitalizations=False, prefix="", matrix=None):
    if matrix is None:
        age_infections_matrix, _ = get_infection_matrix(pathogen, seed, lockdown, option1, option2, NAG, census_age_pop, hospitalizations=hospitalizations, prefix=prefix)
        normalized_age_infections_matrix = age_infections_matrix / census_age_pop[:, None]
    else:
        normalized_age_infections_matrix = matrix
    # imshow of age_infections_matrix with age group names as x and y ticks
    im = ax.imshow(normalized_age_infections_matrix, cmap="viridis")
    ax.set_xticks(np.arange(NAG))
    ax.set_xticklabels(age_group_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(NAG))
    ax.set_yticklabels(age_group_names)
    ax.invert_yaxis()
    return im

def plot_same_age_infection(ax, matrices, age_group_idx, pathogens, colors, census_age_pop=None):
    values = []
    credible_intervals = []
    
    for i, matrix in enumerate(matrices):
        # Handle both single matrix and multiple matrices per pathogen
        if matrix.ndim == 2:
            matrix_list = [matrix]
        else:
            matrix_list = matrix
        
        pathogen_values = jnp.array([m[age_group_idx,age_group_idx]/m[age_group_idx,:].sum() for m in matrix_list])
        if census_age_pop is not None:
            pathogen_values *= census_age_pop.sum()/census_age_pop[age_group_idx]
        
        median_val = jnp.median(pathogen_values)
        ci_lower = jnp.percentile(pathogen_values, 2.5)
        ci_upper = jnp.percentile(pathogen_values, 97.5)
        
        values.append(median_val)
        credible_intervals.append((ci_lower, ci_upper))
    
    # if credible intervals overlap, add -0.1 to one of them and +0.1 to the other to separate them visually
    separation = jnp.zeros(len(matrices))
    needs_separation = jnp.zeros(len(matrices), dtype=bool)
    for i in range(len(matrices)):
        for j in range(i+1, len(matrices)):
            ci_i_lower, ci_i_upper = credible_intervals[i]
            ci_j_lower, ci_j_upper = credible_intervals[j]
            # Check if credible intervals overlap or if values are within 10% of each other
            intervals_overlap = not (ci_i_upper < ci_j_lower or ci_j_upper < ci_i_lower)
            values_close = jnp.abs(values[i] - values[j]) / jnp.maximum(values[i], values[j]) < 0.12
            if intervals_overlap or values_close:
                needs_separation = needs_separation.at[i].set(True)
                needs_separation = needs_separation.at[j].set(True)
                separation = separation.at[i].set(separation[i] - 0.15)
                separation = separation.at[j].set(separation[j] + 0.15)
    separation = jnp.clip(separation, -0.35, 0.35)
    
    # If all pathogens need separation, plot them in order: -0.3, -0.2, -0.1, 0, 0.1, 0.2
    if jnp.all(needs_separation):
        positions = jnp.array([-0.3, -0.18, -0.06, 0.06, 0.18, 0.3])
        separation = positions[:len(matrices)]
    for i in range(len(matrices)):
        ci_lower, ci_upper = credible_intervals[i]
        ax.scatter(separation[i], values[i], label=pathogens[i], color=colors[i], marker="o")
        ax.plot([separation[i], separation[i]], [ci_lower, ci_upper], color=colors[i], linewidth=2)
    
    ax.set_xlim(-0.5,0.5)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def plot_infections_versus(ax, matrices, age_group_indices, pathogens, colors, census_age_pop):
    for i, matrix in enumerate(matrices):
        # Handle both single matrix and multiple matrices per pathogen
        if matrix.ndim == 2:
            matrix_list = [matrix]
        else:
            matrix_list = matrix
        
        x_values = []
        y_values = []
        for m in matrix_list:
            x_value = m[:,age_group_indices[0]].sum()/census_age_pop[age_group_indices[0]].sum()
            y_value = m[:,age_group_indices[1]].sum()/census_age_pop[age_group_indices[1]].sum()
            x_values.append(x_value)
            y_values.append(y_value)
        
        x_median = jnp.median(jnp.array(x_values))
        y_median = jnp.median(jnp.array(y_values))
        x_ci_lower = jnp.percentile(jnp.array(x_values), 2.5)
        x_ci_upper = jnp.percentile(jnp.array(x_values), 97.5)
        y_ci_lower = jnp.percentile(jnp.array(y_values), 2.5)
        y_ci_upper = jnp.percentile(jnp.array(y_values), 97.5)
        
        # ax.scatter(x_median, y_median, label=pathogens[i], color=colors[i], marker="o")
        ax.plot([x_ci_lower, x_ci_upper], [y_median, y_median], color=colors[i], linewidth=2)
        ax.plot([x_median, x_median], [y_ci_lower, y_ci_upper], color=colors[i], linewidth=2)
    # find minimum of x and y limits and set equal, find maximum of x and y limits and set equal
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    min_limit = min(x_min, y_min)
    max_limit = max(x_max, y_max)
    ax.set_xlim(min_limit, max_limit)
    ax.set_ylim(min_limit, max_limit)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

import matplotlib.transforms as mtransforms
def plot_age_figure(axes, pathogens, colors, option1, option2s, seeds, lockdown, NAG, CENSUS_AGE_POP, AGE_GROUP_NAMES, logD=False, age_adjusted=False, samples=100, prefix="", save_data=False, load_data=False):
    ax_top, ax_bottom = axes
    if not load_data:
        matrices = {}
        hosp_matrices = {}
        sampled_matrices = {}
        sampled_hosp_matrices = {}
        for pi, (pathogen, option2, seed) in enumerate(zip(pathogens, option2s, seeds)):
            matrices[pathogen], hosp_matrices[pathogen] = get_infection_matrix(pathogen, seed, lockdown, option1, option2, NAG, CENSUS_AGE_POP, hospitalizations=False, prefix="emcee_")
            sampled_matrices[pathogen], sampled_hosp_matrices[pathogen] = get_infection_matrix(pathogen, seed, lockdown, option1, option2, NAG, CENSUS_AGE_POP, hospitalizations=False, samples=samples, prefix=prefix)
        if save_data:
            np.savez_compressed("age_figure_data.npz", matrices=matrices, hosp_matrices=hosp_matrices, sampled_matrices=sampled_matrices, sampled_hosp_matrices=sampled_hosp_matrices)
    else:
        matrices = {}
        hosp_matrices = {}
        sampled_matrices = {}
        sampled_hosp_matrices = {}
        for pi, (pathogen, option2, seed) in enumerate(zip(pathogens, option2s, seeds)):
            data = np.load("age_figure_data.npz", allow_pickle=True)
            matrices = data["matrices"].item()
            hosp_matrices = data["hosp_matrices"].item()
            sampled_matrices = data["sampled_matrices"].item()
            sampled_hosp_matrices = data["sampled_hosp_matrices"].item()
            break
    for pi, pathogen in enumerate(pathogens):
        normalized_age_infections_matrix = matrices[pathogen] / CENSUS_AGE_POP[:, None]
        plot_infection_matrix(ax_top[pi//3, pi%3], age_group_names=AGE_GROUP_NAMES, NAG=NAG, matrix=normalized_age_infections_matrix)
        ax_top[pi//3, pi%3].set_title(nice_names.get(pathogen, pathogen))
    for age_group_idx in range(NAG):
        if age_adjusted:
            pop_arg = CENSUS_AGE_POP
        else:            
            pop_arg = None
        plot_same_age_infection(ax_bottom[0, age_group_idx], [sampled_matrices[pathogen] for pathogen in pathogens], age_group_idx, pathogens, colors, pop_arg)
        if logD:
            ax_bottom[0, age_group_idx].set_yscale("log")
            ax_bottom[0, age_group_idx].set_ylim([2e-2,4])
            if age_group_idx > 0:
                ax_bottom[0, age_group_idx].spines['left'].set_visible(False)
                ax_bottom[0, age_group_idx].set_yticks([])
        ax_bottom[0, age_group_idx].set_xlabel(AGE_GROUP_NAMES[age_group_idx])
    plot_infections_versus(ax_top[0,3], [sampled_matrices[pathogen] for pathogen in pathogens], [slice(0, 3), -1], pathogens, colors, CENSUS_AGE_POP)
    ax_top[0,3].set_xlabel("<1y")
    ax_top[0,3].set_ylabel(">65y")
    plot_infections_versus(ax_top[1,3], [sampled_hosp_matrices[pathogen] for pathogen in pathogens], [slice(0, 3), -1], pathogens, colors, CENSUS_AGE_POP)
    ax_top[1,3].set_xlabel("<1y")
    ax_top[1,3].set_ylabel(">65y")
    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="None", markersize=5,
            markerfacecolor=colors[i], markeredgecolor=colors[i],
            label=nice_names.get(pathogen, pathogen)
        )
        for i, pathogen in enumerate(pathogens)
    ]
    ax_bottom[0,NAG].axis("off")
    ax_bottom[0,NAG].legend(handles=legend_handles, loc="center", frameon=False, title="Pathogen")
    ax_bottom[0,0].set_ylabel("Within-group transmission", fontsize=9)

    trans_A = mtransforms.blended_transform_factory(ax_top[0,0].transAxes, ax_top[0,3].transAxes)
    # trans_D = mtransforms.blended_transform_factory(ax[0,0].transAxes, ax[2,0].transAxes)
    ax_top[0,0].text(-0.5, 1.1, "A", transform=trans_A, fontsize=16, fontweight="bold")
    ax_top[0,3].text(-0.5, 1.1, "C", transform=ax_top[0,3].transAxes, fontsize=16, fontweight="bold")
    ax_top[1,3].text(-0.5, 1.1, "D", transform=ax_top[1,3].transAxes, fontsize=16, fontweight="bold")
    # ax[2,0].text(-0.5, 1.1, "D", transform=trans_D, fontsize=16, fontweight="bold")
    ax_bottom[0,0].text(-1.1, 1, "B", transform=ax_bottom[0,0].transAxes, fontsize=16, fontweight="bold")

def plot_age_profiles(ax, pathogens):
    return None

def plot_single_pathogen_violin(ax, pathogen_data, color, pathogen_name):
    """Plot violin for a single pathogen."""
    data_clean = pathogen_data.dropna()
    parts = ax.violinplot(data_clean, positions=[0], vert=True, showmedians=True)
    
    # Add scattered individual points with random vertical offset
    np.random.seed(260603)
    jitter = np.random.normal(0, 0.04, size=len(data_clean))
    print(f"Plotting {len(data_clean)} points for {pathogen_name}")
    ax.scatter(np.full(len(data_clean), 0) + jitter, data_clean, 
              color=color, s=20, alpha=4/np.sqrt(len(data_clean)), edgecolors='none')
    
    # Style violin as dashed outline in black
    for pc in parts['bodies']:
        pc.set_facecolor('none')
        pc.set_edgecolor('black')
        pc.set_linestyle('--')
        pc.set_linewidth(1.5)
    
    # Style median bar in black
    for partname in ('cmedians', 'cbars', 'cmaxes', 'cmins'):
        if partname in parts:
            parts[partname].set_color('black')
            parts[partname].set_linewidth(1.5)
    
    # Annotate median value
    median_val = data_clean.median()
    ax.text(0.25, median_val, f'{median_val:.1f}', 
            va='center', ha='center', fontsize=7, color='black')
    
    # Turn off borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.set_xticks([])
    ax.set_xlabel(short_names.get(pathogen_name, pathogen_name))

    return data_clean.max() * 1.05  # Return max value for consistent x-axis limits

def supression_violin(axes, pathogens, colors):
    """Plot violin plots for multiple pathogens in a grid."""
    supression_data = pd.read_csv("Data/Processed/FluNet_suppression_duration_by_country_pathogen.csv")
    nice_names = {"InfluenzaA": "Influenza A", "InfluenzaB": "Influenza B"}
    xmax = 0
    for i, pathogen in enumerate(pathogens):
        pathogen_data = supression_data[(supression_data["pathogen"] == pathogen) & (supression_data["dq_pass"] == True)]["suppression_duration_months"]
        # pathogen_name = nice_names.get(pathogen, pathogen)
        max_val = plot_single_pathogen_violin(axes[i], pathogen_data, colors[i], pathogen)
        xmax = max(xmax, max_val)
        # Only show y-axis label on leftmost plot
        if i > 0:
            axes[i].spines['left'].set_visible(False)
            axes[i].set_yticks([])
    for ax in axes:
        ax.set_ylim(0, xmax)
    axes[0].set_ylabel("Suppression duration (months)", fontsize=9)
    axes[0].set_yticks(np.arange(0, xmax+1, 12))

def plot_supression_rank_heatmap(ax, pathogens):
    supression_data = pd.read_csv("Data/Processed/FluNet_suppression_duration_by_country_pathogen.csv")
    # filter to dq_pass == True
    supression_data = supression_data[supression_data["dq_pass"] == True]
    # for each country, find the rank of each pathogen by suppression_duration_months
    supression_data["rank"] = supression_data.groupby("country")["suppression_duration_months"].rank(method="min", ascending=False)
    pd.set_option('display.max_rows', None)
    print(supression_data[["country", "pathogen", "suppression_duration_months", "rank"]])
    # create a pivot table with index country, columns pathogen, values rank
    pivot = supression_data.pivot(index="country", columns="pathogen", values="rank")
    print(pivot)
    # plot heatmap of how often one pathogen is ranked higher than the other
    image = np.zeros((len(pathogens), len(pathogens)))
    for i in range(len(pathogens)):
        for j in range(i+1, len(pathogens)):
            pathogen_i = pathogens[i]
            pathogen_j = pathogens[j]
            count_i_higher = (pivot[pathogen_i] < pivot[pathogen_j]).sum()
            count_j_higher = (pivot[pathogen_j] < pivot[pathogen_i]).sum()
            total_count = count_i_higher + count_j_higher
            if total_count > 0:
                image[i, j] = count_i_higher / total_count
                image[j, i] = count_j_higher / total_count
    # reorder the pathogens by average rank
    order = ["Adenovirus", "Parainfluenza", "RSV",  "Metapneumovirus", "InfluenzaA", "InfluenzaB"]
    image = image[[pathogens.index(p) for p in order], :][:, [pathogens.index(p) for p in order]]
    # only plot lower triangle
    for i in range(len(order)):
        for j in range(i+1, len(order)):
            image[i, j] = np.nan
    np.fill_diagonal(image, np.nan)
    image = image[1:, :-1]
    print(image)
    #annotate with percentages
    for i in range(len(order)-1):
        for j in range(i, len(order)-1):
            if not np.isnan(image[j, i]):
                ax.text(i, j, f"{image[j, i]*100:.0f}%", ha="center", va="center", color="white")
    # diagonal is NaN
    im = ax.imshow(image, vmin=0.5, vmax=1)
    # set ticks and labels
    ax.set_xticks(np.arange(len(pathogens)-1))
    ax.set_yticks(np.arange(len(pathogens)-1))
    ax.set_xticklabels([short_names.get(p, p) for p in order[:-1]], rotation=45, ha="right")
    ax.set_yticklabels([short_names.get(p, p) for p in order[1:]])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return im

short_reasons = {"low_activity_all_months_below_threshold": "low_activity",
                 "insufficient_post_nonzero_months": "low_activity_post",
                 "zero_testing_explains_excess_gap": "testing_gap",
                 "insufficient_pre2020_baseline": "insufficient_baseline",
                 "no_excess_gap_vs_pre2020_country_level": "no_excess_gap",
                 "no_dip_after_anchor": "no_suppression",
                 }
short_country_names = {"occupied_Palestinian_territory_including_east_Jerusalem": "Palestine",
                        "Venezuela_Bolivarian_Republic_of": "Venezuela",
                        "Serbia_and_Montenegro_2003-2006": "Serbia_and_Montenegro",
                        "Kosovo_in_accordance_with_UN_Security_Council_resolution_1244_1999": "Kosovo",
                        "Bolivia_Plurinational_State_of": "Bolivia",
                        "Democratic_People_s_Republic_of_Korea": "North_Korea",
                        "Iran_Islamic_Republic_of": "Iran",
                        "Lao_People_s_Democratic_Republic": "Laos",
                        "Netherlands_Kingdom_of_the": "Netherlands",
                        "Democratic_Republic_of_the_Congo": "DRC",
                        "United_Republic_of_Tanzania": "Tanzania",
                        "United_States_of_America": "USA",
                        "United_Kingdom_England": "England",
                        "United_Kingdom_Scotland": "Scotland",
                        "United_Kingdom_Wales": "Wales",
                        "United_Kingdom_Northern_Ireland": "Northern_Ireland",
                        }
def plot_FluNet(ax, pathogen, country, color='k', linewidth=1, flag_exclusions=False):
    data = pd.read_csv(f"Data/Processed/FluNetTimeseries/{country}__{pathogen}.csv", parse_dates=['month_start'])
    supression_data = pd.read_csv("Data/Processed/FluNet_suppression_duration_by_country_pathogen.csv")
    def sanitize_filename_token(value: str) -> str:
        token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        token = token.strip("._-")
        return token or "unknown"
    # apply to country names
    supression_data.loc[:, "country"] = supression_data["country"].apply(sanitize_filename_token)
    # restrict to month_start >= 2015-10-01
    data = data[(data["month_start"] >= pd.to_datetime("2015-10-01")) & (data["month_start"] < pd.to_datetime("2026-05-01"))]
    if (len(data) > 0) and (data["count"].sum() > 0):
        ax.plot(data["month_start"], data["count"], label=country, color=color, linewidth=linewidth)
    else:
        years = pd.date_range(start=pd.to_datetime("2015-10-01"), end=pd.to_datetime("2026-05-01"), freq='MS')
        ax.plot(years, [0]*len(years), label=country, color=color, linewidth=linewidth)
        if flag_exclusions:
            ax.annotate("all_zero", xy=(0.01, 0.99), xycoords='axes fraction', fontsize=6, color='magenta', ha='left', va='top')
    if flag_exclusions and ((supression_data["pathogen"] == pathogen) & (supression_data["country"] == country) & (supression_data["status"] == "excluded")).any():
        reason = supression_data[(supression_data["pathogen"] == pathogen) & (supression_data["country"] == country) & (supression_data["status"] == "excluded")]["reason"].iloc[0]
        short_reason = short_reasons.get(reason, reason)
        ax.annotate(short_reason, xy=(0.01, 0.99), xycoords='axes fraction', fontsize=6, color='magenta', ha='left', va='top')

    # Add suppression period indicator
    pre_2020_data = data[data["month_start"] < pd.to_datetime("2020-03-01")]
    if len(pre_2020_data) > 0 and not ((supression_data["pathogen"] == pathogen) & (supression_data["country"] == country) & (supression_data["status"] == "excluded")).any():
        pre_2020_max = pre_2020_data["count"].max()
        threshold = pre_2020_max / 20
        post_2020_data = data[data["month_start"] >= pd.to_datetime("2020-03-01")]
        suppressed = post_2020_data[post_2020_data["count"] < threshold]
        if len(suppressed) > 0:
            supp_start = suppressed.iloc[0]["month_start"]
            not_suppressed = post_2020_data[(post_2020_data["month_start"] > supp_start) & (post_2020_data["count"] >= threshold)]
            supp_end = not_suppressed.iloc[0]["month_start"] if len(not_suppressed) > 0 else suppressed.iloc[-1]["month_start"]
            y_pos = threshold
            ax.plot([supp_start, supp_end], [y_pos, y_pos], color='red', linewidth=linewidth)
            time_diff = supp_end - supp_start
            time_amount = f"{time_diff.days // 30}"
            last_pre_time = data[data["month_start"] < pd.to_datetime("2020-03-01")]["month_start"].iloc[-1]
            ax.annotate(time_amount, xy=(last_pre_time + time_diff/2, threshold), xytext=(0,2), textcoords='offset points', ha='center', va="bottom", color='red')

def plot_FluNet_chunk(axes, countries_chunk):
        for ci, country in enumerate(countries_chunk):
            for pi, pathogen in enumerate(flunet_pathogens):
                plot_FluNet(axes[ci, pi], pathogen, country, flag_exclusions=True)
                axes[ci, pi].set_yticklabels([])
                axes[ci, pi].set_xticklabels([])
                # Remove all spines except bottom
                axes[ci, pi].spines['top'].set_visible(False)
                axes[ci, pi].spines['right'].set_visible(False)
                axes[ci, pi].spines['left'].set_visible(False)
                axes[ci, pi].set_yticks([])
                # Add x tick labels only for middle column
                axes[ci, pi].set_xticks(pd.to_datetime(["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01", "2026-01-01",]))
                if ci == len(countries_chunk)-1:
                    axes[ci, pi].set_xticklabels(["", "2017", "", "2019", "", "2021", "", "2023", "", "2025", "",], fontsize=6)
                    for label in axes[ci, pi].get_xticklabels():
                        label.set_rotation(45)
                        label.set_horizontalalignment('right')
                        label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
                else:
                    axes[ci, pi].set_xticklabels([])
                if ci == 0:
                    axes[ci, pi].set_title(short_names.get(pathogen, pathogen))
                if pi == 0:
                    axes[ci, pi].set_ylabel(short_country_names.get(country, country), rotation=0, ha='right')
                    ax2 = axes[ci, pi].twinx()
                    # ax2.set_ylabel("Detected cases", rotation=90, va='center', fontsize=6)
                    ax2.set_yticks([])
                    ax2.spines['right'].set_visible(False)
                    ax2.spines['left'].set_visible(True)
                    ax2.spines['top'].set_visible(False)
                    ax2.yaxis.set_label_position('left')
                    ax2.yaxis.tick_left()

def plot_suppression_durations(fig, pathogens, countries, colors):
    gs = fig.add_gridspec(9, 6, height_ratios=[1, 1, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 3])
    
    # KPSC plots in top row (row 0)
    kpsc_pathogens = ["RSV","Metapneumovirus","Parainfluenza3","Adenovirus","InfluenzaA","InfluenzaB",]
    kpsc_axis = np.empty(len(kpsc_pathogens), dtype=object)
    for i, pathogen in enumerate(kpsc_pathogens):
        ax = fig.add_subplot(gs[0, i])
        kpsc_axis[i] = ax
        # color = colors[i]
        kpsc_proportion_positive_incidence_plot(
            ax, pathogen=pathogen, AGE_GROUP_NAMES=AGE_GROUP_NAMES, title="",
            color="k", aggregation="MS", factor=100000,
            annotations="simple", definition="", label="Data", hosp=True, dedup=True)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlim(pd.to_datetime("2015-10-01"), pd.to_datetime("2025-10-01"))
        # Remove all spines except bottom
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_ylabel("")
        ax.set_yticks([])
        if i==0:
            ax.set_ylabel("KPSC", rotation=0, ha='right', labelpad=25)
            ax2 = ax.twinx()
            ax2.set_ylabel("Estimated\n+ve hospitalizations", rotation=90, va='center', fontsize=6)
            ax2.set_yticks([])
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(True)
            ax2.spines['top'].set_visible(False)
            ax2.yaxis.set_label_position('left')
            ax2.yaxis.tick_left()

        ax.set_title(short_names.get(pathogen, pathogen))
        ax.set_xticks(pd.to_datetime(["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01",]))
        ax.set_xticklabels(["", "2017", "", "2019", "", "2021", "", "2023", "", "2025",], fontsize=6)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
        
    # season age plots in row 1
    season_age_axes = np.empty(len(kpsc_pathogens), dtype=object)
    for i, pathogen in enumerate(kpsc_pathogens):
        ax = fig.add_subplot(gs[1, i])
        season_age_axes[i] = ax
        season_plot(ax, pathogen, False, True, True)
        ax.set_xticks([])
        # ax.set_xticklabels(["", "2016/17", "", "2018/19", "", "2020/21", "", "2022/23", "", "2024/25",], fontsize=6)
        # ax.set_xticklabels(["", "", "", "", "", "", "", "", "", "",], fontsize=6)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
    season_age_axes[0].set_ylabel("KPSC", rotation=0, ha='right', labelpad=25)
    sax2 = season_age_axes[0].twinx()
    sax2.set_ylabel("Proportion of\nhospitalizations", rotation=90, va='center', fontsize=6)
    sax2.set_yticks([])
    sax2.spines['right'].set_visible(False)
    sax2.spines['left'].set_visible(True)
    sax2.spines['top'].set_visible(False)
    sax2.yaxis.set_label_position('left')
    sax2.yaxis.tick_left()

    # create a legend for the hsv_colors and AGE_GROUP_NAMES, and put it in row 2, spanning all columns
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis("off")
    legend_handles = [
        plt.Line2D(
            [0], [0], marker="s", linestyle="None", markersize=5,
            markerfacecolor=hsv_colors[i], markeredgecolor=hsv_colors[i],
            label=AGE_GROUP_NAMES[i]
        )
        for i in range(len(AGE_GROUP_NAMES))
    ]
    leg = legend_ax.legend(handles=legend_handles, bbox_to_anchor=(0.5, 1.5), loc="center", frameon=False, title="", ncol=len(AGE_GROUP_NAMES), fontsize=6)
    leg.set_in_layout(False)
    legend_ax.set_in_layout(False)

    # FluNet plots in middle rows (rows 3-7)
    country_axes = np.empty((len(countries), len(pathogens)), dtype=object)
    for j, country in enumerate(countries):
        for i, pathogen in enumerate(pathogens):
            ax = fig.add_subplot(gs[j+3, i])
            country_axes[j, i] = ax
            plot_FluNet(ax, pathogen, country, linewidth=0.5)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            # Remove all spines except bottom
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            # Add x tick labels only for middle column
            ax.set_xticks(pd.to_datetime(["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01", "2026-01-01",]))
            # if i == 5:
            #     ax.set_xticklabels(["'16", "", "", "", "'20", "", "", "", "'24", "", ""])
            # Add "Date" label only to middle column bottom row
            # Add country titles at left
            if j == 0:
                ax.set_title(short_names.get(pathogen, pathogen))
            if i == 0:
                ax.set_ylabel(country, rotation=0, ha='right', labelpad=20)
                ax2 = ax.twinx()
                ax2.set_ylabel(f"Detected\ncases", rotation=90, va='center', fontsize=6)
                ax2.set_yticks([])
                ax2.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(True)
                ax2.spines['top'].set_visible(False)
                ax2.yaxis.set_label_position('left')
                ax2.yaxis.tick_left()
            # in last row add small x-axis labels every two years at 45 degrees
            if j == len(countries)-1:
                ax.set_xticklabels(["", "2017", "", "2019", "", "2021", "", "2023", "", "2025", "",], fontsize=6)
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment('right')
                    label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
            # for PIV in India, grey out plot after 2021-01-02 and annotate "No data"
            if country=="India" and pathogen=="Parainfluenza":
                ax.axvspan(pd.to_datetime("2021-03-01"), pd.to_datetime("2026-05-01"), color='grey', alpha=0.2, linewidth=0)
                ax.annotate("No data", xy=(pd.to_datetime("2021-06-01"), 50), xycoords='data', fontsize=6, color='black', ha='left', va='center')

            # if j == 0:
            #     ax.set_ylabel(short_names.get(pathogen, pathogen), rotation=0, ha='right')
    
    # Violin plots in bottom row (row 8)
    violin_axis = np.empty(len(pathogens), dtype=object)
    for i in range(len(pathogens)):
        violin_axis[i] = fig.add_subplot(gs[8, i])
    supression_violin(violin_axis, pathogens, colors)
    
    # big label "C" on top left of kpsc plots, "B" on top left of country plots, "A" on top left of violin plots
    kpsc_axis[0].text(-0.5, 1.1, "A", transform=kpsc_axis[0].transAxes, fontsize=16, fontweight="bold")
    season_age_axes[0].text(-0.5, 1.1, "B", transform=season_age_axes[0].transAxes, fontsize=16, fontweight="bold")
    country_axes[0,0].text(-0.5, 1.1, "C", transform=country_axes[0,0].transAxes, fontsize=16, fontweight="bold")
    violin_axis[0].text(-0.5, 1.1, "D", transform=violin_axis[0].transAxes, fontsize=16, fontweight="bold")

def plot_fits(axes, n_samples=100, save_data=False, load_data=False):
    aggregation = "MS"
    agg_factor = 30.44
    factor = 100000
    for pathogen_idx in range(len(pathogens)):
        ax = axes[:,pathogen_idx]
        pathogen, option2, seed = pathogens[pathogen_idx], option2s[pathogen_idx], seeds[pathogen_idx]
        _, _, _, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=1e-9, incidence_data=False, hosp=True, NAG=NAG, dedup=True)
        chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=0, prefix="")
        np.random.seed(260604)
        # choose n_samples random rows from chain
        random_indices = np.random.choice(chain.shape[0], size=n_samples, replace=False)
        random_samples = chain[random_indices, :]
        if not load_data:
            from fit_MCMC import run_simulation
            def get_solution(x):
                params = x_to_params(x, pathogen, lockdown, option1, option2, NAG=NAG)
                solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
                return solution
            solutions = jax.jit(jax.vmap(get_solution))(random_samples)
        else:
            solutions = np.load("Data/Processed/fit_samples_" + pathogen + option2 + str(seed) + ".npy", allow_pickle=True)
        if save_data:
            np.save("Data/Processed/fit_samples_"+pathogen+option2+str(seed)+".npy", jax.tree_util.tree_map(np.array, solutions))
        for sample_i in range(n_samples):
            print(f"Pathogen {pathogen_idx+1}/{len(pathogens)}: {pathogen}, sample {sample_i+1}/{n_samples}", end="\r")
            if load_data:
                solution = solutions[sample_i]
            else:
                solution = jax.tree_util.tree_map(lambda x: x[sample_i], solutions)
            for age_group_idx, age_group_name in enumerate(AGE_GROUP_NAMES):
                lockdown_incidence_plot(ax[age_group_idx], STATE0, None, POINTS, pd.to_datetime('2020-03-01'),
                                        solution=solution,
                                        by_age=True, select_age_group=age_group_idx, AGE_GROUP_NAMES=AGE_GROUP_NAMES, AGE_GROUPS=AGE_GROUPS,
                                        p_time_to_obs=p_time_to_obs, NAG=NAG,
                                        test_data=data_full,daily_hospitalization_rates=daily_hospitalization_rates,aggregation=aggregation,
                                        uncertainty="draw",
                                        color=hsv_colors[age_group_idx], factor=factor*agg_factor, label="Simulation", linewidth=0.5, alpha=0.01)
        for age_group_idx, age_group_name in enumerate(AGE_GROUP_NAMES):
            kpsc_proportion_positive_incidence_plot(ax[age_group_idx], pathogen=pathogen,
                                                    AGE_GROUPS=AGE_GROUPS, AGE_GROUP_NAMES=AGE_GROUP_NAMES, select_age_group=age_group_idx,
                                                    aggregation=aggregation, factor=factor, annotations=False, definition="", label="Data", hosp=True, dedup=True,
                                                    title="", color="k", linewidth=0.5)
            if pathogen_idx == 0:
                ax[age_group_idx].set_ylabel(age_group_name)
            else:
                ax[age_group_idx].set_ylabel("")
            if age_group_idx == 0:
                ax[age_group_idx].set_title(short_names.get(pathogen, pathogen))
            ax[age_group_idx].set_xticks(pd.to_datetime(["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01",]))
            if age_group_idx == len(AGE_GROUP_NAMES)-1:
                ax[age_group_idx].set_xticklabels(["", "2017", "", "2019", "", "2021", "", "2023", "", "2025",], fontsize=6)
                for label in ax[age_group_idx].get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment('right')
                    label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
            else:
                ax[age_group_idx].set_xticklabels([])
            ax[age_group_idx].legend().set_visible(False)
            ax[age_group_idx].tick_params(axis='y', labelsize=6)
        print(f"\n", end="")

def plot_contact(ax, pathogen, option2, seed, n_samples=100):
    chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=0, prefix="")
    param_names, _ = parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)
    seasonality_idx = param_names.index("SEASONALITY")
    offset_idx = param_names.index("OFFSET")
    np.random.seed(260619)
    # choose n_samples random rows from chain
    random_indices = np.random.choice(chain.shape[0], size=n_samples, replace=False)
    random_samples = chain[random_indices, :]
    def get_forcing_and_cntct(x):
        _, cntct = x_to_params(x, pathogen, lockdown, option1, option2, NAG=NAG, return_contact=True)
        seasonality = x[seasonality_idx]
        offset = x[offset_idx]
        forcing = (1+seasonality*jnp.cos(2*jnp.pi*((POINTS-274)/365-offset)))
        return forcing, cntct
    results = jax.jit(jax.vmap(get_forcing_and_cntct))(random_samples)
    for sample_i in range(n_samples):
        forcing, cntct = jax.tree_util.tree_map(lambda x: x[sample_i], results)
        ax.plot(POINTS, forcing, color="k", alpha=0.01, linewidth=0.5)
        ax.plot(POINTS, cntct[-len(POINTS):], color="hotpink", alpha=0.01, linewidth=0.5)

def plot_susceptibility(ax, pathogen, option2, seed, n_samples=100, load_data=False):
    chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=0, prefix="")
    _, _, _, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=1e-9, incidence_data=False, hosp=True, NAG=NAG, dedup=True)
    chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=0, prefix="")
    np.random.seed(260604)
    # choose n_samples random rows from chain
    random_indices = np.random.choice(chain.shape[0], size=n_samples, replace=False)
    random_samples = chain[random_indices, :]
    if not load_data:
        from fit_MCMC import run_simulation
        def get_solution(x):
            params = x_to_params(x, pathogen, lockdown, option1, option2, NAG=NAG)
            solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
            return solution
        solutions = jax.jit(jax.vmap(get_solution))(random_samples)
    else:
        solutions = np.load("Data/Processed/fit_samples_" + pathogen + option2 + str(seed) + ".npy", allow_pickle=True)
    for sample_i in range(n_samples):
        print(f"Plotting susceptibility sample {sample_i+1}/{n_samples} for {pathogen} {option2} seed {seed}", end="\r")
        if load_data:
            solution = solutions[sample_i]
        else:
            solution = jax.tree_util.tree_map(lambda x: x[sample_i], solutions)
        params = x_to_params(random_samples[sample_i], pathogen, lockdown, option1, option2, NAG=NAG)
        lockdown_susceptibility_plot(ax, STATE0, params, PERIOD, POINTS, date_to_t('2020-03-19'), solution=solution, relative=True, proportion=True, by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES, NAG=NAG,
                                     linewidth=0.5, alpha=0.01)
    print(f"\n", end="")

def plot_mcmc_traces(axes, pathogen, option2, seed, n_walkers=64, prune=0):
    chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=prune, prefix="")
    param_names, _ = parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)
    n_params = len(param_names)
    for param_idx in range(n_params):
        ax = axes[param_idx]
        for walker_idx in range(n_walkers):
            ax.plot(chain[walker_idx::n_walkers, param_idx], alpha=0.4)
        ax.set_title(f"{param_names[param_idx]}")

def plot_mcmc_corner(pathogen, lockdown, option1, option2, seed, prune=0):
    chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=prune, prefix="")
    print(chain.shape)
    param_names, _ = parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)
    fig = corner.corner(chain, labels=param_names,  show_titles=True, title_fmt=".4f", title_kwargs={"fontsize": 8})
    plt.savefig(f"Figures/mcmc_corner_{pathogen}_{option2}_{seed}.pdf")
    plt.close(fig)

def get_srel1_from_constrained_immunity(extra_immunity, first_immunity, first_dis_inf_factor):
    srel, _ = constrained_immunity(extra_immunity, first_immunity, first_dis_inf_factor)
    return srel[1]

def plot_r0_vs_first_immunity(axes, pathogen, option2, seed, color, r0_base=15.24, prune=0, n_samples=50000):
    chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=prune, prefix="")
    # randomly sample n_samples rows from chain
    np.random.seed(260604)
    random_indices = np.random.choice(chain.shape[0], size=n_samples, replace=False)
    chain = chain[random_indices, :]
    param_names, _ = parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)
    beta_idx = param_names.index("BETA")
    beta_values = chain[:, beta_idx]
    if "Influenza" in pathogen:
        eidx = param_names.index("EXTRA_IMMUNITY")
        fidx = param_names.index("FIRST_IMMUNITY")
        fdifdx = param_names.index("FIRST_DIS_INF_FACTOR")
        srel1_samples = jax.jit(jax.vmap(get_srel1_from_constrained_immunity))(chain[:,eidx], chain[:,fidx], chain[:,fdifdx])
    else:
        srel1_idx = param_names.index("S_REL1")
        srel1_samples = chain[:, srel1_idx]
    r0_values = r0_base * beta_values * (3.0 + 1.9 * (pathogen in ["RSV", "Metapneumovirus"]))
    immunity_values = 1 - srel1_samples
    axes.scatter(r0_values, immunity_values, alpha=1, color=color, s=0.001, linewidths=0, rasterized=True)
    return r0_values, immunity_values

def plot_odr_best_fit(ax, r0_values_by_pathogen, immunity_values_by_pathogen, n_iterations=10000):
    from scipy import odr
    def immunity_func(B, x):
        return 1 / (1 + np.exp(-B[0])) * (B[2] - (1 / x)**B[1])
    
    linear_model = odr.Model(immunity_func)

    mult_samples = np.zeros(n_iterations)
    exponent_samples = np.zeros(n_iterations)
    shape_samples = np.zeros(n_iterations)

    # Create a dense x-grid for plotting
    x_grid = np.linspace(1.0, np.max([np.max(r0_values) for r0_values in r0_values_by_pathogen.values()]), 10000)
    x_grid_transformed = x_grid

    # We will store the evaluated y-values for each sampled line here
    y_lines = np.zeros((n_iterations, len(x_grid)))
    y_predictions = np.zeros((n_iterations, len(x_grid)))

    for i in range(n_iterations):
        r0_draws = []
        srel1_draws = []
        
        # Draw exactly one (R0, Immunity) pair from each pathogen's joint posterior
        for pathogen in r0_values_by_pathogen.keys():
            sample_idx = np.random.choice(len(r0_values_by_pathogen[pathogen]))
            r0_draws.append(r0_values_by_pathogen[pathogen][sample_idx])
            srel1_draws.append(immunity_values_by_pathogen[pathogen][sample_idx])
            
        # Transform x-data
        x_data = np.array(r0_draws)
        y_data = np.array(srel1_draws)
        
        # 2. Run Orthogonal Distance Regression
        # We don't need to pass weights (wd, we) here because the Monte Carlo 
        # sampling process itself naturally weights the parameter space!
        data = odr.Data(x_data, y_data)
        
        # Provide a rough initial guess for the solver (scale=1.0, exponent=1.0, shape=1.0)
        myodr = odr.ODR(data, linear_model, beta0=[1.0, 1.0, 1.0])
        output = myodr.run()
        
        # Extract optimized scale and intercept
        mult_samples[i] = output.beta[0]
        exponent_samples[i] = output.beta[1]
        shape_samples[i] = output.beta[2]
        
        # Evaluate the line on our plotting grid
        y_lines[i, :] = 1 / (1 + np.exp(-mult_samples[i])) * (shape_samples[i] - (1 / x_grid_transformed)**exponent_samples[i])

        res_sd = output.res_var**0.5
        y_predictions[i, :] = y_lines[i, :] + np.random.normal(0, res_sd, size=len(x_grid))
    print(f"ODR scale: {np.median(1 / (1 + np.exp(-mult_samples))):.4f} (95% CI: {1 / (1 + np.exp(-np.percentile(mult_samples, 97.5))):.4f} - {1 / (1 + np.exp(-np.percentile(mult_samples, 2.5))):.4f})"
          f", exponent: {np.median(exponent_samples):.4f} (95% CI: {np.percentile(exponent_samples, 2.5):.4f} - {np.percentile(exponent_samples, 97.5):.4f})"
          f", shape: {np.median(shape_samples):.4f} (95% CI: {np.percentile(shape_samples, 2.5):.4f} - {np.percentile(shape_samples, 97.5):.4f})"
          )
    # # plot line of best fit with confidence interval from bootstrap
    y_median = np.median(y_lines, axis=0)
    y_lower = np.percentile(y_predictions, 2.5, axis=0)
    y_upper = np.percentile(y_predictions, 97.5, axis=0)
    # Filter out x values where the predictions go completely out of bounds (below 0 or above 1)
    valid_mask = (y_median >= 0) & (y_median <= 1)
    x_plot = x_grid[valid_mask]
    y_median_plot = y_median[valid_mask]
    # Also ensure bounds are valid for the fill_between
    y_lower_fill = np.clip(y_lower, 0, 1)
    y_upper_fill = np.clip(y_upper, 0, 1)
    ax.plot(x_plot, y_median_plot, color='grey', linestyle='-', label="ODR Best fit", zorder=0)
    ax.fill_between(x_grid, y_lower_fill, y_upper_fill, color='k', linewidths=0, alpha=0.1, label="95% CI", zorder=0)

def plot_heatmaps_and_best_fit(fig, pathogens, option2s, seeds, colors, run_save_path, good_simulations, r0_base, fit_line=True, prune=0):
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.6)
    fit_ax = fig.add_subplot(gs[0, :])
    axes_hm = [fig.add_subplot(gs[1, i]) for i in range(2)]
    generate_2d_heatmap_plot(axes_hm[0], run_save_path, good_simulations, NAG=NAG, p1=0, p2=8, outcome="suppression_length", cbar=True, r0_base=r0_base)
    generate_2d_heatmap_plot(axes_hm[1], run_save_path, good_simulations, NAG=NAG, p1=0, p2=8, outcome="infectors_in_group_0123", cbar=True, r0_base=r0_base)
    # generate_2d_heatmap_plot(axes_hm[2], run_save_path, good_simulations, NAG=NAG, p1=0, p2=8, outcome="infectors_in_group_6", cbar=True, r0_base=r0_base)
    for ax in axes_hm:
        ax.set_xscale('log')
        ax.set_xticks([1,2,3,4,5,6,7,8])
        ax.set_xticklabels([1,2,3,4,5,6,7,8])
    axes_hm[1].set_ylabel("")
    axes_hm[1].set_yticklabels([])
    # axes_hm[2].set_ylabel("")
    # axes_hm[2].set_yticklabels([])
    axes_hm[0].set_title("Suppression duration\n(months)", fontsize=8)
    axes_hm[1].set_title("Proportion of infections\nfrom under 18s", fontsize=8)
    # axes_hm[2].set_title("Proportion of infections\nfrom over 65s", fontsize=8)

    colorbar_axes = [c for c in fig.axes if c not in [axes_hm[0], axes_hm[1], fit_ax]]
    if len(colorbar_axes) >= 2:
        colorbar_axes[0].set_ylabel("")
        colorbar_axes[1].set_ylabel("")
        # colorbar_axes[2].set_ylabel("")

    r0_values_by_pathogen = {}
    immunity_values_by_pathogen = {}
    for pathogen, option2, seed, color in zip(pathogens, option2s, seeds, colors):
        print(f"plotting parameter scatter for {pathogen}...")
        r0_values, immunity_values = plot_r0_vs_first_immunity(fit_ax, pathogen, option2, seed, color, prune=prune)
        r0_values_by_pathogen[pathogen] = r0_values
        immunity_values_by_pathogen[pathogen] = immunity_values
    if fit_line:
        plot_odr_best_fit(fit_ax, r0_values_by_pathogen, immunity_values_by_pathogen, n_iterations=10000)
    else:
        x = np.linspace(1, 10, 100)
        y = 1 - 1 / x
        fit_ax.plot(x, y, color='k', linewidth=0.5, zorder=0)
        # label with "1 - 1/R0" in the middle of the line
        fit_ax.text(3.5, 1-1/3.5, r"$1 - \frac{1}{R_0}$", rotation=0, fontsize=11, color='k', ha='right', va='bottom', zorder=1)
    # build custom legend with colored squares
    handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=8) for color in colors]
    labels = [nice_names.get(pathogen, pathogen) for pathogen in pathogens]
    fit_ax.legend(handles, labels, frameon=False, fontsize=6, loc='lower right')
    fit_ax.set_xscale('log')
    fit_ax.set_xticks([1,2,3,4,5,6,7,8,9,10])
    fit_ax.set_xticklabels([1,2,3,4,5,6,7,8,9,10])
    fit_ax.set_xlim(1, 10)
    fit_ax.set_ylim(0, 0.95)
    fit_ax.set_xlabel("Basic reproduction number (log scale)")
    fit_ax.set_ylabel("Immunity from first infection")

    # add big letter "A" to first row, "B" to second row
    axes_hm[0].text(-0.32, 1.1, "B", transform=axes_hm[0].transAxes, fontsize=16, fontweight="bold")
    fit_ax.text(-0.12, 1.05, "A", transform=fit_ax.transAxes, fontsize=16, fontweight="bold")

## MCMC comparison functions
def calculate_precision_ratio(chain1, chain2, param_names1, param_names2, select_param_names):
    idxs1 = [param_names1.index(name) for name in select_param_names]
    idxs2 = [param_names2.index(name) for name in select_param_names]
    var1 = np.var(chain1[:, idxs1], axis=0)
    var2 = np.var(chain2[:, idxs2], axis=0)
    return var1 / var2

def calculate_log_det_ratio(chain1, chain2, param_names1, param_names2, select_param_names):
    idxs1 = [param_names1.index(name) for name in select_param_names]
    idxs2 = [param_names2.index(name) for name in select_param_names]
    cov1 = np.cov(chain1[:, idxs1].T)
    cov2 = np.cov(chain2[:, idxs2].T)
    log_det1 = np.linalg.slogdet(cov1)[1]
    log_det2 = np.linalg.slogdet(cov2)[1]
    return log_det1 - log_det2

# find kl divergence using k nearest neighbors method
def calculate_kl_divergence(chain1, chain2, param_names1, param_names2, select_param_names, k=5):
    from sklearn.neighbors import NearestNeighbors
    idxs1 = [param_names1.index(name) for name in select_param_names]
    idxs2 = [param_names2.index(name) for name in select_param_names]
    data1 = chain1[:, idxs1]
    data2 = chain2[:, idxs2]
    n1, d = data1.shape
    n2, _ = data2.shape
    # Fit nearest neighbors on both datasets
    nn1 = NearestNeighbors(n_neighbors=k+1).fit(data1)
    nn2 = NearestNeighbors(n_neighbors=k).fit(data2)
    # Find distances to k nearest neighbors in both datasets
    dist1, _ = nn1.kneighbors(data1)
    dist2, _ = nn2.kneighbors(data1)
    # Exclude the point itself in the distance calculation for data1
    dist1 = dist1[:, 1:]
    # Calculate KL divergence using the formula from Perez-Cruz (2008)
    kl_div = (d / n1) * np.sum(np.log(dist2[:, -1] / dist1[:, -1])) + np.log(n2 / (n1 - 1))
    return kl_div

def compare_mcmc_chains(chain1, chain2, param_names1, param_names2, select_param_names):
    precision_ratio = calculate_precision_ratio(chain1, chain2, param_names1, param_names2, select_param_names)
    log_det_ratio = calculate_log_det_ratio(chain1, chain2, param_names1, param_names2, select_param_names)
    kl_divergence = calculate_kl_divergence(chain1, chain2, param_names1, param_names2, select_param_names)
    print(f"Precision ratios: {np.array2string(precision_ratio, formatter={'float_kind': lambda x: f'{x:.2g}'})}")
    print(f"Log determinant ratio: {log_det_ratio:.4f}")
    print(f"KL divergence: {kl_divergence:.4f}")

import seaborn as sns
def plot_correlation_matrix_difference(ax, chain1, chain2, param_names1, param_names2, select_param_names):
    idxs1 = [param_names1.index(name) for name in select_param_names]
    idxs2 = [param_names2.index(name) for name in select_param_names]
    data1 = chain1[:, idxs1]
    data2 = chain2[:, idxs2]
    corr1 = np.corrcoef(data1, rowvar=False)
    corr2 = np.corrcoef(data2, rowvar=False)
    corr_diff = np.sqrt(corr2**2) - np.sqrt(corr1**2)
    std1 = np.std(data1, axis=0)
    std2 = np.std(data2, axis=0)
    std_diff = (std2 - std1)/std1
    # set corr_diff diagonal to std_diff
    np.fill_diagonal(corr_diff, std_diff)
    # set upper triangle to NA
    corr_diff[np.triu_indices_from(corr_diff, k=1)] = np.nan
    sns.heatmap(corr_diff, xticklabels=select_param_names, yticklabels=select_param_names, center=0, cmap="bwr", ax=ax, vmin=-1.05, vmax=1.05)
    # for diagonal elements that are greater than 1, add text to their right with the value of the std_diff
    for i in range(len(select_param_names)):
        if std_diff[i] > 1.05 or std_diff[i] < -1.05:
            ax.text(i+1, i+1, f"{std_diff[i]:.2f}", color="black", ha="left", va="bottom", fontsize=6)
    # print the change in the frobenius norm of the correlation matrix
    # set diagonal to 0 for this calculation
    np.fill_diagonal(corr1, 0)
    np.fill_diagonal(corr2, 0)
    frob_norm1 = np.linalg.norm(corr1, 'fro')
    frob_norm2 = np.linalg.norm(corr2, 'fro')
    print(f"Frobenius norms: {frob_norm1:.4f}, {frob_norm2:.4f}")
    print(f"Proportional change in Frobenius norm: {(frob_norm2 - frob_norm1) / frob_norm1:.4f}")

from matplotlib.collections import LineCollection
def plot_immunity_cascade(axes, n_samples=100, save_data=False, kpsc_incidence=True):
    aggregation = "MS"
    agg_factor = 30.44
    factor = 100000
    for pathogen_idx in range(len(pathogens)):
        print(f"Plotting immunity cascade for {pathogens[pathogen_idx]}...")
        ax = axes[pathogen_idx, :]
        pathogen, option2, seed = pathogens[pathogen_idx], option2s[pathogen_idx], seeds[pathogen_idx]
        _, _, _, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=1e-9, incidence_data=False, hosp=True, NAG=NAG, dedup=True)
        chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=0, prefix="")
        np.random.seed(260604)
        # choose n_samples random rows from chain
        random_indices = np.random.choice(chain.shape[0], size=n_samples, replace=False)
        random_samples = chain[random_indices, :]
        from fit_MCMC import run_simulation
        from utils import susceptibility
        t_key = date_to_t('2020-01-01')
        def run_sims(x):
            params = x_to_params(x, pathogen, lockdown, option1, option2, NAG=NAG)
            solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)

            values = solution.ys.T
            trajectory = jnp.diff(values[-NAG:, :], axis=1).T
            p_time_to_obs_flipped = jnp.flip(p_time_to_obs.flatten())
            def obs_convolution(x):
                return jnp.convolve(x, p_time_to_obs_flipped, mode='same')
            obs = jax.vmap(obs_convolution, in_axes=1, out_axes=1)(trajectory)
            obs = jax.nn.softplus(obs*100)/100
            pop_size_by_age = calculate_population_size(values, NAG=NAG)
            obs = factor*obs/pop_size_by_age[1:,:]
            avg_pre2020_expected_obs = jnp.mean(obs[POINTS[1:] < t_key], axis=0)
            
            shaped_solution = solution.ys.T[1:, :].reshape((2*N_S+1, NAG, -1))
            susceptible = shaped_solution[0:2*N_S:2, :, :]

            D_REL = params[8]
            avg_drel = jnp.sum(D_REL[:, None, None] * susceptible, axis=0) / jnp.sum(susceptible, axis=0)
            avg_pre2020_drel = jnp.mean(avg_drel[:, POINTS < t_key], axis=1)

            S_REL = params[6]
            avg_srel = jnp.sum(S_REL[:, None, None] * susceptible, axis=0) / jnp.sum(susceptible, axis=0)
            avg_pre2020_srel = jnp.mean(avg_srel[:, POINTS < t_key], axis=1)

            OBS_AGE = params[9]
            scaled_obs_age = OBS_AGE/jnp.max(OBS_AGE)

            multiplier_stack = jnp.stack([avg_pre2020_expected_obs, 1/scaled_obs_age, 1/(scaled_obs_age*avg_pre2020_drel), 1/(scaled_obs_age*avg_pre2020_drel*avg_pre2020_srel)])
            return multiplier_stack
        multiplier_stacks = jax.jit(jax.vmap(run_sims))(random_samples)
        if kpsc_incidence:
            incidence = calculate_proportion_positive_incidence(pathogen, aggregation=aggregation, window_size=1, weighting_factor=0, sum_age_groups=False, save_counts=False, pp_only=False, hosp=True, NAG=NAG, detrend=False, dedup=True, sac=True)
            avg_pre2020_incidence = incidence.iloc[incidence.index < pd.to_datetime('2020-01-01')].mean(axis=0).values
            avg_pre2020_incidence *= factor
        else:
            avg_pre2020_incidence = multiplier_stacks[:, 0, :].T
        ymax = 0
        ymin = 0
        for age_group_idx, age_group_name in enumerate(AGE_GROUP_NAMES):
            xs = jnp.array([4, 2.5, 1.75, 1])
            x_values = jnp.repeat(xs[None, :], n_samples, axis=0)
            cascade_multipliers = multiplier_stacks[:, :, age_group_idx]
            if kpsc_incidence:
                cascade = cascade_multipliers.at[:, 1:].set(cascade_multipliers[:, 1:] * avg_pre2020_incidence[age_group_idx])
                cascade = cascade.at[:, 0].set(avg_pre2020_incidence[age_group_idx])
            else:
                cascade = cascade_multipliers.at[:, 1:].set(cascade_multipliers[:, 1:] * cascade_multipliers[:, 0:1])
            cascade = jnp.log(cascade)
            segments = jnp.stack([x_values, cascade], axis=-1)
            lc = LineCollection(segments, alpha=0.01, linewidths=0.5, color=hsv_colors[age_group_idx], zorder=0)
            # plot median and iqr at each stage of the cascade
            median_cascade = jnp.median(cascade, axis=0)
            q1_cascade = jnp.percentile(cascade, 25, axis=0)
            q3_cascade = jnp.percentile(cascade, 75, axis=0)
            ax[age_group_idx].scatter(xs, median_cascade, color="k", label="Median", s=4, marker="_", zorder=2)
            for i in range(4):
                ax[age_group_idx].plot([xs[i], xs[i]], [q1_cascade[i], q3_cascade[i]], color="k", linewidth=1, zorder=2)
            ax[age_group_idx].add_collection(lc)
            ax[age_group_idx].set_xticks(xs)
            ax[age_group_idx].set_xticklabels([])
            ymax = max(ymax, jnp.percentile(cascade, 90))
            ymin = min(ymin, jnp.percentile(cascade, 10))
            if age_group_idx == 0:
                ax[age_group_idx].set_ylabel(short_names.get(pathogen, pathogen))
            else:
                ax[age_group_idx].set_ylabel("")
                ax[age_group_idx].spines['left'].set_visible(False)
                ax[age_group_idx].set_yticks([])
            if pathogen_idx == 0:
                ax[age_group_idx].set_title(age_group_name)
            ax[age_group_idx].spines['top'].set_visible(False)
            ax[age_group_idx].spines['right'].set_visible(False)
            ax[age_group_idx].set_xlim(0.5, 4.5)
        for age_group_idx in range(len(AGE_GROUP_NAMES)):
            ax[age_group_idx].set_ylim(ymin*1.1, ymax*1.1)
        ax[0].set_yticks([np.log(1), np.log(10), np.log(100)])
        ax[0].set_yticklabels([1, 10, 100])

def plot_data_totals(axes, aggregation="MS"):
    # load hospitalization counts
    daily_hospitalization_counts = pd.read_csv(f'Data/Processed/KPSC_ARI_nonCOVID_hospitalizations_by_day_age_group_sac_dedup.csv', index_col=0, parse_dates=True)
    # load test counts
    daily_test_counts = pd.read_csv(f'Data/Processed/KPSC_ARI_hospitalized_pathogen_panel_test_counts_by_hosp_day_pathogen_age_group_sac_dedup.csv', index_col=0, parse_dates=True)
    # filter test counts to just one pathogen (RSV) and just total counts
    daily_test_counts = daily_test_counts[daily_test_counts['pathogen'] == 'RSV']
    daily_test_counts = daily_test_counts[daily_test_counts['result_type'] == 'Total']
    num_groups = len(AGE_GROUP_NAMES)
    for i, age_group in enumerate(AGE_GROUP_NAMES):
        age_group_data = daily_test_counts[daily_test_counts['age_group'] == age_group]
        age_group_data = age_group_data.sort_index()
        total_tests = age_group_data.groupby('Hospitalization date')['count'].sum()
        hosp_counts = daily_hospitalization_counts[age_group]
        hosp_counts = hosp_counts[hosp_counts.index < pd.Timestamp('2025-05-01')]
        # Aggregate to monthly data
        hosp_agg = hosp_counts.resample(aggregation).sum()
        test_agg = total_tests.resample(aggregation).sum()
        # Create mask for missing data period
        gap_start = pd.Timestamp('2024-05-01')
        gap_end = pd.Timestamp('2024-10-01')
        axes[i // 2, i % 2].plot(hosp_agg.index[(hosp_agg.index < gap_start)], hosp_agg.values[(hosp_agg.index < gap_start)], color='silver', label='Hospitalizations', linewidth=1.5)
        axes[i // 2, i % 2].plot(test_agg.index[(test_agg.index < gap_start)], test_agg.values[(test_agg.index < gap_start)], color='k', label='Panel tests', linewidth=1.5)
        axes[i // 2, i % 2].plot(hosp_agg.index[(hosp_agg.index >= gap_end)], hosp_agg.values[(hosp_agg.index >= gap_end)], color='silver', label='Hospitalizations', linewidth=1.5)
        axes[i // 2, i % 2].plot(test_agg.index[(test_agg.index >= gap_end)], test_agg.values[(test_agg.index >= gap_end)], color='k', label='Panel tests', linewidth=1.5)
        axes[i // 2, i % 2].set_title(age_group)
        if i < 5:
            axes[i // 2, i % 2].set_xticklabels([])
            axes[i // 2, i % 2].set_xlabel("")
        else:
            # set x-axis tick lable font size to 6
            axes[i // 2, i % 2].tick_params(axis='x', labelsize=6)
            axes[i // 2, i % 2].set_xlabel("Date")
        if i % 2 == 0:
            axes[i // 2, i % 2].set_ylabel("Counts")
    axes[-1, -1].axis("off")
    legend_handles = [
                      plt.Line2D([0], [0], color='silver', label='ARI hospitalizations', linewidth=1.5),
                      plt.Line2D([0], [0], color='k', label='Panel tests', linewidth=1.5),]
    axes[-1, -1].legend(handles=legend_handles, fontsize=6, loc='center', frameon=False)

import pandas as pd
import numpy as np
import jax
from utils import consistent_x_from_DE

def print_parameter_table(pathogens, option2s, seeds, prune=0):
    param_table = pd.DataFrame(columns=["Pathogen", "Value"])
    
    if lockdown != "Default":
        base_parameter_names = ["b", "A", "t", "f", "r", "w", "s2", "s3/s2", "ph2", "ph3/ph2", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"]
        parameter_indices = [2, 3, 4, 12, 13, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20]
    else:
        base_parameter_names = ["b", "A", "t", "w", "s2", "s3/s2", "ph2", "ph3/ph2", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"]
        parameter_indices = [2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18]
    
    # Position 'h' right before the 'ps' parameters
    ps1_idx = base_parameter_names.index("ps1")
    parameter_names = base_parameter_names[:ps1_idx] + ["h"] + base_parameter_names[ps1_idx:]
    
    print(len(parameter_names), len(parameter_names))
    
    for pathogen, option2, seed in zip(pathogens, option2s, seeds):
        chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=prune, prefix="")
        consistent_xs = jax.vmap(lambda x: consistent_x_from_DE(pathogen, lockdown, option1, option2, seed, NAG=NAG, x_DE=x))(chain)
        print(consistent_xs.shape)
        
        # Extract base parameters from the chain
        consistent_xs = np.array(consistent_xs)[:, parameter_indices]
        
        # Identify column positions of all 'ps' parameters
        ps_indices = [i for i, name in enumerate(base_parameter_names) if name.startswith("ps")]
        
        # Calculate 'h' (maximum value across all samples and all ps parameters for this pathogen)
        h_val = np.max(consistent_xs[:, ps_indices])
        
        # Rescale the ps parameters as a fraction of h
        consistent_xs[:, ps_indices] = consistent_xs[:, ps_indices] / h_val
        
        # Create a constant column for h and inject it at the correct position
        h_column = np.full((consistent_xs.shape[0], 1), h_val)
        consistent_xs = np.hstack([consistent_xs[:, :ps1_idx], h_column, consistent_xs[:, ps1_idx:]])
        
        param_medians = np.median(consistent_xs, axis=0)
        lower_bounds = np.percentile(consistent_xs, 2.5, axis=0)
        upper_bounds = np.percentile(consistent_xs, 97.5, axis=0)
        
        values = [f"{m:.4f}" if p2_5 == p97_5 else f"{m:.4f} ({p2_5:.4f}—{p97_5:.4f})" for m, p2_5, p97_5 in zip(param_medians, lower_bounds, upper_bounds)]
        
        new_param_table = pd.DataFrame({
            "Pathogen": pathogen,
            "Value": values,
            "Parameter": parameter_names,
        })
        param_table = pd.concat([param_table, new_param_table], ignore_index=True)
        
    # pivot with pathogens as columns
    param_table = param_table.pivot(index="Parameter", columns="Pathogen", values="Value")
    param_table = param_table.reindex(parameter_names)
    param_table = param_table[pathogens]
    
    # print table in comma-separated format
    print(param_table.to_csv(sep=",", index=False))

# plot trajectory of infecteds and effective susceptibles over time from one mcmc sample for one pathogen
def plot_phase_diagram(ax, x, pathogen, option2, age_group_idx=None, end_date='2025-05-01', prune=0, no_lockdown=False, proportion=False):
    from fit_MCMC import run_simulation
    lockd = lockdown + ("_no_lockdown" if no_lockdown else "")
    params = x_to_params(x, pathogen, lockd, option1, option2, NAG=NAG, end_date=end_date)
    solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
    maternal = solution.ys[:,0]
    values = solution.ys.T[1:].reshape((2*N_S+1, NAG, -1))
    if age_group_idx is not None:
        age_pop = jnp.sum(values[:,age_group_idx,:], axis=0) + maternal * (age_group_idx == 0)
        infecteds = jnp.sum(values[1:2*N_S:2,age_group_idx,:], axis=0)/age_pop
        susceptible = values[0:2*N_S:2,age_group_idx,:]
        effective_susceptible = jnp.sum(susceptible * params[6][:,None], axis=0)/age_pop
        mult_s, mult_i = 1e2, 1e2
    elif proportion:
        infecteds = jnp.sum(values[1:2*N_S:2,:,:], axis=(0,1))
        susceptible = jnp.sum(values[0:2*N_S:2,:,:], axis=1)
        effective_susceptible = jnp.sum(susceptible * params[6][:,None], axis=0)
        total_population = jnp.sum(susceptible, axis=0) + infecteds
        infecteds /= total_population
        infecteds = jnp.log(infecteds)
        effective_susceptible /= total_population
        mult_s, mult_i = 1, 1
    else:
        infecteds = jnp.sum(values[1:2*N_S:2,:,:], axis=(0,1))
        susceptible = jnp.sum(values[0:2*N_S:2,:,:], axis=1)
        effective_susceptible = jnp.sum(susceptible * params[6][:,None], axis=0)
        mult_s, mult_i = 1e-6, 1e-3
    if ax is not None:
        ax.plot(effective_susceptible[:1719]*mult_s, infecteds[:1719]*mult_i, color='k')
        if no_lockdown:
            ax.plot(effective_susceptible[1719:]*mult_s, infecteds[1719:]*mult_i, color='silver', linewidth=1, linestyle='--')
        else:
            # Plot with color gradient from light magenta to dark magenta using LineCollection
            x_data = effective_susceptible[1719:]*mult_s
            y_data = infecteds[1719:]*mult_i
            points = np.array([x_data, y_data]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='plasma_r', linewidth=1)
            lc.set_array(np.arange(len(x_data)))
            ax.add_collection(lc)
            ax.autoscale_view()
        if age_group_idx is not None:
            ax.set_xlabel("Proportion effective susceptibles")
            ax.set_ylabel("Proportion infecteds")
            ax.set_title(f"{short_names.get(pathogen, pathogen)} - {AGE_GROUP_NAMES[age_group_idx]}")
        else:
            ax.set_xlabel("Effective susceptibles (millions)")
            ax.set_ylabel("Infecteds (thousands)")
            ax.set_title(short_names.get(pathogen, pathogen))
    return effective_susceptible, infecteds

# plot phase diagrams using samples from chain for each pathogen in passed list
def plot_phase_diagrams_from_chains(axes, pathogens, option2s, seeds, lockdown, option1, end_date='2025-05-01', n_samples=1, prune=0, no_lockdown=False):
    for pathogen, option2, seed, ax in zip(pathogens, option2s, seeds, axes):
        chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=prune, prefix="")
        # sample n_samples evenly spaced indices from the chain
        indices = np.linspace(0, len(chain)-1, n_samples, dtype=int)
        for idx in indices:
            print(f"Plotting phase diagram for {pathogen} sample {idx}/{n_samples}...", end="\r")
            if n_samples > 1:
                x = chain[idx, :]
            else:
                x = np.median(chain, axis=0)
            plot_phase_diagram(ax, x, pathogen, option2, end_date=end_date, prune=prune, no_lockdown=no_lockdown)
        print(f"Finished plotting phase diagrams for {pathogen}.")

def plot_phase_diagrams_by_age(axes, pathogen, option2, seed, lockdown, option1, end_date='2025-05-01', n_samples=1, prune=0, no_lockdown=False):
    chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=prune, prefix="")
    # sample n_samples evenly spaced indices from the chain
    indices = np.linspace(0, len(chain)-1, n_samples, dtype=int)
    for age_group_idx in range(len(AGE_GROUP_NAMES)):
        ax = axes[age_group_idx]
        for idx in indices:
            print(f"Plotting phase diagram for {pathogen} age group {AGE_GROUP_NAMES[age_group_idx]} sample {idx}/{n_samples}...", end="\r")
            if n_samples > 1:
                x = chain[idx, :]
            else:
                x = np.median(chain, axis=0)
            plot_phase_diagram(ax, x, pathogen, option2, age_group_idx=age_group_idx, end_date=end_date, prune=prune, no_lockdown=no_lockdown)
        print(f"Finished plotting phase diagrams for {pathogen} age group {AGE_GROUP_NAMES[age_group_idx]}.")

def arrange_phase_plots(fig, axes, pathogens, option2s, seeds, lockdown, option1):
    for idx, (pathogen, option2, seed) in enumerate(zip(pathogens, option2s, seeds)):
        plot_phase_diagrams_by_age(axes[:,idx], pathogen, option2, seed, lockdown, option1, n_samples=1, prune=0)
    
    # Suppress interior axis labels
    for ax in axes.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        # set tick label size to 6
        ax.tick_params(axis='both', which='major', labelsize=6)
    
    # leftmost column has extra y-axis labels for age group
    for i in range(7):
        axes[i,0].set_ylabel(AGE_GROUP_NAMES[i], fontsize=8)
    # top row has titles for each pathogen
    for j in range(6):
        axes[0,j].set_title(short_names.get(pathogens[j], pathogens[j]), fontsize=8)

    # overall x title at bottom
    fig.text(0.5, 0.0, "Effective susceptibles (%)", ha='center', va='bottom', fontsize=9)
    # overall y title at left
    fig.text(0.0, 0.5, "Infecteds (%)", va='center', ha='left', rotation='vertical', fontsize=9)

# run plot_phase_diagram with no_lockdown=True and no_lockdown=False and plot the euclidean distance between the two trajectories over time for each pathogen in the passed list
def plot_phase_diagram_distance(axes, pathogens, option2s, seeds, lockdown, option1, end_date='2025-05-01', n_samples=1, prune=0):
    for pathogen, option2, seed, ax in zip(pathogens, option2s, seeds, axes):
        chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=prune, prefix="")
        # randomly sample n_samples indices from the chain
        indices = np.random.choice(len(chain), n_samples, replace=False)
        if n_samples > 1:
            xs = chain[indices, :]
        else:
            xs = np.tile(np.median(chain, axis=0), (1, 1))
        
        def compute_distance(x):
            eff_sus_no_lockdown, infecteds_no_lockdown = plot_phase_diagram(None, x, pathogen, option2, end_date=end_date, prune=prune, proportion=True, no_lockdown=True)
            eff_sus_lockdown, infecteds_lockdown = plot_phase_diagram(None, x, pathogen, option2, end_date=end_date, prune=prune, proportion=True, no_lockdown=False)
            distance = jnp.sqrt(((eff_sus_no_lockdown - eff_sus_lockdown)/jnp.std(eff_sus_no_lockdown))**2 + ((infecteds_no_lockdown - infecteds_lockdown)/jnp.std(infecteds_no_lockdown))**2)
            return distance

        # vmap the calculation of eff_sus and infecteds for all samples
        compute_distances = jax.jit(jax.vmap(compute_distance))
        distances = compute_distances(xs)
        dates = [t_to_date(t) for t in POINTS]

        # fit an exponential decay to the distances from 2021-01-01, plot it, and print the half-life of the decay
        from scipy.optimize import curve_fit
        def exp_decay(x, a, b):
            return a * np.exp(-b * x)
        start_idx = np.where(np.array(dates) >= pd.Timestamp('2021-01-01'))[0][0]
        
        # vmap curve_fit over each sampled distance array to estimate uncertainty
        def fit_decay(dist_array):
            popt, _ = curve_fit(exp_decay, np.arange(len(dates[start_idx:])), dist_array[start_idx:], p0=(1, 0.01))
            return popt
        
        popts = np.array([fit_decay(distances[i, :]) for i in range(len(distances))])
        popt = np.median(popts, axis=0)
        half_life = np.log(2) / popt[1]
        half_life_std = np.std(np.log(2) / popts[:, 1])
        half_life_ci_lower = np.percentile(np.log(2) / popts[:, 1], 2.5)
        half_life_ci_upper = np.percentile(np.log(2) / popts[:, 1], 97.5)
        print(f"Half-life of phase diagram distance decay for {pathogen}: {half_life:.2f} ({half_life_ci_lower:.2f} to {half_life_ci_upper:.2f}) days")
        print(f"Resilience for {pathogen}: {365/half_life:.4f} ({365/half_life_ci_upper:.4f} to {365/half_life_ci_lower:.4f}) per year")
        
        # Convert dates to numeric values for LineCollection
        dates_numeric = np.array([d.toordinal() for d in dates])
        
        # plot using LineCollection
        segments = [list(zip(dates_numeric[1719:], distances[i, 1719:])) for i in range(len(distances))]
        lc = LineCollection(segments, colors='k', linewidths=0.5, alpha=0.01)
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_xlabel("Date")
        ax.set_ylabel("Euclidean distance")

        # plot fitted exponential decay curve with uncertainty interval
        x_fit = np.arange(len(dates[start_idx:]))
        y_fit = exp_decay(x_fit, *popt)
        
        # compute uncertainty bounds from sampled popts
        y_samples = np.array([exp_decay(x_fit, *p) for p in popts])
        y_lower = np.percentile(y_samples, 2.5, axis=0)
        y_upper = np.percentile(y_samples, 97.5, axis=0)
        
        # plot uncertainty interval behind line collection
        ax.fill_between(dates_numeric[start_idx:], y_lower, y_upper, color='r', alpha=0.2, zorder=0)
        ax.plot(dates_numeric[start_idx:], y_fit, color='r', label=f"Fitted exp decay (half-life: {half_life:.0f} ({half_life_ci_lower:.0f} to {half_life_ci_upper:.0f}) days)", zorder=0)
        ax.legend(fontsize=6)
        
        # Format x-axis with year ticks
        year_ticks = np.array([pd.Timestamp(year=y, month=1, day=1).toordinal() for y in range(2020, 2026)])
        ax.set_xticks(year_ticks)
        ax.set_xticklabels([str(y) for y in range(2020, 2026)], rotation=45)
        
        ax.set_title(short_names.get(pathogen, pathogen))
        print(f"Finished plotting phase diagram distances for {pathogen}.")

if __name__ == "__main__":
    plt.rcParams.update({'font.size':8})
    # text type is palatino
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica']

    seed = 260531
    option1 = "dedupsac"
    lockdown = "ExponentialODipLinear"

    NAG = 7 + ("split" in option1)
    if "split" in option1:
        from Parameters.census_population import CENSUS_AGE_POP_split as CENSUS_AGE_POP, AGE_GROUP_NAMES_split as AGE_GROUP_NAMES, AGE_GROUPS_split as AGE_GROUPS
    elif "sac" in option1:
        from Parameters.census_population import CENSUS_AGE_POP_sac as CENSUS_AGE_POP, AGE_GROUP_NAMES_sac as AGE_GROUP_NAMES, AGE_GROUPS_sac as AGE_GROUPS
    else:
        from Parameters.census_population import CENSUS_AGE_POP, AGE_GROUP_NAMES
    CONTACT_MATRIX = jnp.asarray(pd.read_csv('Data/Processed/contact_matrices/KP'+['', '_split'][NAG>7]+['', '_sac']["sac" in option1]+'_contact_all_US_Census.csv', delimiter=',', header=None).values)
    r0_base = calculate_R0_from_values(1, 1, CONTACT_MATRIX, CENSUS_AGE_POP, jnp.zeros(NAG))

    PERIOD = pd.date_range(start=pd.to_datetime('2015-07-04'), end=pd.to_datetime('2025-05-01'), freq='D')
    POINTS = np.array(date_to_t(PERIOD))
    start_idx = int(date_to_t(PERIOD[0]) + 90 - date_to_t('2015-10-01'))
    end_idx = int(date_to_t(PERIOD[-1]) - date_to_t('2015-10-01'))
    ## Initial conditions
    STATE0 = jnp.zeros((2*N_S+1,NAG))
    STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
    STATE0 = STATE0.at[1,:].set(1)
    # # flatten initial state and add maternal immunity compartment
    STATE0 = STATE0.flatten()
    STATE0 = jnp.concatenate((jnp.array([0]), STATE0))
    daily_hospitalization_rates_pd = pd.read_csv('Data/Processed/KPSC_ARI_nonCOVID_hospitalization_rates_by_day_age_group'+['','_split'][NAG>7]+['','_detrended']["detrend" in option1]+["","_dedup"]["dedup" in option1]+'.csv',index_col=0,parse_dates=True)
    daily_hospitalization_rates_full = jnp.asarray(daily_hospitalization_rates_pd.values)
    daily_hospitalization_rates = daily_hospitalization_rates_full[start_idx:end_idx,]

    lockdown = "ExponentialODipp25"
    pathogens = ["RSV","Metapneumovirus","Parainfluenza3","Adenovirus","InfluenzaA","InfluenzaB",]
    flunet_pathogens = ["RSV","Metapneumovirus","Parainfluenza","Adenovirus","InfluenzaA","InfluenzaB",]
    colors = [ "#785EF0", "#004D40", "#648FFF", "#FFB000" ,"#FF832B","#DC267F",]
    option2s = ["maxagep028","fixage0maxagep006","maxagep004","maxagep003","maxagep035","maxagep035",]
    seeds = [260612, 260622, 260612, 260612, 260612, 260612,]

    # print_parameter_table(pathogens, option2s, seeds, prune=0)

    # fig, axes = plt.subplots(3, 2, figsize=(6.5, 6.5), layout="constrained", sharex=False, sharey=False)
    # plot_phase_diagrams_from_chains(axes.flatten(), pathogens, option2s, seeds, lockdown, option1, n_samples=1, prune=0, no_lockdown=True)
    # plot_phase_diagrams_from_chains(axes.flatten(), pathogens, option2s, seeds, lockdown, option1, n_samples=1, prune=0, no_lockdown=False)
    # plt.savefig(f"Figures/phase_diagrams_w_no_lockdown.png", dpi=300)

    # fig, axes = plt.subplots(3, 2, figsize=(6.5, 6.5), layout="constrained", sharex=False, sharey=False)
    # plot_phase_diagram_distance(axes.flatten(), pathogens, option2s, seeds, lockdown, option1, n_samples=400, prune=100)
    # plt.savefig(f"Figures/phase_diagram_proportion_distance_w_exp_fit_logi_normed.png", dpi=300)

    # fig, axes = plt.subplots(7, 6, figsize=(6.5, 6.5), sharex=False, sharey=False)
    # arrange_phase_plots(fig, axes, pathogens, option2s, seeds, lockdown, option1)
    # plt.tight_layout()
    # plt.savefig(f"Figures/phase_diagrams_{lockdown}_age.pdf", dpi=300)

    # fig, axes = plt.subplots(3, 2, figsize=(6.5, 8), layout="constrained", sharex=False, sharey=False)
    # for pathogen, option2, seed, ax in zip(pathogens, option2s, seeds, axes.flatten()):
    #     print(f"Difference measures for {pathogen}...")
    #     chain2 = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=0, prefix="")
    #     chain1 = load_mcmc_chain(pathogen, [260615, 260624][pathogen=="Metapneumovirus"], ["Default", "ExponentialODipp25"][pathogen=="Metapneumovirus"], option1, "2020-01-01"+option2, just_chain=True, prune=0, prefix="")
    #     param_names1, _ = parameters_names_bounds(pathogen, "Default", "dedupsac", "2020-01-01"+option2, NAG=NAG)
    #     param_names2, _ = parameters_names_bounds(pathogen, lockdown, "dedupsac", option2, NAG=NAG)
    #     select_param_names = [param_name for param_name in param_names1 if param_name in param_names2]
    #     compare_mcmc_chains(chain1, chain2, param_names1, param_names2, select_param_names)
    #     plot_correlation_matrix_difference(ax, chain1, chain2, param_names1, param_names2, select_param_names)
    #     ax.set_title(short_names.get(pathogen, pathogen))
    # plt.savefig(f"Figures/correlation_matrix_difference_{lockdown}_updated.png", dpi=300)

    # fig, axes = plt.subplots(6,7, figsize=(4, 5), layout="constrained", sharex=False, sharey=False)
    # plot_immunity_cascade(axes, n_samples=400, save_data=True, kpsc_incidence=True)
    # for ax in axes[-1,:]:
    #     ax.set_xticklabels(["No imm.", "+ I. exp.", "+ D. exp.", "+ D. age"][::-1], fontsize=6, rotation=90, ha='center')
    # plt.savefig(f"Figures/immunity_cascade_{lockdown}_log_twopart.png", dpi=300)

    ### Figure 1 with diagram
    # fig, axes = plt.subplots(10, 6, figsize=(6.5, 7), layout="constrained", sharex=False, sharey=False,
    #                          gridspec_kw={'height_ratios': [2.7, 1, 1, 1, 1, 1, 1, 1, 1, 1]})

    # import fitz
    # import io
    # from PIL import Image

    # compartment_diagram_path = "Figures/condensed_compartmental_diagram-2.pdf"
    # doc = fitz.open(compartment_diagram_path)
    # page = doc[0]

    # pix = page.get_pixmap(matrix=fitz.Matrix(5, 5))
    # img_data = pix.tobytes("png")
    # img = Image.open(io.BytesIO(img_data))
    # # 1. Remove the original 6 individual axes in the first row
    # for ax in axes[0, :]:
    #     ax.remove()
        
    # # 2. Get the underlying GridSpec from another axis (e.g., row 1)
    # gs = axes[1, 0].get_gridspec()
    
    # # 3. Create a new single axis that spans row 0, all columns (:)
    # ax_top = fig.add_subplot(gs[0, :])
    
    # # 4. Show the image and turn off the axis on this new spanning subplot
    # ax_top.imshow(img, interpolation="lanczos")
    # ax_top.axis("off")

    # sample_size = 400

    # plot_fits(axes[1:8,:], n_samples=sample_size, load_data=False, save_data=True)
    # for ax in axes[7, :]:
    #     ax.set_xticklabels([])

    # # fig, axes = plt.subplots(1, 6, figsize=(6.5, 1.5), sharex=True, sharey=True)
    # for pathogen, option2, seed, ax in zip(pathogens, option2s, seeds, axes[8, :]):
    #     plot_contact(ax, pathogen, option2, seed, n_samples=sample_size)
    #     # ax.set_title(short_names.get(pathogen, pathogen))
    #     ax.set_ylim(0, 1.2)
    #     ax.set_xticks(pd.to_datetime(["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01",]))
    #     ax.set_xticklabels([])
    #     ax.tick_params(axis='y', labelsize=6)
    #     # ax.set_xticklabels(["", "2017", "", "2019", "", "2021", "", "2023", "", "2025",], fontsize=6)
    #     # for label in ax.get_xticklabels():
    #     #     label.set_rotation(45)
    #     #     label.set_horizontalalignment('right')
    #     #     label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
    # axes[8,0].set_ylabel("Contact\nmultiplier", fontsize=6)

    # import time
    # # fig, axes = plt.subplots(1, 6, figsize=(6.5,1.5))
    # for pathogen, option2, seed, ax in zip(pathogens, option2s, seeds, axes[9,:]):
    #     print(pathogen)
    #     start_time = time.time()
    #     plot_susceptibility(ax, pathogen, option2, seed, n_samples=sample_size)
    #     print(f"Time taken to plot susceptibility for {pathogen}: {time.time() - start_time:.2f} seconds")
    #     # ax.set_title(short_names.get(pathogen, pathogen))
    #     ax.set_xticks(pd.to_datetime(["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01",]))
    #     ax.set_xticklabels(["", "2017", "", "2019", "", "2021", "", "2023", "", "2025",], fontsize=6)
    #     ax.tick_params(axis='y', labelsize=6)
    #     ax.set_ylim(0.5, 1.5)
    #     ax.set_yticks([0.5, 1.0, 1.5])
    #     for label in ax.get_xticklabels():
    #         label.set_rotation(45)
    #         label.set_horizontalalignment('right')
    #         label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
    # axes[9,0].set_ylabel("Relative\nsusceptibility", fontsize=6)

    # # add big letter "A" to first row, "B" to eighth row, "C" to ninth row
    # ax_top.text(-0.05, 0.8, "A", transform=ax_top.transAxes, fontsize=16, fontweight="bold")
    # axes[1,0].text(-0.5, 1.2, "B", transform=axes[1,0].transAxes, fontsize=16, fontweight="bold")
    # axes[8,0].text(-0.5, 1.1, "C", transform=axes[8,0].transAxes, fontsize=16, fontweight="bold")
    # axes[9,0].text(-0.5, 1.1, "D", transform=axes[9,0].transAxes, fontsize=16, fontweight="bold")

    # plt.savefig(f"Figures/fits_figure_diagram_test.png", dpi=300)

    # plot_mcmc_corner("InfluenzaB", "Default", "dedupsac", "2020-01-01maxagep035", "260615", prune=0)

    # fig, ax = plt.subplots(figsize=(3,3))
    # im = plot_supression_rank_heatmap(ax, flunet_pathogens)
    # # ax.set_title("How often does y re-emerge before x?")
    # ax.set_ylabel("How often does...")
    # ax.set_xlabel("re-emerge after ...?")
    # plt.tight_layout()
    # plt.savefig(f"Figures/supression_rank_heatmap.png", dpi=300)

    # ## plot MCMC corners and traces for all pathogens
    # for pathogen, option2, seed in zip(pathogens, option2s, seeds):
    #     print(pathogen, option2, seed)
    #     plot_mcmc_corner(pathogen, option2, seed, prune=10000)
    #     print(f"plotting MCMC traces for {pathogen}...")
    #     fig, axes = plt.subplots(4, 4, figsize=(13.3,7.5), sharex=True)
    #     # Calculate this once to avoid repeating the function call
    #     n_params = len(parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)[0])
    #     plot_mcmc_traces(axes.flatten(), pathogen, option2, seed, prune=0)
    #     fig.suptitle(f"MCMC traces for {nice_names.get(pathogen, pathogen)}", fontsize=16)
    #     for i, ax in enumerate(axes.flatten()):
    #         if i >= n_params:
    #             ax.axis('off')
    #     for col in range(axes.shape[1]):
    #         for row in reversed(range(axes.shape[0])):
    #             flat_idx = row * axes.shape[1] + col
    #             if flat_idx < n_params:
    #                 axes[row, col].tick_params(labelbottom=True)
    #                 axes[row, col].set_xlabel("Iteration number")
    #                 break
    #     plt.tight_layout()
    #     plt.savefig(f"Figures/mcmc_traces_{pathogen}_{option2}_{seed}_slide.png", dpi=300)
    #     plt.close()

    # ## Generate Figure 1: timeseries and suppression duration figure
    fig = plt.figure(layout="constrained", figsize=(6.5,7.5))
    countries = ["Brazil", "Canada", "India", "Malaysia", "Qatar"]
    plot_suppression_durations(fig, flunet_pathogens, countries, colors)
    plt.savefig(f"Figures/supression_durations_w_season_age.png", dpi=300)

    # ## Generate supplemental figures of all FluNet timeseries
    # directory_path = "Data/Processed/FluNetTimeseries/"
    # countries = []
    # for filename in os.listdir(directory_path):
    #     if filename.endswith(".csv"):
    #         country = filename.split("__")[0]
    #         if country not in countries:
    #             countries.append(country)
    # countries.sort()
    # # iterate over chunks of ten countries and plot their FluNet data
    # for i in range(0, len(countries), 15):
    #     fig, axes = plt.subplots(min(15, len(countries) - i), 6, figsize=(6.5,9))
    #     countries_chunk = countries[i:i+15]
    #     plot_FluNet_chunk(axes, countries_chunk)
    #     plt.tight_layout()
    #     plt.savefig(f"Figures//FluNetTimeseries_{countries[i]}_to_{countries[min(i+14, len(countries)-1)]}.png", dpi=300)
    
    # ## Generate Figure 2: age-structured fits figure
    # fig, axes = plt.subplots(7, 6, figsize=(6.5,6.5), layout="constrained")
    # plot_fits(axes, n_samples=400, load_data=True)
    # fig.text(0.001, 0.5, 'Estimated incidence of hospitalization per 100k members', va='center', rotation='vertical')
    # plt.savefig(f"Figures/age_structured_fits.png", dpi=300)

    # ## Generate Figure 4: age infection figure
    # fig = plt.figure(figsize=(6.5, 6), layout="constrained")
    # gs_main = fig.add_gridspec(2, 1, height_ratios=[2, 1.2], hspace=0.05) 
    # gs_top = gs_main[0].subgridspec(2, 5, width_ratios=[1, 1, 1, 0.2, 1])
    # gs_bottom = gs_main[1].subgridspec(1, 8)
    # import numpy as np
    # ax_top = np.empty((2, 4), dtype=object)
    # for r in range(2):
    #     for c in range(3):
    #         ax_top[r, c] = fig.add_subplot(gs_top[r, c])
    #     ax_top[r, 3] = fig.add_subplot(gs_top[r, 4])
    # ax_bottom = np.empty((1,8), dtype=object)
    # for c in range(8):
    #     ax_bottom[0, c] = fig.add_subplot(gs_bottom[c])
    # axes = [ax_top, ax_bottom]
    # plot_age_figure(axes, pathogens, colors, option1, option2s, seeds, lockdown, NAG, CENSUS_AGE_POP, AGE_GROUP_NAMES, age_adjusted=True, logD=True, samples=400, load_data=True, prefix="")
    # plt.savefig(f"Figures/infection_matrices_{seeds[0]}_{option1}_{lockdown}_uncertainty_logD_adjusted.png", dpi=300)

    # # Generate Figure 5: age group heatmaps and line of best fit
    # from sim_grid import generate_2d_heatmap_plot
    # good_simulations = [[pathogen, seed, lockdown, option1, option2] for pathogen, seed, option2 in zip(pathogens, seeds, option2s)]
    # run_save_path = "Outputs/sim_grid_lh_n80000_chunk10000_seed260624_lockdownExponentialODipp25_2d"
    # fig = plt.figure(figsize=(4.5, 4.5))
    # plot_heatmaps_and_best_fit(fig, pathogens, option2s, seeds, colors, run_save_path, good_simulations, r0_base, fit_line=False, prune=100)
    # plt.savefig(f"Figures/figure_five_update5.png", dpi=1000)
    # plt.close()

    # fig, ax1 = plt.subplots(1, 1, figsize=(4.5,4))
    
    # # Plot proportion positive on left y-axis
    # color1 = "red"
    # kpsc_proportion_positive_incidence_plot(ax1, pathogen="RSV", AGE_GROUPS=None, AGE_GROUP_NAMES=AGE_GROUP_NAMES, title=None, color=color1, legend=False, aggregation="W", window_size=1, weighting_factor=0, label="Proportion positive", factor=10000, annotations=False, pp_only=False, hosp=True)
    # ax1.tick_params(axis='y', labelcolor=color1)
    # for line in ax1.get_lines():
    #     line.set_alpha(0.7)
    # ax1.set_ylabel('Proportion positive incidence per 10k', color=color1)
    
    # # Create second y-axis for positive tests
    # ax2 = ax1.twinx()
    # color2 = "blue"
    # ax2.set_ylabel('Positive tests incidence per 10k', color=color2)
    # kpsc_positive_test_plot(ax2, pathogen="RSV", AGE_GROUPS=None, AGE_GROUP_NAMES=AGE_GROUP_NAMES, color=color2, legend=False, aggregation="W", factor=10000, label="Positive tests")
    # ax2.tick_params(axis='y', labelcolor=color2)
    # for line in ax2.get_lines():
    #     line.set_alpha(0.7)
    # fig.tight_layout()
    # plt.savefig("Figures/KPSC_RSV_proportion_positive_vs_positive_test_incidence_weekly.png", dpi=300)

    # fig = plt.figure(layout="constrained", figsize=(7,4))

    # # pathogens = ["RSV","Metapneumovirus","InfluenzaA","InfluenzaB","Adenovirus","Parainfluenza3",]

    # subfigs = fig.subfigures(1, 2, wspace=0.05, width_ratios=[7, 3])
    # axA = subfigs[1].subplots(len(pathogens), 1, sharex = True)
    # axB = subfigs[0].subplots((len(pathogens) + 1)//2, 2, sharex = True)

    # # # figA, axA = plt.subplots(6, 1, figsize=(2.5,4), sharex = True)
    # for pi,pathogen in enumerate(pathogens):
    #     print(pathogen)
    #     age_group_incidence_plot(axA[pi],pathogen,color="k",season="pre_median", AGE_GROUPS=AGE_GROUPS, AGE_GROUP_NAMES=AGE_GROUP_NAMES, NAG=NAG, label="Pre-COVID-19")
    #     age_group_incidence_plot(axA[pi],pathogen,color="silver",season="rebound", AGE_GROUPS=AGE_GROUPS, AGE_GROUP_NAMES=AGE_GROUP_NAMES, NAG=NAG, label="Re-emergence")
    #     axA[pi].set_title(nice_names.get(pathogen, pathogen))
    # # set singe x label for all subplots
    # axA[-1].set_xlabel("Age group")
    # axA[0].legend(loc="upper right", fontsize=6)
    # # set single y label for all subplots
    # subfigs[1].text(-0.05, 0.5, 'Incidence per 100k members', va='center', rotation='vertical')
    # # # plt.tight_layout()
    # # # plt.savefig("Figures/KPSC_age_group_incidence_pre_median_rebound.png",dpi=300)

    # # figB, axB = plt.subplots(3, 2, figsize=(6.5,4), sharex = True)
    # data_color = "#648FFF"
    # aggregation = "MS"
    # agg_factor = {"D":1, "W-MON":7, "MS":30.44}[aggregation]
    # factor = 100000
    # if factor >= 1000000:
    #     factor_label = f"{factor // 1000000}M"
    # elif factor >= 1000:
    #     factor_label = f"{factor // 1000}k"
    # else:
    #     factor_label = str(factor)
    # for pi, pathogen in enumerate(pathogens):
    #     kpsc_proportion_positive_incidence_plot(
    #         axB[pi//2, pi%2], pathogen=pathogen, AGE_GROUP_NAMES=AGE_GROUP_NAMES, title=nice_names.get(pathogen, pathogen),
    #         color=data_color, aggregation=aggregation, factor=factor,
    #         annotations=True, definition="30% median", label="Data", hosp=True, dedup=True)
    # # suppress all y labels and replace with single label on left
    # for i in range(len(pathogens)//2):
    #     for j in range(2):
    #         axB[i,j].set_ylabel("")
    # axB[len(pathogens)//4,0].set_ylabel(f"Incidence per {factor_label} members")
    # # only include every other year label
    # for axB_i in axB.flatten():
    #     axB_i.set_xticks(pd.date_range(start='2016-01-01',end='2025-01-01',freq='2YS'))
    #     axB_i.set_xticklabels([str(year) for year in range(2016,2026,2)])
    #     axB_i.set_xticks(pd.date_range(start='2016-01-01',end='2025-01-01',freq='YS'), minor=True)

    # # plt.tight_layout()
    # # plt.savefig("Figures/KPSC_proportion_positive_incidence_weekly_annotated.png",dpi=300)

    # # now include plots of simulations on top of data
    
    # PERIOD = pd.date_range(start=pd.to_datetime('2015-07-04'), end=pd.to_datetime('2025-05-01'), freq='D')
    # POINTS = np.array(date_to_t(PERIOD))
    # start_idx = int(date_to_t(PERIOD[0]) + 90 - date_to_t('2015-10-01'))
    # end_idx = int(date_to_t(PERIOD[-1]) - date_to_t('2015-10-01'))
    # ## Initial conditions
    # STATE0 = jnp.zeros((2*N_S+1,NAG))
    # STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
    # STATE0 = STATE0.at[1,:].set(1)
    # # # flatten initial state and add maternal immunity compartment
    # STATE0 = STATE0.flatten()
    # STATE0 = jnp.concatenate((jnp.array([0]), STATE0))
    # _, _, _, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=1e-9, incidence_data=False, hosp=True, NAG=NAG, dedup=True)
    # data = data_full[start_idx:end_idx]
    # daily_hospitalization_rates_pd = pd.read_csv('Data/Processed/KPSC_ARI_nonCOVID_hospitalization_rates_by_day_age_group'+['','_split'][NAG>7]+['','_detrended']["detrend" in option1]+["","_dedup"]["dedup" in option1]+'.csv',index_col=0,parse_dates=True)
    # daily_hospitalization_rates_full = jnp.asarray(daily_hospitalization_rates_pd.values)
    # daily_hospitalization_rates = daily_hospitalization_rates_full[start_idx:end_idx,]

    # good_simulations = [
    #     [pathogen, seed_i, lockdown, option1, option2]
    #     for pathogen, option2, seed_i in zip(pathogens, option2s, seeds)
    # ]
    # from Parameters.times_and_contacts import PERIOD
    # POINTS = jnp.array(date_to_t(PERIOD))
    # T_LOCKDOWN = date_to_t(pd.to_datetime("2020-03-20"))
    # for pi, sim in enumerate(good_simulations):
    #     pathogen = sim[0]
    #     print(f"Plotting simulation for {pathogen}...")
    #     _, x, _ = load_optimization_results("", sim[0], sim[1], sim[2], sim[3], sim[4])
    #     params = x_to_params(x, sim[0], sim[2], sim[3], sim[4], NAG=NAG, return_contact=False, print_params=False)
    #     lockdown_incidence_plot(axB[pi//2,pi%2], STATE0, params, POINTS, T_LOCKDOWN,
    #                             p_time_to_obs=p_time_to_obs, NAG=NAG,
    #                             test_data=data_full,daily_hospitalization_rates=daily_hospitalization_rates,aggregation=aggregation,
    #                             color="#DC267F", factor=factor*agg_factor, label="Simulation")
    # axB[1,1].legend(loc="upper right")
    
    # # plt.tight_layout()
    # # plt.savefig("Figures/ReportOverallIncidenceWeeklyExponential2604172_annotate100.png",dpi=300)

    # subfigs[0].suptitle("A", x=0.01, fontweight='bold')
    # subfigs[1].suptitle("B", x=0.01, fontweight='bold')
    # # plt.tight_layout()
    # plt.savefig("Figures/Figure1_test_monthly.png",dpi=300)