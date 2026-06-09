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

from utils import date_to_t, t_to_date, calculate_population_size, susceptibility, infections_by_age, observations, load_optimization_results, x_to_params, sum_age_to, pathogen_parameters, load_mcmc_chain, load_random_mcmc_result

N_C = 2
N_S = 3
# NAG = 7
from JAX_ODEs import deltas

##### General plotting parameters #####
hsv_colors = colormaps.hsv(-0.02+np.arange(7)/7)
hsv_colors[3] = colormaps.hsv((3/7)+0.04)
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


def lockdown_susceptibility_plot(ax,state0,params,period,points,T_LOCKDOWN,solution=None,label='Susceptible_population',color='#648FFF',relative=True,proportion=False,by_age=False,AGE_GROUP_NAMES=None,style='-',delta=deltas,NAG=7,NAG_eff=7,AGE_GROUPS=None, max_month=None):
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

def get_infection_matrix(pathogen, seed, lockdown, option1, option2, NAG, census_age_pop, hospitalizations=False, prefix=""):
    from fit_MCMC import run_simulation
    _, x, _ = load_optimization_results(prefix, pathogen, seed, lockdown, option1, option2)
    params = x_to_params(x, pathogen, lockdown, option1, option2, NAG=NAG)
    PERIOD = pd.date_range(start=pd.to_datetime('2015-10-01'), end=pd.to_datetime('2025-10-01'), freq='D')
    POINTS = np.array(date_to_t(PERIOD))
    ## Initial conditions
    STATE0 = jnp.zeros((2*N_S+1,NAG))
    STATE0 = STATE0.at[0,:].set(census_age_pop-1)
    STATE0 = STATE0.at[1,:].set(1)
    # # flatten initial state and add maternal immunity compartment
    STATE0 = STATE0.flatten()
    STATE0 = jnp.concatenate((jnp.array([0]), STATE0))
    solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
    # find total observed infections each season in each age group
    values = solution.ys.T
    shaped_values = values[1:, :].reshape((1+2*N_S, NAG, -1))
    infectious = shaped_values[1:2*N_S:2, :, :]
    susceptible = shaped_values[0:2*N_S:2, :, :]
    all_infectious = (params[7][:, :, None] * infectious).sum(axis=0)
    population_size = calculate_population_size(values, NAG=NAG)
    foi_matrix = params[4] * params[3][:, :, None] * all_infectious[None, :, :] / jnp.sum(population_size, axis=1)[None, None, :]
    infections_matrix = params[6][:, None, None, None] * foi_matrix[None, :, :, :] * susceptible[:, :, None, :]
    if hospitalizations:
        infections_matrix = infections_matrix * params[8][:, None, None, None] * params[9][None, :, None, None]
    age_infections_matrix = infections_matrix.sum(axis=0)[:1553].mean(axis=-1)
    return age_infections_matrix

def plot_infection_matrix(ax, pathogen=None, seed=None, lockdown=None, option1=None, option2=None, NAG=None, age_group_names=None, census_age_pop=None, hospitalizations=False, prefix="", matrix=None):
    if matrix is None:
        normalized_age_infections_matrix = get_infection_matrix(pathogen, seed, lockdown, option1, option2, NAG, census_age_pop, hospitalizations, prefix)
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
    values = jnp.zeros(len(matrices))
    for i, matrix in enumerate(matrices):
        value = matrix[age_group_idx,age_group_idx]/matrix[age_group_idx,:].sum()
        if census_age_pop is not None:
            value *= census_age_pop.sum()/census_age_pop[age_group_idx]
        values = values.at[i].set(value)
    # if two values are close together, add -0.1 to one of them and +0.1 to the other to separate them visually
    separation =jnp.zeros(len(matrices))
    for i in range(len(matrices)):
        for j in range(i+1, len(matrices)):
            if abs(values[i] - values[j]) < 0.05 * (jnp.max(values)-jnp.min(values)):
                separation = separation.at[i].set(separation[i] - 0.135)
                separation = separation.at[j].set(separation[j] + 0.135)
    for i in range(len(matrices)):
        ax.scatter(separation[i], values[i], label=pathogens[i], color=colors[i], marker="o")
    ax.set_xlim(-0.5,0.5)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def plot_infections_versus(ax, matrices, age_group_indices, pathogens, colors, census_age_pop):
    for i, matrix in enumerate(matrices):
        x_value = matrix[:,age_group_indices[0]].sum()/census_age_pop[age_group_indices[0]].sum()
        y_value = matrix[:,age_group_indices[1]].sum()/census_age_pop[age_group_indices[1]].sum()
        ax.scatter(x_value, y_value, label=pathogens[i], color=colors[i], marker="o")
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
def plot_age_figure(axes, pathogens, colors, option1, option2s, seeds, lockdown, NAG, CENSUS_AGE_POP, AGE_GROUP_NAMES, age_adjusted=False, prefix=""):
    ax_top, ax_bottom = axes
    matrices = {}
    hosp_matrices = {}
    for pi, (pathogen, option2, seed) in enumerate(zip(pathogens, option2s, seeds)):
        age_infections_matrix = get_infection_matrix(pathogen, seed, lockdown, option1, option2, NAG, CENSUS_AGE_POP, hospitalizations=False, prefix=prefix)
        normalized_age_infections_matrix = age_infections_matrix / CENSUS_AGE_POP[:, None]
        age_hospitalizations_matrix = get_infection_matrix(pathogen, seed, lockdown, option1, option2, NAG, CENSUS_AGE_POP, hospitalizations=True, prefix=prefix)
        plot_infection_matrix(ax_top[pi//3, pi%3], age_group_names=AGE_GROUP_NAMES, NAG=7, matrix=normalized_age_infections_matrix)
        ax_top[pi//3, pi%3].set_title(nice_names.get(pathogen, pathogen))
        matrices[pathogen] = age_infections_matrix
        hosp_matrices[pathogen] = age_hospitalizations_matrix
    for age_group_idx in range(NAG):
        if age_adjusted:
            pop_arg = CENSUS_AGE_POP
        else:            
            pop_arg = None
        plot_same_age_infection(ax_bottom[0, age_group_idx], [matrices[pathogen] for pathogen in pathogens], age_group_idx, pathogens, colors, pop_arg)
        ax_bottom[0, age_group_idx].set_xlabel(AGE_GROUP_NAMES[age_group_idx])
    plot_infections_versus(ax_top[0,3], [matrices[pathogen] for pathogen in pathogens], [slice(0, 3), -1], pathogens, colors, CENSUS_AGE_POP)
    ax_top[0,3].set_xlabel("<1y")
    ax_top[0,3].set_ylabel(">65y")
    plot_infections_versus(ax_top[1,3], [hosp_matrices[pathogen] for pathogen in pathogens], [slice(0, 3), -1], pathogens, colors, CENSUS_AGE_POP)
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

    trans_A = mtransforms.blended_transform_factory(ax_top[0,0].transAxes, ax_top[0,3].transAxes)
    # trans_D = mtransforms.blended_transform_factory(ax[0,0].transAxes, ax[2,0].transAxes)
    ax_top[0,0].text(-0.5, 1.1, "A", transform=trans_A, fontsize=16, fontweight="bold")
    ax_top[0,3].text(-0.5, 1.1, "B", transform=ax_top[0,3].transAxes, fontsize=16, fontweight="bold")
    ax_top[1,3].text(-0.5, 1.1, "C", transform=ax_top[1,3].transAxes, fontsize=16, fontweight="bold")
    # ax[2,0].text(-0.5, 1.1, "D", transform=trans_D, fontsize=16, fontweight="bold")
    ax_bottom[0,0].text(-0.5, 1, "D", transform=ax_bottom[0,0].transAxes, fontsize=16, fontweight="bold")

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

def plot_FluNet(ax, pathogen, country, color='k', linewidth=1):
    data = pd.read_csv(f"Data/Processed/FluNetTimeseries/{country}__{pathogen}.csv", parse_dates=['month_start'])
    supression_data = pd.read_csv("Data/Processed/FluNet_suppression_duration_by_country_pathogen.csv")
    # restrict to month_start >= 2015-10-01
    data = data[(data["month_start"] >= pd.to_datetime("2015-10-01")) & (data["month_start"] < pd.to_datetime("2026-05-01"))]
    if data["count"].max() > 0:
        ax.plot(data["month_start"], data["count"], label=country, color=color, linewidth=linewidth)
        
        # Add suppression period indicator
        pre_2020_data = data[data["month_start"] < pd.to_datetime("2020-03-01")]
        if len(pre_2020_data) > 0:
            pre_2020_max = pre_2020_data["count"].max()
            threshold = pre_2020_max / 20
            post_2020_data = data[data["month_start"] >= pd.to_datetime("2020-03-01")]
            suppressed = post_2020_data[post_2020_data["count"] < threshold]
            if ((supression_data["pathogen"] == pathogen) & (supression_data["country"] == country) & (supression_data["status"] == "excluded")).any():
                # ax.plot([0.1], [0.9], marker='x', markersize=10, color='grey', transform=ax.transAxes)
                ax.annotate("X", xy = (pd.to_datetime("2020-03-01"), post_2020_data["count"].min()), xytext=(0,2), textcoords='offset points', ha='right', va="top", color='grey')
            elif len(suppressed) > 0:
                supp_start = suppressed.iloc[0]["month_start"]
                not_suppressed = post_2020_data[(post_2020_data["month_start"] > supp_start) & (post_2020_data["count"] >= threshold)]
                supp_end = not_suppressed.iloc[0]["month_start"] if len(not_suppressed) > 0 else suppressed.iloc[-1]["month_start"]
                y_pos = threshold
                ax.plot([supp_start, supp_end], [y_pos, y_pos], color='red', linewidth=linewidth)
                time_diff = supp_end - supp_start
                time_amount = f"{time_diff.days // 30}"
                last_pre_time = data[data["month_start"] < pd.to_datetime("2020-03-01")]["month_start"].iloc[-1]
                ax.annotate(time_amount, xy=(last_pre_time + time_diff/2, threshold), xytext=(0,2), textcoords='offset points', ha='center', va="bottom", color='red')
    else:
        ax.axis('off')

def plot_suppression_durations(fig, pathogens, countries, colors):
    gs = fig.add_gridspec(8, 6, height_ratios=[1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 3], hspace=0.1)
    
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
        # Remove all spines except bottom
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_ylabel("")
        ax.set_yticks([])
        if i==0:
            ax.set_ylabel("KPSC", rotation=0, ha='right', labelpad=25)
            ax2 = ax.twinx()
            ax2.set_ylabel("Estimated\n+ve hospitalizations", rotation=90, va='center')
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
    
    # FluNet plots in middle rows (rows 1-5)
    country_axes = np.empty((len(countries), len(pathogens)), dtype=object)
    for j, country in enumerate(countries):
        for i, pathogen in enumerate(pathogens):
            ax = fig.add_subplot(gs[j+2, i])
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
                ax2.set_ylabel("Detected cases", rotation=90, va='center', fontsize=6)
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
            # if j == 0:
            #     ax.set_ylabel(short_names.get(pathogen, pathogen), rotation=0, ha='right')
    
    # Violin plots in bottom row (row 7)
    violin_axis = np.empty(len(pathogens), dtype=object)
    for i in range(len(pathogens)):
        violin_axis[i] = fig.add_subplot(gs[7, i])
    supression_violin(violin_axis, pathogens, colors)
    
    # big label "C" on top left of kpsc plots, "B" on top left of country plots, "A" on top left of violin plots
    kpsc_axis[0].text(-0.5, 1.1, "A", transform=kpsc_axis[0].transAxes, fontsize=16, fontweight="bold")
    country_axes[0,0].text(-0.5, 1.1, "B", transform=country_axes[0,0].transAxes, fontsize=16, fontweight="bold")
    violin_axis[0].text(-0.5, 1.1, "C", transform=violin_axis[0].transAxes, fontsize=16, fontweight="bold")



if __name__ == "__main__":
    plt.rcParams.update({'font.size':8})
    # text type is palatino
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica']

    seed = 260531
    option1 = "dedupsac"
    NAG = 7 + ("split" in option1)
    if "split" in option1:
        from Parameters.census_population import CENSUS_AGE_POP_split as CENSUS_AGE_POP, AGE_GROUP_NAMES_split as AGE_GROUP_NAMES, AGE_GROUPS_split as AGE_GROUPS
    elif "sac" in option1:
        from Parameters.census_population import CENSUS_AGE_POP_sac as CENSUS_AGE_POP, AGE_GROUP_NAMES_sac as AGE_GROUP_NAMES, AGE_GROUPS_sac as AGE_GROUPS
    else:
        from Parameters.census_population import CENSUS_AGE_POP, AGE_GROUP_NAMES
    lockdown = "ExponentialODipLinear"

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

    pathogens = ["RSV","Metapneumovirus","Parainfluenza3","Adenovirus","InfluenzaA","InfluenzaB",]
    colors = ["#DC267F", "#FFB000", "#FF832B", "#648FFF", "#785EF0", "#004D40"]
    option2s = ["maxagep028","maxagep015","maxagep004","maxagep003","maxagep035","maxagep035",]
    seeds = [260531, 260603, 260602, 260531, 260531, 260531,]

    # ### Figure 5 - very janky version
    # from sim_grid import plot_two_heatmaps
    # from test import plot_line_figure
    # fig = plt.figure(figsize=(6.5,6.5))
    # gs = fig.add_gridspec(2, 2)
    # ax1 = [fig.add_subplot(gs[0, i]) for i in range(2)]
    # ax2 = fig.add_subplot(gs[1, :])
    # plot_two_heatmaps(ax1)
    # plot_line_figure(ax2)
    # plt.savefig("Figures/figure_five_draft.png", dpi=300)

    # # ## Generate Figure 1: timeseries and suppression duration figure
    # fig = plt.figure(layout="constrained", figsize=(6.5,8.5))
    # countries = ["Brazil", "Canada", "India", "Malaysia", "Qatar"]
    # flunet_pathogens = ["RSV","Metapneumovirus","Parainfluenza","Adenovirus","InfluenzaA","InfluenzaB",]
    # plot_suppression_durations(fig, flunet_pathogens, countries, colors)
    # plt.savefig(f"Figures/supression_durations_cherry.png", dpi=300)

    # ## Generate Figure 2: age-structured fits figure
    # fig, axes = plt.subplots(7, 6, figsize=(6.5,6.5), layout="constrained")
    # aggregation = "MS"
    # agg_factor = 30.44
    # factor = 100000
    # for pathogen_idx in range(len(pathogens)):
    #     ax = axes[:,pathogen_idx]
    #     pathogen, option2, seed = pathogens[pathogen_idx], option2s[pathogen_idx], seeds[pathogen_idx]
    #     _, _, _, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=1e-9, incidence_data=False, hosp=True, NAG=NAG, dedup=True)
    #     data = data_full[start_idx:end_idx]
    #     n_samples = 100
    #     _, chain, _ = load_mcmc_chain(pathogen, seed, lockdown, option1, option2)
    #     np.random.seed(260604)
    #     # choose n_samples random rows from chain
    #     random_indices = np.random.choice(chain.shape[0], size=n_samples, replace=False)
    #     random_samples = chain[random_indices, :]
    #     from fit_MCMC import run_simulation
    #     def get_solution(x):
    #         params = x_to_params(x, pathogen, lockdown, option1, option2, NAG=NAG)
    #         solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
    #         return solution
    #     solutions = jax.jit(jax.vmap(get_solution))(random_samples)
    #     for sample_i in range(n_samples):
    #         solution = jax.tree_util.tree_map(lambda x: x[sample_i], solutions)
    #         for age_group_idx, age_group_name in enumerate(AGE_GROUP_NAMES):
    #             lockdown_incidence_plot(ax[age_group_idx], STATE0, None, POINTS, pd.to_datetime('2020-03-01'),
    #                                     solution=solution,
    #                                     by_age=True, select_age_group=age_group_idx, AGE_GROUP_NAMES=AGE_GROUP_NAMES, AGE_GROUPS=AGE_GROUPS,
    #                                     p_time_to_obs=p_time_to_obs, NAG=NAG,
    #                                     test_data=data_full,daily_hospitalization_rates=daily_hospitalization_rates,aggregation=aggregation,
    #                                     uncertainty="draw",
    #                                     color=colors[pathogen_idx], factor=factor*agg_factor, label="Simulation", linewidth=0.5, alpha=0.05)
    #     for age_group_idx, age_group_name in enumerate(AGE_GROUP_NAMES):
    #         kpsc_proportion_positive_incidence_plot(ax[age_group_idx], pathogen=pathogen,
    #                                                 AGE_GROUPS=AGE_GROUPS, AGE_GROUP_NAMES=AGE_GROUP_NAMES, select_age_group=age_group_idx,
    #                                                 aggregation=aggregation, factor=factor, annotations=False, definition="", label="Data", hosp=True, dedup=True,
    #                                                 title="", color="k", linewidth=0.5)
    #         if pathogen_idx == 0:
    #             ax[age_group_idx].set_ylabel(age_group_name)
    #         else:
    #             ax[age_group_idx].set_ylabel("")
    #         if age_group_idx == 0:
    #             ax[age_group_idx].set_title(short_names.get(pathogen, pathogen))
    #         ax[age_group_idx].set_xticks(pd.to_datetime(["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01",]))
    #         if age_group_idx == len(AGE_GROUP_NAMES)-1:
    #             ax[age_group_idx].set_xticklabels(["", "2017", "", "2019", "", "2021", "", "2023", "", "2025",], fontsize=6)
    #             for label in ax[age_group_idx].get_xticklabels():
    #                 label.set_rotation(45)
    #                 label.set_horizontalalignment('right')
    #                 label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
    #         else:
    #             ax[age_group_idx].set_xticklabels([])
    #         ax[age_group_idx].legend().set_visible(False)
    #         ax[age_group_idx].tick_params(axis='y', labelsize=6)
    # # big y label for all plots
    # fig.text(0.001, 0.5, 'Estimated incidence of hospitalization per 100k members', va='center', rotation='vertical')
    # # plt.tight_layout(rect=[0.03, 0, 1, 1])
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
    # plot_age_figure(axes, pathogens, colors, option1, option2s, seeds, lockdown, NAG, CENSUS_AGE_POP, AGE_GROUP_NAMES, prefix="")
    # plt.savefig(f"Figures/infection_matrices_{seeds[0]}_{option1}_{lockdown}_vert_sameage.png", dpi=300)

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