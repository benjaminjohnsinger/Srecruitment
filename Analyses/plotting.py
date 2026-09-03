## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

import os
import re
import jax
import jax.numpy as jnp
from matplotlib.patches import Patch
import numpy as np
import scipy as sp
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
from matplotlib import cm as colormaps, ticker
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from math import comb
import corner
import pickle
import colorsys
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController
import time

from utils import date_to_t, t_to_date, calculate_population_size, susceptibility, load_optimization_results, x_to_params, sum_age_to, pathogen_parameters, load_mcmc_chain, parameters_names_bounds, constrained_immunity, calculate_R0_from_values

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

            if uncertainty == "draw":
                p = np.clip(expected_prop, 0, 1)
                p = np.nan_to_num(p, nan=0.0)
                n_int = np.maximum(n_tests.astype(int), 0)
                draw_k = np.random.binomial(n_int, p)
                denom = n_tests * pop_size_col
                with np.errstate(divide='ignore', invalid='ignore'):
                    drawn_path = np.where(n_tests > 0, overall_hosp * draw_k / denom, np.nan)
            elif uncertainty == "confidence":
                for t_idx in range(len(expected_prop)):
                    if n_tests[t_idx] > 0:
                        p = np.clip(expected_prop[t_idx], 0, 1)
                        ci_lower[t_idx] = overall_hosp[t_idx] * sp.stats.binom.ppf(0.025, n_tests[t_idx], p) / (n_tests[t_idx] * pop_size_col[t_idx])
                        ci_upper[t_idx] = overall_hosp[t_idx] * sp.stats.binom.ppf(0.975, n_tests[t_idx], p) / (n_tests[t_idx] * pop_size_col[t_idx])
            
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
                mx = np.nanmax(factor * drawn_path / agg_factor)

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
                with np.errstate(divide='ignore', invalid='ignore'):
                    expected_prop = expected_obs_agg / overall_hosp
                n_tests = df_agg['n_tests'].resample(agg_freq).sum().values
                dates_agg = df_agg['expected_obs'].resample(agg_freq).sum().index.to_list()
            else:
                dates_agg = dates[(start_index + 1):end_index]
                pop_size_agg = np.sum(pop_size_by_age[start_index:end_index, :], axis=1)
            
            ci_lower = np.zeros_like(expected_prop)
            ci_upper = np.zeros_like(expected_prop)
            drawn_path = np.zeros_like(expected_prop)

            if uncertainty=="draw":
                p = np.clip(expected_prop, 0, 1)
                n_int = np.maximum(n_tests.astype(int), 0)
                draw_k = np.random.binomial(n_int, p)
                denom = n_tests * pop_size_agg
                with np.errstate(divide='ignore', invalid='ignore'):
                    drawn_path = np.where(n_tests > 0, overall_hosp * draw_k / denom, np.nan)
            elif uncertainty=="confidence":
                for t_idx in range(len(expected_prop)):
                    if n_tests[t_idx] > 0:
                        p = np.clip(expected_prop[t_idx], 0, 1)
                        ci_lower[t_idx] = overall_hosp[t_idx] * sp.stats.binom.ppf(0.025, n_tests[t_idx], p) / (n_tests[t_idx] * pop_size_agg[t_idx])
                        ci_upper[t_idx] = overall_hosp[t_idx] * sp.stats.binom.ppf(0.975, n_tests[t_idx], p) / (n_tests[t_idx] * pop_size_agg[t_idx])

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
                mx = np.nanmax(factor * drawn_path / agg_factor)

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
short_names = {"RSV": "RSV", "InfluenzaA": "IAV", "InfluenzaB": "IBV", "Metapneumovirus": "hMPV", "Adenovirus": "AdV", "Parainfluenza": "PIV", "Parainfluenza3": "PIV3", "Rhinovirus": "RhV", "Pertussis": "Pertussis", "M.pneumoniae": "M. pneumo", "C.pneumoniae": "C. pneumo", "SARS-CoV-2": "COVID-19", "Enterovirus": "EV"}
from data_processing import calculate_proportion_positive_incidence
def kpsc_proportion_positive_incidence_plot(ax, pathogen="RSV", AGE_GROUPS=None, AGE_GROUP_NAMES=None, select_age_group=None, title=None, color=hsv_colors, linewidth=1, legend=True, aggregation="D", window_size=28, weighting_factor=0.5, label=None, factor=1000000, annotations=False, definition="50% median", pp_only=False, hosp=False, detrend=False, dedup=False, sac=True, mask=[3135,3288], zorder=3):
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
                ax.plot(incidence.index, incidence[AGE_GROUP_NAMES[i]], label=AGE_GROUP_NAMES[i], color=color[i], linewidth=linewidth, zorder=zorder)
        else:
            # print(AGE_GROUP_NAMES[select_age_group], "incidence:")
            # print(list(incidence[AGE_GROUP_NAMES[select_age_group]]))
            if label is None:
                label = AGE_GROUP_NAMES[select_age_group]
            ax.plot(incidence.index, incidence[AGE_GROUP_NAMES[select_age_group]], label=label, color=color, linewidth=linewidth, zorder=zorder)
    else:
        if label is None:
            label = nice_names.get(pathogen, pathogen)
        ax.plot(incidence.index, incidence["Total"], color=color, label=label, linewidth=linewidth, zorder=zorder)

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
        # return the beginning and end of the suppression period
        return last_pre_time, rebound_time
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

def plot_relative_age_incidence(ax,pathogen,group_names=None,aggregation="YS-OCT"):
    proportional_incidence, lower_ci, upper_ci = calculate_proportion_positive_incidence(pathogen, aggregation=aggregation, window_size=1, weighting_factor=0, sum_age_groups=False, save_counts=False, pp_only=False, hosp=True, NAG=NAG, detrend=False, dedup=True, sac=True, return_ci=True, cum_sum=False)
    pop_by_age_group_month = pd.read_csv('Data/Processed/KPSC_population_by_age_group_monthly_sac.csv', index_col=0, parse_dates=['month_start'])
    pop_size = pop_by_age_group_month.resample(aggregation).ffill()[:proportional_incidence.shape[0]]
    pop_size.index = pd.to_datetime(pop_size.index)
    incidence = proportional_incidence.multiply(pop_size.values, axis=0)
    lower_ci = lower_ci.multiply(pop_size.values, axis=0)
    upper_ci = upper_ci.multiply(pop_size.values, axis=0)
    if "<1y" in group_names or "<5y" in group_names:
        cum_proportional_incidence, cum_lower_ci, cum_upper_ci = calculate_proportion_positive_incidence(pathogen, aggregation=aggregation, window_size=1, weighting_factor=0, sum_age_groups=False, save_counts=False, pp_only=False, hosp=True, NAG=NAG, detrend=False, dedup=True, sac=True, return_ci=True, cum_sum=True)
        cum_pop_size = pop_size.cumsum(axis=1)
        cum_incidence = cum_proportional_incidence.multiply(cum_pop_size.values, axis=0)
        cum_lower_ci = cum_lower_ci.multiply(cum_pop_size.values, axis=0)
        cum_upper_ci = cum_upper_ci.multiply(cum_pop_size.values, axis=0)
    age_incidence = {}
    for group in group_names:
        if group in incidence.columns:
            age_incidence[group] = incidence[group] / incidence.sum(axis=1)
            lower_ci[group] = lower_ci[group] / incidence.sum(axis=1)
            upper_ci[group] = upper_ci[group] / incidence.sum(axis=1)
        elif group == "<1y":
            age_incidence[group] = cum_incidence["3-11m"] / incidence.sum(axis=1)
            lower_ci[group] = cum_lower_ci["3-11m"] / incidence.sum(axis=1)
            upper_ci[group] = cum_upper_ci["3-11m"] / incidence.sum(axis=1)
        elif group == "<5y":
            age_incidence[group] = cum_incidence["1-4y"] / incidence.sum(axis=1)
            lower_ci[group] = cum_lower_ci["1-4y"] / incidence.sum(axis=1)
            upper_ci[group] = cum_upper_ci["1-4y"] / incidence.sum(axis=1)
    age_incidence = pd.DataFrame(age_incidence)
    # set to NA for 2020/21
    if "Parainfluenza" not in pathogen:
        age_incidence.loc[(age_incidence.index >= pd.to_datetime("2020-10-01")) & (age_incidence.index < pd.to_datetime("2021-10-01")), :] = np.nan
        if "Influenza" in pathogen:
            age_incidence.loc[(age_incidence.index >= pd.to_datetime("2020-10-01")) & (age_incidence.index < pd.to_datetime("2022-10-01")), :] = np.nan
            if "B" in pathogen:
                age_incidence.loc[(age_incidence.index >= pd.to_datetime("2018-10-01")) & (age_incidence.index < pd.to_datetime("2019-10-01")), :] = np.nan
                age_incidence.loc[(age_incidence.index >= pd.to_datetime("2022-10-01")) & (age_incidence.index < pd.to_datetime("2025-10-01")), :] = np.nan
    else:
        age_incidence.loc[(age_incidence.index >= pd.to_datetime("2019-10-01")) & (age_incidence.index < pd.to_datetime("2021-10-01")), :] = np.nan
    # set to NA when incidence is zero for that age group
    for group in group_names:
        age_incidence.loc[age_incidence[group] == 0, group] = np.nan
    colors = ["silver", "k"]
    days_in_agg = {"YS-OCT": 365, "MS": 30.44, "W": 7, "D": 1}[aggregation]
    offset = (1/5) * len(group_names)
    for i, group in enumerate(group_names):
        x_offset = (i - len(group_names)/2 + 0.5) * offset * days_in_agg
        ax.scatter(age_incidence.index + pd.Timedelta(days=x_offset), age_incidence[group], color=colors[i], label=group, marker='s', s=3)
        for idx in age_incidence.index:
            if not np.isnan(age_incidence.loc[idx, group]):
                ax.plot([idx + pd.Timedelta(days=x_offset), idx + pd.Timedelta(days=x_offset)], [lower_ci.loc[idx, group], upper_ci.loc[idx, group]], color=colors[i], linewidth=0.7)
    # ax x ticks as years
    print(pd.date_range(start=age_incidence.index.min(), end=age_incidence.index.max(), freq='YS-OCT'))
    ax.set_xticks(pd.date_range(start=age_incidence.index.min(), end=age_incidence.index.max(), freq='YS-OCT'))
    ax.set_xticklabels([str(x.year) for x in pd.date_range(start=age_incidence.index.min(), end=age_incidence.index.max(), freq='YS-OCT')])
    return cum_incidence

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

def get_infection_matrix(pathogen, seed, lockdown, option1, option2, NAG, census_age_pop, prune=None, samples=None, hospitalizations=False, prefix=""):
    from likelihood import run_simulation
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
        _, chain, _ = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, prune=prune, prefix=prefix)
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
    print(np.min(normalized_age_infections_matrix), np.max(normalized_age_infections_matrix))
    im = ax.imshow(normalized_age_infections_matrix, cmap="viridis")
    ax.set_xticks(np.arange(NAG))
    ax.set_xticklabels(age_group_names, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(np.arange(NAG))
    ax.set_yticklabels(age_group_names, fontsize=6)
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
        print(values[3])
        ci_lower, ci_upper = credible_intervals[i]
        ax.scatter(separation[i], values[i], label=pathogens[i], color=colors[i], marker="o", s=4)
        ax.plot([separation[i], separation[i]], [ci_lower, ci_upper], color=colors[i], linewidth=1)
    
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

def get_age_susceptibility_over_time(pathogens, seeds, lockdown, option1, option2s, pruners, n_samples=400):
    n_samples = 400
    age_sus = np.zeros((len(pathogens), len(AGE_GROUP_NAMES), 3, n_samples))

    for i, pathogen, seed, option2, prune in zip(range(6), pathogens, seeds, option2s, pruners):
        print(pathogen)
        chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=prune, prefix="")
        sampled_indices = np.random.choice(len(chain), n_samples, replace=False)
        sampled_xs = chain[sampled_indices, :]
        from likelihood import run_simulation
        def get_sus(x):
            params = x_to_params(x, pathogen, lockdown, option1, option2, NAG=NAG)
            solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
            values = solution.ys.T[1:].reshape((2*N_S+1, NAG, -1))
            age_pop = jnp.sum(values[:2*N_S], axis=0)
            age_pop = age_pop.at[0,:].set(age_pop[0,:] + solution.ys.T[0,:])
            sus = susceptibility(solution, params=params)/age_pop.T
            # pre_sus = sus[(POINTS >= date_to_t('2015-10-01')) & (POINTS < date_to_t('2019-10-01'))].mean(axis=0)
            # mid_sus = sus[(POINTS >= date_to_t('2020-10-01')) & (POINTS < date_to_t('2021-10-01'))].mean(axis=0)
            # post_sus = sus[(POINTS >= date_to_t('2024-05-01')) & (POINTS < date_to_t('2025-05-01'))].mean(axis=0)
            pre_sus = sus[(POINTS == date_to_t('2017-07-01'))]
            mid_sus = sus[(POINTS == date_to_t('2021-07-01'))]
            post_sus = sus[(POINTS == date_to_t('2024-07-01'))]
            return jnp.stack([pre_sus, mid_sus, post_sus], axis=0)
        mapped_sus = jax.jit(jax.vmap(get_sus))
        suses = mapped_sus(sampled_xs)

        for j in range(n_samples):
            age_sus[i, :, 0, j] = suses[j, 0]
            age_sus[i, :, 1, j] = suses[j, 1]
            age_sus[i, :, 2, j] = suses[j, 2]

    # pre_median, pre_lower, pre_upper = np.percentile(age_sus[:, :, 0, :], [50, 2.5, 97.5], axis=-1)
    # mid_median, mid_lower, mid_upper = np.percentile(age_sus[:, :, 1, :], [50, 2.5, 97.5], axis=-1)
    # post_median, post_lower, post_upper = np.percentile(age_sus[:, :, 2, :], [50, 2.5, 97.5], axis=-1)
    medians_and_cis = np.percentile(age_sus, [50, 2.5, 97.5], axis=-1)
    return medians_and_cis

def plot_age_susceptibilty(ax, medians_and_cis):
    pre_median, pre_lower, pre_upper = medians_and_cis[:, :, 0]
    mid_median, mid_lower, mid_upper = medians_and_cis[:, :, 1]
    post_median, post_lower, post_upper = medians_and_cis[:, :, 2]
    offset = 0.2
    ax.scatter(np.arange(len(AGE_GROUP_NAMES))-offset, pre_median, color='k', label="2017-07-01", marker='s', s=3)
    ax.scatter(np.arange(len(AGE_GROUP_NAMES)), mid_median, color='hotpink', label="2021-07-01", marker='s', s=3)
    ax.scatter(np.arange(len(AGE_GROUP_NAMES))+offset, post_median, color='silver', label="2024-07-01", marker='s', s=3)
    for ag in range(len(AGE_GROUP_NAMES)):
        ax.plot([ag-offset, ag-offset], [pre_lower[ag], pre_upper[ag]], color='k', linewidth=0.7)
        ax.plot([ag, ag], [mid_lower[ag], mid_upper[ag]], color='hotpink', linewidth=0.7)
        ax.plot([ag+offset, ag+offset], [post_lower[ag], post_upper[ag]], color='silver', linewidth=0.7)
    ax.set_xticks(np.arange(len(AGE_GROUP_NAMES)))
    ax.set_xticklabels(AGE_GROUP_NAMES, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.grid(axis='x', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(axis='y', labelsize=6)
    ax.tick_params(axis='x', labelsize=6)

import matplotlib.transforms as mtransforms
def add_panel_label(ax, text, fig, x_fig=0.01, y_ax=1.1):
    """Adds a panel label aligned in Figure X-coords and Axis Y-coords."""
    trans = mtransforms.blended_transform_factory(fig.transFigure, ax.transAxes)
    ax.text(x_fig, y_ax, text, transform=trans, fontsize=16, fontweight="bold", va="bottom", ha="left")

def plot_age_figure(fig, pathogens, colors, option1, option2s, pruners, seeds, lockdown, NAG, CENSUS_AGE_POP, AGE_GROUP_NAMES, logD=False, age_adjusted=False, samples=100, prefix="", save_data=False, load_data=False):
    gs_main = fig.add_gridspec(3, 1, height_ratios=[2.3, 1.75, 1.0], hspace=0.05)
    # 1. Add a 4-column container: [Left Spacer, Heatmap Grid, Colorbar Column, Right Spacer]
    gs_top_container = gs_main[0].subgridspec(1, 4, width_ratios=[0.01, 0.9, 0.02, 0.01], wspace=0.12)    
    # 2. Heatmaps grid in column 1, Colorbar axis in column 2
    gs_top = gs_top_container[0, 1].subgridspec(2, 3, wspace=0.08, hspace=0.01)
    cax = fig.add_subplot(gs_top_container[0, 2])
    # 3. Section B spans the full figure width independently
    gs_mid = gs_main[1].subgridspec(2, 3, wspace=0.1, hspace=0.01)
    gs_bottom = gs_main[2].subgridspec(1, 8)
    import numpy as np
    ax_top = np.empty((2, 3), dtype=object)
    ax_mid = np.empty((2, 3), dtype=object)
    ax_mid[0, 0] = fig.add_subplot(gs_mid[0, 0])
    for r in range(2):
        for c in range(3):
            if r!=0:
                ax_top[r, c] = fig.add_subplot(gs_top[r, c], sharex=ax_top[0,c])
            else:
                ax_top[r, c] = fig.add_subplot(gs_top[r, c])
            ax_top[r, c].label_outer()
            if not (r == 0 and c == 0):
                ax_mid[r, c] = fig.add_subplot(gs_mid[r, c], sharex=ax_mid[0, 0], sharey=ax_mid[0, 0])
            ax_mid[r, c].label_outer()
    ax_bottom = np.empty((1,8), dtype=object)
    for c in range(8):
        ax_bottom[0, c] = fig.add_subplot(gs_bottom[c])
    if not load_data:
        matrices = {}
        hosp_matrices = {}
        sampled_matrices = {}
        sampled_hosp_matrices = {}
        for pi, (pathogen, option2, seed) in enumerate(zip(pathogens, option2s, seeds)):
            matrices[pathogen], hosp_matrices[pathogen] = get_infection_matrix(pathogen, seed, lockdown, option1, option2, NAG, CENSUS_AGE_POP, prune=pruners[pi], hospitalizations=False, prefix="emcee_")
            sampled_matrices[pathogen], sampled_hosp_matrices[pathogen] = get_infection_matrix(pathogen, seed, lockdown, option1, option2, NAG, CENSUS_AGE_POP, prune=pruners[pi], hospitalizations=False, samples=samples, prefix=prefix)
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
    full_medians_and_cis = get_age_susceptibility_over_time(pathogens, seeds, lockdown, option1, option2s, pruners, n_samples=samples)
    for pi, pathogen in enumerate(pathogens):
        normalized_age_infections_matrix = matrices[pathogen] / CENSUS_AGE_POP[:, None]
        plot_infection_matrix(ax_top[pi//3, pi%3], age_group_names=AGE_GROUP_NAMES, NAG=NAG, matrix=normalized_age_infections_matrix)
        ax_top[pi//3, pi%3].set_title(nice_names.get(pathogen, pathogen))
        ax_top[1, 1].set_xlabel("Transmitting age group", fontsize=9)
        plot_age_susceptibilty(ax_mid[pi//3, pi%3], full_medians_and_cis[:,pi])
        ax_mid[pi//3, pi%3].set_title(nice_names.get(pathogen, pathogen))
    ax_mid[0,0].set_ylim([-0.1,1.1])
    ax_mid[0,0].set_yticks([0,0.2,0.4,0.6,0.8,1])
    legend = ax_mid[0,-1].legend(fontsize=6,)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_alpha(1)
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_boxstyle('square,pad=0')

    # Create a standalone mappable with Viridis scaled from 0 to 1
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = colormaps.ScalarMappable(cmap="viridis", norm=norm)

    # Draw colorbar in cax
    cbar = fig.colorbar(sm, cax=cax)

    # Customization
    cbar.set_label("Population-weighted transmission", fontsize=9, labelpad=8)
    cbar.ax.tick_params(labelsize=8)
    
    for age_group_idx in range(NAG):
        if age_adjusted:
            pop_arg = CENSUS_AGE_POP
        else:            
            pop_arg = None
        plot_same_age_infection(ax_bottom[0, age_group_idx], [sampled_matrices[pathogen] for pathogen in pathogens], age_group_idx, pathogens, colors, pop_arg)
        if logD:
            ax_bottom[0, age_group_idx].set_yscale("log")
            ax_bottom[0, age_group_idx].set_ylim([5e-5,1])
            if age_group_idx == 0:
                ax_bottom[0, 0].yaxis.set_minor_locator(
                    ticker.LogLocator(
                        base=10.0, subs=np.arange(2, 10), numticks=100
                    )
                )
                ax_bottom[0, 0].tick_params(axis="y", which="minor", left=True)
            if age_group_idx > 0:
                ax_bottom[0, age_group_idx].spines['left'].set_visible(False)
                ax_bottom[0, age_group_idx].set_yticks([])
        ax_bottom[0, age_group_idx].set_xlabel(AGE_GROUP_NAMES[age_group_idx])

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
    ax_bottom[0,0].set_ylabel("Within-group\ntransmission", fontsize=9)

    fig.text(0.07, 0.785, "Contracting age group", fontsize=9, rotation=90, va="center", ha="center")
    fig.text(0.05, 0.39, "Population susceptibility", fontsize=9, rotation=90, va="center", ha="center")

    add_panel_label(ax_top[0, 0], "A", fig)
    add_panel_label(ax_mid[0, 0], "B", fig)
    add_panel_label(ax_bottom[0, 0], "C", fig)

def plot_age_profiles(ax, pathogens):
    return None

def plot_single_pathogen_violin(ax, pathogen_data, color, pathogen_name, left_annotation=None):
    """Plot violin for a single pathogen."""
    data_clean = pathogen_data.dropna()
    parts = ax.violinplot(data_clean, positions=[0], vert=True, showmedians=True)
    
    # Add scattered individual points with random vertical offset
    np.random.seed(260603)
    jitter = np.random.normal(0, 0.04, size=len(data_clean))
    print(f"Plotting {len(data_clean)} points for {pathogen_name}")
    ax.scatter(np.full(len(data_clean), 0) + jitter, data_clean, 
              color=color, s=20, alpha=4/np.sqrt(len(data_clean)), edgecolors='none')
    
    # Style violin body with a light grey fill and no outline
    for pc in parts['bodies']:
        pc.set_facecolor('lightgrey')
        pc.set_edgecolor('none')
        pc.set_alpha(0.6)
    
    # Reduce linewidth for range bar and caps
    for partname in ('cmedians', 'cbars', 'cmaxes', 'cmins'):
        if partname in parts:
            parts[partname].set_color('black')
            parts[partname].set_linewidth(0.8)
            
    # Reduce horizontal width of the caps and median bar
    cap_scale = 0.35
    for partname in ('cmedians', 'cmaxes', 'cmins'):
        if partname in parts:
            segments = parts[partname].get_segments()
            new_segments = []
            for seg in segments:
                center_x = (seg[0][0] + seg[1][0]) / 2.0
                seg[:, 0] = center_x + (seg[:, 0] - center_x) * cap_scale
                new_segments.append(seg)
            parts[partname].set_segments(new_segments)
    
    # Annotate median value on the right side
    median_val = data_clean.median()
    ax.text(0.25, median_val, f'{median_val:.1f}', 
            va='center', ha='center', fontsize=7, color='black')
    
    # Annotate second value on the left side in bold
    if left_annotation is not None:
        y_pos = left_annotation if isinstance(left_annotation, (int, float)) else median_val
        text_str = f'{left_annotation}' if isinstance(left_annotation, (int, float)) else str(left_annotation)
        ax.text(-0.15, y_pos, f'$\\mathbf{{{text_str}}}$', 
                va='center', ha='center', fontsize=7, color='black')
        # add a segment line on the central line of the violin, same size as the median annotation
        ax.scatter(0, y_pos, color='black', marker='D', s=5, zorder=5)
    
    # Turn off borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.set_xticks([])
    ax.set_xlabel(short_names.get(pathogen_name, pathogen_name))

    return data_clean.max() * 1.05

def suppression_violin(axes, pathogens, colors, left_annotations=None):
    """Plot violin plots for multiple pathogens in a grid."""
    suppression_data = pd.read_csv("Data/Processed/FluNet_suppression_duration_by_country_pathogen.csv")
    nice_names = {"InfluenzaA": "Influenza A", "InfluenzaB": "Influenza B"}
    xmax = 0
    for i, pathogen in enumerate(pathogens):
        pathogen_data = suppression_data[(suppression_data["pathogen"] == pathogen) & (suppression_data["dq_pass"] == True)]["suppression_duration_months"]
        # pathogen_name = nice_names.get(pathogen, pathogen)
        left_annotation = None if left_annotations is None else left_annotations[i]
        max_val = plot_single_pathogen_violin(axes[i], pathogen_data, colors[i], pathogen, left_annotation=left_annotation)
        xmax = max(xmax, max_val)
        # Only show y-axis label on leftmost plot
        if i > 0:
            axes[i].spines['left'].set_visible(False)
            axes[i].set_yticks([])
    for ax in axes:
        ax.set_ylim(0, xmax)
    axes[0].set_ylabel("Suppression duration (months)", fontsize=9)
    axes[0].set_yticks(np.arange(0, xmax+1, 12))

@jax.jit
def compute_kendall_w_single(data):
    """Computes Kendall's W for a single m x n matrix with NaNs."""
    m, n = data.shape
    valid = ~jnp.isnan(data)
    
    # Pairwise overlap mask: shape (m, m, n)
    M = valid[:, None, :] & valid[None, :, :]
    
    # Element-wise comparisons per rater: shape (m, n, n)
    diffs = data[:, :, None] - data[:, None, :]
    G = diffs > 0
    E = diffs == 0
    
    # Compute relative ranks using pairwise mask
    greater_count = jnp.sum(G[:, None, :, :] * M[:, :, None, :], axis=-1)
    equal_count = jnp.sum(E[:, None, :, :] * M[:, :, None, :], axis=-1)
    
    rank_x = greater_count + 0.5 * (equal_count - 1.0)
    rank_x = jnp.where(M, rank_x, 0.0)
    rank_y = jnp.swapaxes(rank_x, 0, 1)
    
    # Pairwise sample sizes and rank means
    N = jnp.sum(M, axis=-1)
    N_safe = jnp.where(N > 1, N, 1.0)
    
    mean_x = jnp.sum(rank_x, axis=-1) / N_safe
    mean_y = jnp.sum(rank_y, axis=-1) / N_safe
    
    # Mean-centered ranks and Pearson correlation
    dx = jnp.where(M, rank_x - mean_x[:, :, None], 0.0)
    dy = jnp.where(M, rank_y - mean_y[:, :, None], 0.0)
    
    cov = jnp.sum(dx * dy, axis=-1)
    var_x = jnp.sum(dx * dx, axis=-1)
    var_y = jnp.sum(dy * dy, axis=-1)
    
    denom = jnp.sqrt(var_x * var_y)
    denom_safe = jnp.where(denom > 0, denom, 1.0)
    corr = cov / denom_safe
    
    # Mean upper-triangle correlation
    triu_mask = jnp.triu(jnp.ones((m, m), dtype=bool), k=1)
    valid_pair = triu_mask & (N > 1) & (denom > 0)
    
    mean_r = jnp.nanmean(jnp.where(valid_pair, corr, jnp.nan))
    return (mean_r * (m - 1) + 1.0) / m

from functools import partial
@partial(jax.jit, static_argnames=('n_permutations',))
def run_permutation_test(key, data, n_permutations):
    m, n = data.shape
    valid_mask = ~jnp.isnan(data)
    
    # 1. Observed statistic
    obs_w = compute_kendall_w_single(data)
    
    # 2. Vectorized Permutation Generation across all K permutations
    keys = jax.random.uniform(key, shape=(n_permutations, m, n))
    keys = jnp.where(valid_mask[None, :, :], keys, 1e9)
    shuffle_idx = jnp.argsort(keys, axis=-1)
    
    valid_pos = jnp.argsort(~valid_mask, axis=-1)
    N_valid = jnp.sum(valid_mask, axis=-1, keepdims=True)
    col_idx = jnp.arange(n)
    valid_pos_safe = jnp.where(col_idx < N_valid, valid_pos, n)
    
    shuffled_vals = jnp.take_along_axis(data[None, :, :], shuffle_idx, axis=-1)
    
    out = jnp.full((n_permutations, m, n + 1), jnp.nan)
    out = out.at[:, jnp.arange(m)[:, None], valid_pos_safe].set(shuffled_vals)
    perm_data = out[:, :, :n]
    
    # 3. Parallel statistic calculation across all permutations via vmap
    perm_ws = jax.vmap(compute_kendall_w_single)(perm_data)
    
    p_value = jnp.mean(perm_ws >= obs_w)
    return obs_w, p_value

def test_concordance_with_nans(data, n_permutations=5000, seed=260728):
    """Fast JAX permutation test for Kendall's W with NaNs."""
    data_arr = jnp.array(data, dtype=jnp.float32)
    key = jax.random.PRNGKey(seed)
    obs_w, p_val = run_permutation_test(key, data_arr, n_permutations)
    return float(obs_w), float(p_val)

def plot_suppression_rank_heatmap(ax, pathogens, hemisphere='All', tropical="All", pvals=False):
    suppression_data = pd.read_csv("Data/Processed/FluNet_suppression_duration_by_country_pathogen.csv")
    if hemisphere!='All':
        country_hemisphere = pd.read_csv("Data/Processed/CountryHemisphere.csv", sep=';')
        hemisphere_countries = country_hemisphere[country_hemisphere["Hemisphere"]==hemisphere]["Country"].values
        suppression_data = suppression_data[suppression_data["country"].isin(hemisphere_countries)]
    if tropical!="All":
        country_tropical = pd.read_csv("Data/Processed/CountryHemisphere.csv", sep=';')
        tropical_countries = country_tropical[country_tropical["Tropical"]==tropical]["Country"].values
        suppression_data = suppression_data[suppression_data["country"].isin(tropical_countries)]
    # filter to dq_pass == True
    suppression_data = suppression_data[suppression_data["dq_pass"] == True]
    # for each country, find the rank of each pathogen by suppression_duration_months
    suppression_data["rank"] = suppression_data.groupby("country")["suppression_duration_months"].rank(method="min", ascending=False)
    # create a pivot table with index country, columns pathogen, values rank
    pivot = suppression_data.pivot(index="country", columns="pathogen", values="rank")
    n_permutations = 10000
    w_stat, p_val = test_concordance_with_nans(pivot.values, n_permutations=n_permutations)

    print(f"Kendall's W (Concordance): {w_stat:.4f}")
    if p_val == 0.0:
        print(f"Permutation p-value:       < {1/n_permutations:.0e} (based on {n_permutations} permutations)")
    else:
        print(f"Permutation p-value:       {p_val}")
    # plot heatmap of how often one pathogen is ranked higher than the other
    image = np.zeros((len(pathogens), len(pathogens)))
    p_values = np.zeros((len(pathogens), len(pathogens)))
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
                # find p-value using binomial test
                p_values[i, j] = sp.stats.binomtest(count_i_higher, total_count, p=0.5, alternative='two-sided').pvalue
                p_values[j, i] = sp.stats.binomtest(count_j_higher, total_count, p=0.5, alternative='two-sided').pvalue
            else:
                image[i, j] = np.nan
                image[j, i] = np.nan
                p_values[i, j] = np.nan
                p_values[j, i] = np.nan
    # reorder the pathogens by average rank
    order = ["Adenovirus", "Parainfluenza", "RSV",  "Metapneumovirus", "InfluenzaA", "InfluenzaB"]
    image = image[[pathogens.index(p) for p in order], :][:, [pathogens.index(p) for p in order]]
    # only plot lower triangle
    for i in range(len(order)):
        for j in range(i+1, len(order)):
            image[i, j] = np.nan
            p_values[i, j] = np.nan
    np.fill_diagonal(image, np.nan)
    image = image[1:, :-1]
    np.fill_diagonal(p_values, np.nan)
    p_values = p_values[1:, :-1]
    print(image)
    #annotate with percentages
    if pvals:
        # annotate with p-values
        for i in range(len(order)-1):
            for j in range(i, len(order)-1):
                if not np.isnan(image[j, i]):
                    ax.text(i, j, f"{p_values[j, i]:.2e}", ha="center", va="center", color="white", fontsize=5)
    else:
        for i in range(len(order)-1):
            for j in range(i, len(order)-1):
                if not np.isnan(image[j, i]):
                    ax.text(i, j, f"{image[j, i]*100:.0f}%", ha="center", va="center", color="white", fontsize=6)
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
    suppression_data = pd.read_csv("Data/Processed/FluNet_suppression_duration_by_country_pathogen.csv")
    def sanitize_filename_token(value: str) -> str:
        token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        token = token.strip("._-")
        return token or "unknown"
    # apply to country names
    suppression_data.loc[:, "country"] = suppression_data["country"].apply(sanitize_filename_token)
    # restrict to month_start >= 2015-10-01
    data = data[(data["month_start"] >= pd.to_datetime("2015-10-01")) & (data["month_start"] < pd.to_datetime("2026-05-01"))]
    if (len(data) > 0) and (data["count"].sum() > 0):
        ax.plot(data["month_start"], data["count"], label=country, color=color, linewidth=linewidth)
    else:
        years = pd.date_range(start=pd.to_datetime("2015-10-01"), end=pd.to_datetime("2026-05-01"), freq='MS')
        ax.plot(years, [0]*len(years), label=country, color=color, linewidth=linewidth)
        if flag_exclusions:
            ax.annotate("all_zero", xy=(0.01, 0.99), xycoords='axes fraction', fontsize=6, color='magenta', ha='left', va='top')
    if flag_exclusions and ((suppression_data["pathogen"] == pathogen) & (suppression_data["country"] == country) & (suppression_data["status"] == "excluded")).any():
        reason = suppression_data[(suppression_data["pathogen"] == pathogen) & (suppression_data["country"] == country) & (suppression_data["status"] == "excluded")]["reason"].iloc[0]
        short_reason = short_reasons.get(reason, reason)
        ax.annotate(short_reason, xy=(0.01, 0.99), xycoords='axes fraction', fontsize=6, color='magenta', ha='left', va='top')

    # Add suppression period indicator
    pre_2020_data = data[data["month_start"] < pd.to_datetime("2020-03-01")]
    if len(pre_2020_data) > 0 and not ((suppression_data["pathogen"] == pathogen) & (suppression_data["country"] == country) & (suppression_data["status"] == "excluded")).any():
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

def plot_toy_model(ax, colors, flulike=False):
    from sim_grid import suppression_duration_single_series
    PERIOD = pd.date_range(start=pd.to_datetime('2015-07-04'), end=pd.to_datetime('2025-05-01'), freq='W')
    if flulike:
        traj = np.genfromtxt("Data/Processed/favourite_fluonly_clustered_trajectories_scaled2.csv")
    else:
        traj = np.genfromtxt("Data/Processed/favourite_nonflulike_samefr_clustered_trajectories.csv")
    for i in range(traj.shape[0]):
        duration = str(int(suppression_duration_single_series(traj[i,:],anchor_idx=260)/4.35))
        ax.plot(PERIOD[:-1],traj[i,:], color=colors[i], label=f"{duration} months")
    ax.set_yticks([])
    ax.set_xticks(pd.to_datetime(["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01",]))
    ax.set_xticklabels(["", "2017", "", "2019", "", "2021", "", "2023", "", "2025",], fontsize=8)
    if flulike:
        ax.legend(fontsize=6, title="Suppression duration", title_fontsize=7, frameon=False)
    if not flulike:
        ax.legend(fontsize=6, title="Suppression duration", title_fontsize=7, frameon=False, loc=(0.1, 0.6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# toy model in top row, then violins and suppression heatmap in lower row
# violins need more horizontal space than suppression heatmap, so need unevengrid
def plot_alternative_suppression(fig, pathogens, colors):
    # Main grid for top and bottom rows
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.13, 0.9])

    # Top row: 2 perfectly symmetric subplots
    gs_top = gs[0].subgridspec(1, 2)
    toy_ax1 = fig.add_subplot(gs_top[0])
    toy_ax2 = fig.add_subplot(gs_top[1])

    plot_toy_model(toy_ax2, ["#DC267F", "#FFB000", "#648FFF", "#785EF0"], flulike=False)
    plot_toy_model(toy_ax1, ["#DC267F", "#FFB000", "#648FFF"], flulike=True)
    toy_ax1.set_ylabel("Incidence")
    # toy_ax2.set_title(" ")

    # Bottom row: split into left (violin plots) and right (heatmap)
    n_pathogens = len(pathogens)
    gs_bottom = gs[2].subgridspec(1, 2, width_ratios=[1.5, 1], wspace=2)

    # Left figure: violin axes span full vertical height of bottom row
    gs_violins = gs_bottom[0].subgridspec(1, n_pathogens)
    violin_axes = np.empty(n_pathogens, dtype=object)
    for i in range(n_pathogens):
        violin_axes[i] = fig.add_subplot(gs_violins[i])
    suppression_violin(violin_axes, pathogens, colors, left_annotations=[17,20,16,4,32,61])

    gs_heatmap = gs_bottom[1].subgridspec(1, 1)
    heatmap_ax = fig.add_subplot(gs_heatmap[0])
    im = plot_suppression_rank_heatmap(heatmap_ax, pathogens)
    heatmap_ax.set_ylabel("Re-emerging after")
    heatmap_ax.set_xlabel("Re-emerging before")
    cbar = fig.colorbar(im, ax=heatmap_ax)
    cbar.set_label(
        "Frequency", 
        fontsize=8
    )

    # # Panel labels A B C
    fig.text(0.035, 0.98, "A", fontsize=16, fontweight="bold", va="top", ha="left")
    fig.text(0.525, 0.98, "B", fontsize=16, fontweight="bold", va="top", ha="left")
    fig.text(0.01, 0.5, "C", fontsize=16, fontweight="bold", va="top", ha="left")
    fig.text(0.64, 0.5, "D", fontsize=16, fontweight="bold", va="top", ha="left")


def plot_suppression_durations(fig, pathogens, countries, colors):
    gs = fig.add_gridspec(9, 6, height_ratios=[1, 1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 3])
    
    # KPSC plots in top row (row 0)
    kpsc_pathogens = ["RSV","Metapneumovirus","Parainfluenza3","Adenovirus","InfluenzaA","InfluenzaB",]
    kpsc_axis = np.empty(len(kpsc_pathogens), dtype=object)
    # season age plots in row 1
    season_age_axes = np.empty(len(kpsc_pathogens), dtype=object)
    for i, pathogen in enumerate(kpsc_pathogens):
        incax = fig.add_subplot(gs[0, i])
        kpsc_axis[i] = incax
        # color = colors[i]
        last_pre_time, rebound_time = kpsc_proportion_positive_incidence_plot(
            incax, pathogen=pathogen, AGE_GROUP_NAMES=AGE_GROUP_NAMES, title="",
            color="k", aggregation="MS", factor=100000,
            annotations="simple", definition="", label="Data", hosp=True, dedup=True)
        # shade pink for period of suppression
        incax.axvspan(last_pre_time, rebound_time, color='red', alpha=0.1, linewidth=0)
        incax.set_yticklabels([])
        incax.set_xticklabels([])
        incax.set_xlim(pd.to_datetime("2015-05-01"), pd.to_datetime("2025-10-01"))
        # Remove all spines except bottom
        incax.spines['top'].set_visible(False)
        incax.spines['right'].set_visible(False)
        incax.spines['left'].set_visible(False)
        incax.set_ylabel("")
        incax.set_yticks([])
        if i==0:
            incax.set_ylabel("KPSC", rotation=0, ha='right', labelpad=25)
            incax2 = incax.twinx()
            incax2.set_ylabel("Estimated\n+ve hospitalizations", rotation=90, va='center', fontsize=6)
            incax2.set_yticks([])
            incax2.spines['right'].set_visible(False)
            incax2.spines['left'].set_visible(True)
            incax2.spines['top'].set_visible(False)
            incax2.yaxis.set_label_position('left')
            incax2.yaxis.tick_left()

        incax.set_title(short_names.get(pathogen, pathogen))
        incax.set_xticks(pd.to_datetime(["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01",]))
        incax.set_xticklabels(["", "2017", "", "2019", "", "2021", "", "2023", "", "2025",], fontsize=6)
        for label in incax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
        
        ageax = fig.add_subplot(gs[1, i])
        season_age_axes[i] = ageax
        plot_relative_age_incidence(ageax, pathogen, group_names=["<1y","<5y"], aggregation="YS-OCT")
        ageax.set_xticklabels(["", "2016/17", "", "2018/19", "", "2020/21", "", "2022/23", "", "2024/25",], fontsize=6)
        ageax.set_xlim(pd.to_datetime("2015-05-01"), pd.to_datetime("2025-10-01"))
        for label in ageax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
        ageax.axvspan(last_pre_time, rebound_time, color='red', alpha=0.1, linewidth=0)
        # add grid
        ageax.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.3)
        ageax.grid(axis='x', linestyle='-', linewidth=0.5, alpha=0.3)
        ageax.set_ylim(0.0, 0.5)
        ageax.set_yticks([0,0.1,0.2,0.3,0.4,0.5])
        ageax.set_yticklabels(["0.0","","","","","0.5"], fontsize=6)
        if i>0:
            ageax.set_yticklabels(["","","","","",""], fontsize=6)
    season_age_axes[0].set_ylabel("KPSC", rotation=0, ha='right', labelpad=16)
    sax2 = season_age_axes[0].twinx()
    sax2.set_ylabel("Proportion of\nhospitalizations", rotation=90, va='center', fontsize=6, labelpad=16)
    sax2.set_yticks([])
    sax2.spines['right'].set_visible(False)
    sax2.spines['left'].set_visible(True)
    sax2.spines['top'].set_visible(False)
    sax2.yaxis.set_label_position('left')
    sax2.yaxis.tick_left()
    # put legend in first plot
    legend = season_age_axes[-1].legend(loc='best', fontsize=6)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_alpha(1)
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_boxstyle('square,pad=0')

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
    suppression_violin(violin_axis, pathogens, colors)
    
    # big label "C" on top left of kpsc plots, "B" on top left of country plots, "A" on top left of violin plots
    kpsc_axis[0].text(-0.5, 1.1, "A", transform=kpsc_axis[0].transAxes, fontsize=16, fontweight="bold")
    season_age_axes[0].text(-0.5, 1.1, "B", transform=season_age_axes[0].transAxes, fontsize=16, fontweight="bold")
    country_axes[0,0].text(-0.5, 1.1, "C", transform=country_axes[0,0].transAxes, fontsize=16, fontweight="bold")
    violin_axis[0].text(-0.5, 1.1, "D", transform=violin_axis[0].transAxes, fontsize=16, fontweight="bold")

nice_age_group_names = ['<3m', '3–11m', '1–4y', '5–17y', '18–49y', '50–64y', '≥65y']
def plot_fits(axes, n_samples=100, save_data=False, load_data=False, colors=None):
    aggregation = "MS"
    agg_factor = 30.44
    factor = 100000
    for pathogen_idx in range(len(pathogens)):
        ax = axes[:,pathogen_idx]
        pathogen, option2, seed = pathogens[pathogen_idx], option2s[pathogen_idx], seeds[pathogen_idx]
        cs = [colors[pathogen_idx]]*NAG if colors is not None else hsv_colors
        _, _, _, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=1e-9, incidence_data=False, hosp=True, NAG=NAG, dedup=True)
        chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=pruners[pathogen_idx], prefix="")
        np.random.seed(260604)
        # choose n_samples random rows from chain
        random_indices = np.random.choice(chain.shape[0], size=n_samples, replace=False)
        random_samples = chain[random_indices, :]
        if not load_data:
            from likelihood import run_simulation
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
                                        color=cs[age_group_idx], factor=factor*agg_factor, label="Simulation", linewidth=0.5, alpha=0.01)
        for age_group_idx, age_group_name in enumerate(AGE_GROUP_NAMES):
            kpsc_proportion_positive_incidence_plot(ax[age_group_idx], pathogen=pathogen,
                                                    AGE_GROUPS=AGE_GROUPS, AGE_GROUP_NAMES=AGE_GROUP_NAMES, select_age_group=age_group_idx,
                                                    aggregation=aggregation, factor=factor, annotations=False, definition="", label="Data", hosp=True, dedup=True,
                                                    title="", color="k", linewidth=0.5)
            if pathogen_idx == 0:
                ax[age_group_idx].set_ylabel(nice_age_group_names[age_group_idx])
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

def plot_fits_and_sus(axes, n_samples=100):
    aggregation = "MS"
    agg_factor = 30.44
    factor = 100000
    for pathogen_idx in range(len(pathogens)):
        ax = axes.flatten()[pathogen_idx]
        pathogen, option2, seed = pathogens[pathogen_idx], option2s[pathogen_idx], seeds[pathogen_idx]
        maxinc = kpsc_proportion_positive_incidence_plot(ax, pathogen=pathogen,
                                                aggregation=aggregation, factor=factor, annotations=False, definition="", label="Data", hosp=True, dedup=True,
                                                title="", color="k", linewidth=1, zorder=3)
        _, _, _, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=1e-9, incidence_data=False, hosp=True, NAG=NAG, dedup=True)
        chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=pruners[pathogen_idx], prefix="")
        np.random.seed(260604)
        # choose n_samples random rows from chain
        random_indices = np.random.choice(chain.shape[0], size=n_samples, replace=False)
        random_samples = chain[random_indices, :]
        from likelihood import run_simulation
        def get_solution(x):
            params = x_to_params(x, pathogen, lockdown, option1, option2, NAG=NAG)
            solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
            return solution
        solutions = jax.jit(jax.vmap(get_solution))(random_samples)
        first_solution = jax.tree_util.tree_map(lambda x: x[0], solutions)
        population_size = calculate_population_size(first_solution.ys.T, NAG=NAG)
        maxis = np.zeros(n_samples)
        for sample_i in range(n_samples):
            print(f"Pathogen {pathogen_idx+1}/{len(pathogens)}: {pathogen}, sample {sample_i+1}/{n_samples}", end="\r")
            solution = jax.tree_util.tree_map(lambda x: x[sample_i], solutions)
            smaxi = lockdown_incidence_plot(ax, STATE0, None, POINTS, pd.to_datetime('2020-03-01'),
                                    solution=solution,
                                    by_age=False,
                                    p_time_to_obs=p_time_to_obs, NAG=NAG,
                                    test_data=data_full,daily_hospitalization_rates=daily_hospitalization_rates,aggregation=aggregation,
                                    uncertainty="draw",
                                    color=colors[pathogen_idx], factor=factor*agg_factor, label="Simulation", linewidth=0.5, alpha=0.01)
            maxis[sample_i] = smaxi
        maxi = np.percentile(maxis, 95)
        print("\n",end="\r")
        mymax = np.maximum(maxi, maxinc)
        for sample_i in range(n_samples):
            params = x_to_params(random_samples[sample_i], pathogen, lockdown, option1, option2, NAG=NAG)
            print(f"Pathogen {pathogen_idx+1}/{len(pathogens)}: {pathogen}, sample {sample_i+1}/{n_samples}", end="\r")
            solution = jax.tree_util.tree_map(lambda x: x[sample_i], solutions)
            sus = susceptibility(solution,params,NAG=NAG).sum(axis=1) / population_size.sum(axis=1)
            sus = mymax*(1.18 + (sus - sus.mean())*1.7)
            ax.plot(PERIOD, sus, color='grey', linewidth=0.5, alpha=0.01)
        print("\n",end="\r")
        ax.set_title(nice_names.get(pathogen, pathogen), fontsize=8)
        ax.set_xticks(pd.to_datetime(["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01",]))
        ax.set_xticklabels(["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025",], fontsize=6)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
        ax.legend().set_visible(False)
        ax.tick_params(axis='y', labelsize=6)
        ax.set_ylabel("")

def plot_contact(ax, pathogen, option2, prune, seed, n_samples=100):
    chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=prune, prefix="")
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

def plot_susceptibility(ax, pathogen, option2, prune, seed, n_samples=100, load_data=False):
    chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=prune, prefix="")
    _, _, _, p_time_to_obs, data_full = pathogen_parameters(pathogen, import_multiplier=1e-9, incidence_data=False, hosp=True, NAG=NAG, dedup=True)
    chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=prune, prefix="")
    np.random.seed(260604)
    # choose n_samples random rows from chain
    random_indices = np.random.choice(chain.shape[0], size=n_samples, replace=False)
    random_samples = chain[random_indices, :]
    if not load_data:
        from likelihood import run_simulation
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

def plot_r0_vs_first_immunity(axes, pathogen, option2, seed, color, r0_base=15.24, prune=0, n_samples=10000):
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
    axes.scatter(r0_values, immunity_values, alpha=1, color=color, s=np.sqrt(1/n_samples)/4, linewidths=0, rasterized=True)
    return r0_values, immunity_values

def plot_r0_vs_population_immunity(axes, pathogen, option2, seed, color, r0_base=15.24, prune=0, n_samples=400, summary="mean"):
    chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=prune, prefix="")
    np.random.seed(260604)
    # choose n_samples random rows from chain
    random_indices = np.random.choice(chain.shape[0], size=n_samples, replace=False)
    random_samples = chain[random_indices, :]
    from likelihood import run_simulation
    from utils import susceptibility
    t_key = date_to_t('2020-01-01')
    def run_sims(x):
        params = x_to_params(x, pathogen, lockdown, option1, option2, NAG=NAG)
        solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
        values = solution.ys.T
        pop_size_by_age = calculate_population_size(values, NAG=NAG)

        shaped_solution = solution.ys.T[1:, :].reshape((2*N_S+1, NAG, -1))
        susceptible = shaped_solution[0:2*N_S:2, :, :]

        S_REL = params[6]
        # avg_srel = jnp.sum(S_REL[:, None, None] * susceptible)/jnp.sum(susceptible)
        avg_srel = jnp.sum(S_REL[:, None, None] * susceptible, axis=(0,1))/jnp.sum(pop_size_by_age,axis=1)
        if summary == "mean":
            summary_srel = jnp.mean(avg_srel[:t_key])
        elif summary == "min":
            summary_srel = jnp.min(avg_srel[:t_key])
        elif summary == "max":
            summary_srel = jnp.max(avg_srel[:t_key])
        return summary_srel
    weighted_pre2020_srel_samples = jax.jit(jax.vmap(run_sims))(random_samples)
    r0_values = r0_base * random_samples[:, 0] * (3.0 + 1.9 * (pathogen in ["RSV", "Metapneumovirus"]))
    immunity_values = 1 - weighted_pre2020_srel_samples
    axes.scatter(r0_values, immunity_values, alpha=1, color=color, s=np.sqrt(1/n_samples), linewidths=0, rasterized=True)
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

def plot_heatmaps_and_best_fit(fig, pathogens, option2s, pruners, seeds, colors, run_save_path, good_simulations, r0_base, fit_line=True):
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.6)
    fit_ax = fig.add_subplot(gs[0, :])
    axes_hm = [fig.add_subplot(gs[1, i]) for i in range(3)]
    generate_2d_heatmap_plot(axes_hm[0], run_save_path, good_simulations, NAG=NAG, p1=0, p2=8, outcome="suppression_length", cbar=True, r0_base=r0_base)
    generate_2d_heatmap_plot(axes_hm[1], run_save_path, good_simulations, NAG=NAG, p1=0, p2=8, outcome="infectors_in_group_0123", cbar=True, r0_base=r0_base)
    generate_2d_heatmap_plot(axes_hm[2], run_save_path, good_simulations, NAG=NAG, p1=0, p2=8, outcome="excess_susceptibility", cbar=True, r0_base=r0_base)
    # generate_2d_heatmap_plot(axes_hm[2], run_save_path, good_simulations, NAG=NAG, p1=0, p2=8, outcome="infectors_in_group_6", cbar=True, r0_base=r0_base)
    for ax in axes_hm:
        ax.set_xscale('log')
        ax.set_xticks([1,2,3,4,5,6,7])
        ax.set_xticklabels([1,2,3,4,5,6,7])
    axes_hm[1].set_ylabel("")
    axes_hm[1].set_yticklabels([])
    axes_hm[2].set_ylabel("")
    axes_hm[2].set_yticklabels([])
    axes_hm[0].set_xlabel("")
    axes_hm[1].set_xlabel("Basic reproduction number (log scale)")
    axes_hm[2].set_xlabel("")
    axes_hm[0].set_title("Suppression duration\n(months)", fontsize=8)
    axes_hm[1].set_title("Proportion of infections\nfrom under 18s", fontsize=8)
    axes_hm[2].set_title("Maximum excess\nsusceptibility", fontsize=8)

    colorbar_axes = [c for c in fig.axes if c not in [axes_hm[0], axes_hm[1], axes_hm[2], fit_ax]]
    if len(colorbar_axes) >= 3:
        colorbar_axes[0].set_ylabel("")
        colorbar_axes[1].set_ylabel("")
        colorbar_axes[2].set_ylabel("")

    r0_values_by_pathogen = {}
    immunity_values_by_pathogen = {}
    for pathogen, option2, prune, seed, color in zip(pathogens, option2s, pruners, seeds, colors):
        print(f"plotting parameter scatter for {pathogen}...")
        r0_values, immunity_values = plot_r0_vs_first_immunity(fit_ax, pathogen, option2, seed, color, prune=prune)
        r0_values_by_pathogen[pathogen] = r0_values
        immunity_values_by_pathogen[pathogen] = immunity_values
    if fit_line:
        plot_odr_best_fit(fit_ax, r0_values_by_pathogen, immunity_values_by_pathogen, n_iterations=10000)
    else:
        x = np.linspace(1, 7, 100)
        y = 1 - 1 / x
        fit_ax.plot(x, y, color='k', linewidth=0.5, zorder=0)
        # label with "1 - 1/R0" in the middle of the line
        fit_ax.text(3.5, 1-1/3.5, r"$\iota_1 = 1 - \frac{1}{R_0}$", rotation=0, fontsize=11, color='k', ha='right', va='bottom', zorder=1)
    # build custom legend with colored squares
    handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=8) for color in colors]
    labels = [nice_names.get(pathogen, pathogen) for pathogen in pathogens]
    fit_ax.legend(handles, labels, frameon=False, fontsize=6, loc='lower right', ncol=2)
    fit_ax.set_xscale('log')
    fit_ax.set_xticks([1,2,3,4,5,6,7])
    fit_ax.set_xticklabels([1,2,3,4,5,6,7])
    fit_ax.set_xlim(1, 7)
    fit_ax.set_ylim(0, 0.95)
    fit_ax.set_xlabel("Basic reproduction number (log scale)")
    fit_ax.set_ylabel("Immunity from first infection")

    # add big letter "A" to first row, "B" to second row
    axes_hm[0].text(-0.5, 1.1, "B", transform=axes_hm[0].transAxes, fontsize=16, fontweight="bold")
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
math_names = {
    "BETA": r"$\beta$",
    "SEASONALITY": r"$A$",
    "OFFSET": r"$\theta$",
    "WANE2": r"$w_{2}$",
    "S_REL1": r"$\sigma_{2}$",
    "S_REL2": r"$\sigma_{3} / \sigma_{2}$",
    "D_REL1": r"$\phi_{2}$",
    "D_REL2": r"$\phi_{3} / \phi_{2}$",
    "EXTRA_IMMUNITY": r"$x$",
    "FIRST_IMMUNITY": r"$y$",
    "FIRST_DIS_INF_FACTOR": r"$z$",
    "F1": r"$f$",
    "R1": r"$r$",
    "AGE_OBS_1": r"$\psi_{\text{<3m}}$",
    "AGE_OBS_2": r"$\psi_{\text{3–11m}}$",
    "AGE_OBS_3": r"$\psi_{\text{1–4y}}$",
    "AGE_OBS_4": r"$\psi_{\text{5–17y}}$",
    "AGE_OBS_5": r"$\psi_{\text{18–39y}}$",
    "AGE_OBS_6": r"$\psi_{\text{40–64y}}$",
    "AGE_OBS_7": r"$\psi_{\text{≥65y}}$",
}
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
    param_math_names = [math_names[name] for name in select_param_names]
    sns.heatmap(corr_diff, xticklabels=param_math_names, yticklabels=param_math_names, center=0, cmap="bwr", ax=ax, vmin=-1.05, vmax=1.05)
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
    return (frob_norm2 - frob_norm1) / frob_norm1

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
        chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=pruners[pathogen_idx], prefix="")
        np.random.seed(260604)
        # choose n_samples random rows from chain
        random_indices = np.random.choice(chain.shape[0], size=n_samples, replace=False)
        random_samples = chain[random_indices, :]
        from likelihood import run_simulation
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

def print_parameter_table(pathogens, option2s, seeds, pruners=None):
    param_table = pd.DataFrame(columns=["Pathogen", "Value"])

    if pruners is None:
        pruners = [0] * len(pathogens)
    
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
    
    for pathogen, option2, seed, prune in zip(pathogens, option2s, seeds, pruners):
        chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=prune, prefix="")
        consistent_xs = jax.vmap(lambda x: consistent_x_from_DE(pathogen, lockdown, option1, option2, seed, NAG=NAG, x_DE=x))(chain)
        print(consistent_xs.shape)


        R0 = consistent_xs[:, 2] * r0_base / consistent_xs[:, 0]
        print(np.median(R0), np.percentile(R0, 2.5), np.percentile(R0, 97.5))


        SREL2 = consistent_xs[:, 8] * consistent_xs[:, 7]
        print(np.median(SREL2), np.percentile(SREL2, 2.5), np.percentile(SREL2, 97.5))


        DREL2 = consistent_xs[:, 10] * consistent_xs[:, 9]
        print(np.median(DREL2), np.percentile(DREL2, 2.5), np.percentile(DREL2, 97.5))
        
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
    from likelihood import run_simulation
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
    pruners = [100, 3000, 2000, 2200, 100, 100,]

    # ==========================================
    # Figure 1: Toy Model and FluNet Figure
    # ==========================================
    fig = plt.figure(figsize=(6.5, 4), constrained_layout=True)
    plot_alternative_suppression(fig, flunet_pathogens, colors)
    plt.savefig("Figures/Figure1.svg", dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # Figure 2: Fits and Susceptibility Figure
    # ==========================================
    sample_size = 400
    fig = plt.figure(figsize=(6.5,7), layout="constrained")
    subfigs = fig.subfigures(4, 1, height_ratios=[2.3, 5, 1.15, 0.1])

    axes_A = subfigs[0].subplots(2, 3, sharex=True)
    plot_fits_and_sus(axes_A, sample_size)
    subfigs[0].supylabel("Monthly incidence\n(per 100,000)", fontsize=8, x=0.03, va='center', ha='center')

    axes_B = subfigs[1].subplots(7, 6, sharex=False, sharey=False)
    plot_fits(axes_B, n_samples=sample_size, load_data=False, save_data=True, colors=colors)
    for ax in axes_B[6, :]:
        ax.set_xticklabels([])

    axes_C = subfigs[2].subplots(1, 6, sharex=False, sharey=False)
    for pathogen, option2, prune, seed, ax in zip(pathogens, option2s, pruners, seeds, axes_C):
        plot_susceptibility(ax, pathogen, option2, prune, seed, n_samples=sample_size)
        ax.set_xticks(pd.to_datetime([
            "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", 
            "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"
        ]))
        ax.set_xticklabels(["", "2017", "", "2019", "", "2021", "", "2023", "", "2025"], fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.set_ylim(0.5, 1.5)
        ax.set_yticks([0.5, 1.0, 1.5])
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
    axes_C[0].set_ylabel("Relative\nsusceptibility", fontsize=6)
    legend_handles = [
        Patch(facecolor=color, label=name) 
        for name, color in zip(nice_age_group_names, hsv_colors)
    ]
    subfigs[3].legend(
        handles=legend_handles,
        loc="center",
        ncol=len(nice_age_group_names),
        fontsize=6,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2
    )
    # label panels A, B, C
    axes_A[0, 0].text(-0.35, 1.15, "A", transform=axes_A[0, 0].transAxes, fontsize=16, fontweight="bold")
    axes_B[0, 0].text(-0.7, 1.15, "B", transform=axes_B[0, 0].transAxes, fontsize=16, fontweight="bold")
    axes_C[0].text(-0.7, 1.15, "C", transform=axes_C[0].transAxes, fontsize=16, fontweight="bold")

    plt.savefig("Figures/Figure2.svg", dpi=500, bbox_inches='tight')
    plt.close()


    # ==========================================
    # Figure 3: WAIFW and age susceptibility
    # ==========================================
    fig = plt.figure(figsize=(5, 7), layout="constrained")
    plot_age_figure(fig, pathogens, colors, option1, option2s, pruners, seeds, lockdown, NAG, CENSUS_AGE_POP, AGE_GROUP_NAMES, age_adjusted=False, logD=True, samples=400, load_data=True, prefix="")
    plt.savefig(f"Figures/Figure3.svg", dpi=300)
    plt.close()

    # ==========================================
    # Figure 4: HIT and heat-blobs
    # ==========================================
    from sim_grid import generate_2d_heatmap_plot
    good_simulations = [[pathogen, seed, lockdown, option1, option2, prune] for pathogen, seed, option2, prune in zip(pathogens, seeds, option2s, pruners)]
    run_save_path = "Outputs/sim_grid_lh_n80000_chunk10000_seed260728_lockdownExponentialODipp25_2d"
    fig = plt.figure(figsize=(5.25, 4.5))
    plot_heatmaps_and_best_fit(fig, pathogens, option2s, pruners, seeds, colors, run_save_path, good_simulations, r0_base, fit_line=False)
    plt.savefig(f"Figures/.svg", dpi=1000)
    plt.close()

    # fig, axes = plt.subplots(1, 6, figsize=(6.5, 1.5), constrained_layout=True)
    # sample_size = 400
    # for pathogen, option2, prune, seed, ax in zip(pathogens, option2s, pruners, seeds, axes):
    #     plot_contact(ax, pathogen, option2, prune, seed, n_samples=sample_size)
    #     ax.set_title(short_names.get(pathogen, pathogen))
    #     ax.set_ylim(0, 1.2)
    #     ax.set_xticks(pd.to_datetime(["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01",]))
    #     ax.set_xticklabels([])
    #     ax.tick_params(axis='y', labelsize=6)
    #     ax.set_xticklabels(["", "2017", "", "2019", "", "2021", "", "2023", "", "2025",], fontsize=6)
    #     for label in ax.get_xticklabels():
    #         label.set_rotation(45)
    #         label.set_horizontalalignment('right')
    #         label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
    #     axes[0].set_ylabel("Contact\nmultiplier", fontsize=6)
    # plt.savefig(f"Figures/Contact_multipliers_{lockdown}_{option1}.pdf", dpi=300, bbox_inches='tight')



    ##### KPSC and age incidence figure now in supplementary
    # fig = plt.figure(figsize=(6.5, 2.5), constrained_layout=True)
    # gs = fig.add_gridspec(2, 6, height_ratios=[1, 1])
    # # KPSC plots in top row (row 0)
    # kpsc_pathogens = ["RSV","Metapneumovirus","Parainfluenza3","Adenovirus","InfluenzaA","InfluenzaB",]
    # kpsc_axis = np.empty(len(kpsc_pathogens), dtype=object)
    # # season age plots in row 1
    # season_age_axes = np.empty(len(kpsc_pathogens), dtype=object)
    # for i, pathogen in enumerate(kpsc_pathogens):
    #     incax = fig.add_subplot(gs[0, i])
    #     kpsc_axis[i] = incax
    #     # color = colors[i]
    #     last_pre_time, rebound_time = kpsc_proportion_positive_incidence_plot(
    #         incax, pathogen=pathogen, AGE_GROUP_NAMES=AGE_GROUP_NAMES, title="",
    #         color="k", aggregation="MS", factor=100000,
    #         annotations="simple", definition="", label="Data", hosp=True, dedup=True)
    #     # shade pink for period of suppression
    #     incax.axvspan(last_pre_time, rebound_time, color='red', alpha=0.1, linewidth=0)
    #     incax.set_yticklabels([])
    #     incax.set_xticklabels([])
    #     incax.set_xlim(pd.to_datetime("2015-05-01"), pd.to_datetime("2025-10-01"))
    #     # Remove all spines except bottom
    #     incax.spines['top'].set_visible(False)
    #     incax.spines['right'].set_visible(False)
    #     incax.spines['left'].set_visible(False)
    #     incax.set_ylabel("")
    #     incax.set_yticks([])
    #     if i==0:
    #         incax.set_ylabel("KPSC", rotation=0, ha='right', labelpad=25)
    #         incax2 = incax.twinx()
    #         incax2.set_ylabel("Estimated\n+ve hospitalizations", rotation=90, va='center', fontsize=6)
    #         incax2.set_yticks([])
    #         incax2.spines['right'].set_visible(False)
    #         incax2.spines['left'].set_visible(True)
    #         incax2.spines['top'].set_visible(False)
    #         incax2.yaxis.set_label_position('left')
    #         incax2.yaxis.tick_left()
    #     incax.set_title(short_names.get(pathogen, pathogen))
    #     incax.set_xticks(pd.to_datetime(["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01",]))
    #     incax.set_xticklabels(["", "2017", "", "2019", "", "2021", "", "2023", "", "2025",], fontsize=6)
    #     for label in incax.get_xticklabels():
    #         label.set_rotation(45)
    #         label.set_horizontalalignment('right')
    #         label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
    #     ageax = fig.add_subplot(gs[1, i])
    #     season_age_axes[i] = ageax
    #     plot_relative_age_incidence(ageax, pathogen, group_names=["<1y","<5y"], aggregation="YS-OCT")
    #     ageax.set_xticklabels(["", "2016/17", "", "2018/19", "", "2020/21", "", "2022/23", "", "2024/25",], fontsize=6)
    #     ageax.set_xlim(pd.to_datetime("2015-05-01"), pd.to_datetime("2025-10-01"))
    #     for label in ageax.get_xticklabels():
    #         label.set_rotation(45)
    #         label.set_horizontalalignment('right')
    #         label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
    #     ageax.axvspan(last_pre_time, rebound_time, color='red', alpha=0.1, linewidth=0)
    #     # add grid
    #     ageax.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.3)
    #     ageax.grid(axis='x', linestyle='-', linewidth=0.5, alpha=0.3)
    #     ageax.set_ylim(0.0, 0.5)
    #     ageax.set_yticks([0,0.1,0.2,0.3,0.4,0.5])
    #     ageax.set_yticklabels(["0.0","","","","","0.5"], fontsize=6)
    #     if i>0:
    #         ageax.set_yticklabels(["","","","","",""], fontsize=6)
    # season_age_axes[0].set_ylabel("KPSC", rotation=0, ha='right', labelpad=16)
    # sax2 = season_age_axes[0].twinx()
    # sax2.set_ylabel("Proportion of\nhospitalizations", rotation=90, va='center', fontsize=6, labelpad=16)
    # sax2.set_yticks([])
    # sax2.spines['right'].set_visible(False)
    # sax2.spines['left'].set_visible(True)
    # sax2.spines['top'].set_visible(False)
    # sax2.yaxis.set_label_position('left')
    # sax2.yaxis.tick_left()
    # # put legend in first plot
    # legend = season_age_axes[-1].legend(loc='best', fontsize=6)
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_edgecolor('black')
    # legend.get_frame().set_alpha(1)
    # legend.get_frame().set_linewidth(0.5)
    # legend.get_frame().set_boxstyle('square,pad=0')
    # kpsc_axis[0].text(-0.5, 1.1, "A", transform=kpsc_axis[0].transAxes, fontsize=16, fontweight="bold")
    # season_age_axes[0].text(-0.5, 1.1, "B", transform=season_age_axes[0].transAxes, fontsize=16, fontweight="bold")
    # plt.savefig("Figures/KPSC_and_age_incidence.png", dpi=500)

    # #### sesonal maximum and minimum immunity vs r0 plot
    # fig, ax = plt.subplots(1,2,figsize=(6.5,3), layout="constrained", sharey=False, sharex=True)
    # for pathogen, option2, seed, color, prune in zip(pathogens, option2s, seeds, colors, pruners):
    #     print(pathogen)
    #     plot_r0_vs_population_immunity(ax[0], pathogen, option2, seed, color, r0_base=15.24, prune=prune, n_samples=2000, summary="min")
    #     plot_r0_vs_population_immunity(ax[1], pathogen, option2, seed, color, r0_base=15.24, prune=prune, n_samples=2000, summary="max")
    # for fit_ax in ax:
    #     x = np.linspace(1, 7, 100)
    #     y = 1 - 1 / x
    #     fit_ax.plot(x, y, color='k', linewidth=0.3, zorder=0)
    #     # label with "1 - 1/R0" in the middle of the line
    #     fit_ax.text(3.5, 1-1/3.5, r"$1 - \frac{1}{R_0}$", rotation=0, fontsize=11, color='k', ha='right', va='bottom', zorder=1)
    #     # build custom legend with colored squares
    #     handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=8) for color in colors]
    #     labels = [nice_names.get(pathogen, pathogen) for pathogen in pathogens]
    #     fit_ax.legend(handles, labels, frameon=False, fontsize=6, loc='lower right')
    #     fit_ax.set_xscale('log')
    #     fit_ax.set_xticks([1,2,3,4,5,6,7,])
    #     fit_ax.set_xticklabels([1,2,3,4,5,6,7,])
    #     fit_ax.set_xlim(1, 7)
    #     fit_ax.set_ylim(0, 0.95)
    #     fit_ax.set_xlabel("Basic reproduction number (log scale)")
    # ax[0].set_ylabel("Seasonal minimum of\neffective population immunity")
    # ax[1].set_ylabel("Seasonal maximum of\neffective population immunity")
    # plt.savefig("Figures/R0_vs_population_immunity_min_max.png", dpi=500)

    ### suppression order heatmaps with tropical and nontropical
    # fig, ax = plt.subplots(1,3,figsize=(6.5,3), layout="constrained", sharey=True, sharex=True)
    # plot_suppression_rank_heatmap(ax[0], flunet_pathogens, tropical="All", pvals=False)
    # ax[0].set_title("All")
    # plot_suppression_rank_heatmap(ax[1], flunet_pathogens, tropical="Tropical", pvals=False)
    # ax[1].set_title("Tropical")
    # im = plot_suppression_rank_heatmap(ax[2], flunet_pathogens, tropical="Nontropical", pvals=False)
    # ax[2].set_title("Nontropical")
    # ax[0].set_ylabel("Re-emerging after")
    # ax[1].set_xlabel("Re-emerging before")
    # # add colorbar
    # cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.6)
    # cbar.set_label("Frequency")
    # plt.savefig("Figures/Suppression_rank_heatmap_tropical.png", dpi=500)

    # # 1) % of all cases occurring in previously-infected individuals
    # # 2) % of all transmission coming from previously-infected individuals
    # for i, pathogen, seed, option2, prune in zip(range(6), pathogens, seeds, option2s, pruners):
    #     # get maximum relative susceptibility
    #     from likelihood import run_simulation
    #     _, _x, _ = load_optimization_results("emcee_median_", pathogen, seed, lockdown, option1, option2)
    #     params = x_to_params(_x, pathogen, lockdown, option1, option2, NAG=NAG)
    #     solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
    #     values = solution.ys.T
    #     shaped_values = values[1:, :].reshape((1+2*N_S, NAG, -1))
    #     # only pre-pandemic
    #     shaped_values = shaped_values[:, :, (POINTS < date_to_t('2020-03-01'))]
    #     infectious = shaped_values[1:2*N_S:2, :, :]
    #     susceptible = shaped_values[0:2*N_S:2, :, :]
    #     all_infectious = (params[7][:, :, None] * infectious).sum(axis=0)
    #     infected_before = infectious.copy()
    #     infected_before = infected_before.at[0].set(0)
    #     all_infectious_before = (params[7][:, :, None] * infected_before).sum(axis=0)
    #     population_size = calculate_population_size(values, NAG=NAG)[POINTS < date_to_t('2020-03-01')]
    #     foi_matrix = params[4] * params[3][:, :, None] * all_infectious[None, :, :] / jnp.sum(population_size, axis=1)[None, None, :]
    #     infections_matrix = params[6][:, None, None, None] * foi_matrix[None, :, :, :] * susceptible[:, :, None, :]
    #     foi_matrix_before = params[4] * params[3][:, :, None] * all_infectious_before[None, :, :] / jnp.sum(population_size, axis=1)[None, None, :]
    #     infections_matrix_before = params[6][:, None, None, None] * foi_matrix_before[None, :, :, :] * susceptible[:, :, None, :]

    #     infected_infected_before = (infectious[1:].sum(axis=(0,1)) / infectious.sum(axis=(0,1))).mean()
    #     infections_from_infected_before = (infections_matrix_before.sum(axis=(0,1)) / infections_matrix.sum(axis=(0,1))).mean()
    #     print(f"{pathogen}: {infected_infected_before:.4f}, {infections_from_infected_before:.4f}")



    # susceptibility by age pre mid and post figure
    # medians_and_cis = get_age_susceptibility_over_time(pathogens, seeds, lockdown, option1, option2s, pruners)

    # pre_median, pre_lower, pre_upper = medians_and_cis[:, :, :, 0]
    # mid_median, mid_lower, mid_upper = medians_and_cis[:, :, :, 1]
    # post_median, post_lower, post_upper = medians_and_cis[:, :, :, 2]

    # fig, axes = plt.subplots(2, 3, figsize=(6.5, 2), sharey=True, sharex=True)
    # for i, pathogen in enumerate(pathogens):
    #     ax = axes[i // 3, i % 3]
    #     ax.scatter(np.arange(len(AGE_GROUP_NAMES))-0.1, pre_median[i], color='k', label="2017-07-01", marker='s', s=3)
    #     ax.scatter(np.arange(len(AGE_GROUP_NAMES)), mid_median[i], color='hotpink', label="2021-07-01", marker='s', s=3)
    #     ax.scatter(np.arange(len(AGE_GROUP_NAMES))+0.1, post_median[i], color='silver', label="2024-07-01", marker='s', s=3)
    #     for ag in range(len(AGE_GROUP_NAMES)):
    #         ax.plot([ag-0.1, ag-0.1], [pre_lower[i, ag], pre_upper[i, ag]], color='k', linewidth=0.7)
    #         ax.plot([ag, ag], [mid_lower[i, ag], mid_upper[i, ag]], color='hotpink', linewidth=0.7)
    #         ax.plot([ag+0.1, ag+0.1], [post_lower[i, ag], post_upper[i, ag]], color='silver', linewidth=0.7)
    #     ax.set_title(short_names.get(pathogen, pathogen))
    #     ax.set_xticks(np.arange(len(AGE_GROUP_NAMES)))
    #     ax.set_xticklabels(AGE_GROUP_NAMES, rotation=45, ha='right', fontsize=6)
    #     ax.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.3)
    #     ax.grid(axis='x', linestyle='-', linewidth=0.5, alpha=0.3)
    #     ax.tick_params(axis='y', labelsize=6)
    # legend = axes[0,-1].legend(fontsize=6,)
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_edgecolor('black')
    # legend.get_frame().set_alpha(1)
    # legend.get_frame().set_linewidth(0.5)
    # legend.get_frame().set_boxstyle('square,pad=0')
    # # increase whitespace between rows
    # plt.subplots_adjust(hspace=0.4)
    # plt.savefig("Figures/age_susceptibility_by_times2test.png", dpi=300, bbox_inches='tight')
    

    ## age differences over time
    # age_inc = np.zeros((len(pathogens), 10, 3))

    # for pathogen in pathogens:
    #     _, ax = plt.subplots(figsize=(4, 3))
    #     cum_incidence = plot_relative_age_incidence(ax, pathogen, group_names=["<1y","<5y"], aggregation="YS-OCT")
    #     inc_under1 = cum_incidence["3-11m"]
    #     inc_under5 = cum_incidence["1-4y"]
    #     total_inc = cum_incidence[">=65y"]
    #     age_inc[pathogens.index(pathogen), :, 0] = np.floor(inc_under1)
    #     age_inc[pathogens.index(pathogen), :, 1] = np.floor(inc_under5)
    #     age_inc[pathogens.index(pathogen), :, 2] = np.floor(total_inc)

    # print(age_inc[0,:,1:])

    # import statsmodels.api as sm

    # for i, pathogen in enumerate(pathogens):
    #     # 2. Transform to [successes, failures] for statsmodels
    #     successes = age_inc[i,:, 0]
    #     totals = age_inc[i,:, -1]
    #     proportions = successes / totals

    #     baseline_prop = proportions[0:5]
    #     baseline_totals = totals[0:5]

    #     results = []

    #     # 3. Loop through the test years
    #     for j in range(6, 10):
    #         test_prop = proportions[j:j+1]
    #         test_totals = totals[j:j+1]
            
    #         # Combine baseline with the 1 test year
    #         endog = np.concatenate((baseline_prop, test_prop))
    #         weights = np.concatenate((baseline_totals, test_totals))
            
    #         # Create the predictor variable
    #         exog = np.array([0, 0, 0, 0, 0, 1])
    #         exog = sm.add_constant(exog) 
            
    #         # 4. Fit the model using var_weights
    #         model = sm.GLM(endog, exog, family=sm.families.Binomial(), var_weights=weights)
    #         fit = model.fit(scale='X2')

    #         # 5. Extract Predictions and 95% Confidence Intervals
    #         # exog=[1, 0] represents the baseline, exog=[1, 1] represents the test year
    #         pred_base = fit.get_prediction([1, 0]).summary_frame(alpha=0.05)
    #         pred_test = fit.get_prediction([1, 1]).summary_frame(alpha=0.05)
            
    #         base_mean = pred_base['mean'].values[0]
    #         base_ci_low = pred_base['mean_ci_lower'].values[0]
    #         base_ci_upp = pred_base['mean_ci_upper'].values[0]
            
    #         test_mean = pred_test['mean'].values[0]
    #         test_ci_low = pred_test['mean_ci_lower'].values[0]
    #         test_ci_upp = pred_test['mean_ci_upper'].values[0]
            
    #         p_val = fit.pvalues[1] 
    #         dispersion = fit.scale
            
    #         results.append({
    #             'Test Year': j,
    #             'Baseline': f"{base_mean:.1%}",
    #             'Base 95% CI': f"[{base_ci_low:.1%} - {base_ci_upp:.1%}]",
    #             'Test Prop': f"{test_mean:.1%}",
    #             'Test 95% CI': f"[{test_ci_low:.1%} - {test_ci_upp:.1%}]",
    #             'Dispersion': round(dispersion, 2),
    #             'P-Value': round(p_val, 4),
    #             'Sig.': "Yes" if p_val < 0.05 else "No"
    #         })

    #     # 5. Display the results
    #     results_df = pd.DataFrame(results)
    #     print(f"\nResults for {pathogen}:")
    #     print(results_df.to_string(index=False))

    # Regression
    # r0srels = np.zeros((6, 3))
    # seasonwane = np.zeros((6,3))
    # ageobssect = np.zeros((6,3))
    # badageobssect = np.zeros((6,3))
    # all_x = np.zeros((6, 21))
    # suppression = np.zeros(6)
    # excess_sus = np.zeros(6)

    # transmission_under5 = np.zeros(6)
    # transmission_under18 = np.zeros(6)

    # from sim_grid import extract_target_value_from_data

    # for i, pathogen, seed, option2, prune in zip(range(6), pathogens, seeds, option2s, pruners):
        # x = consistent_x_from_DE(pathogen, lockdown, option1, option2, seed, NAG=NAG, prefix="emcee_median_")
        # r0 = np.log(r0_base * x[2] / x[0])
        # # find srel1 and srel2
        # srel1 = x[7]
        # srel2 = x[8]
        # r0srels[i,:] = [r0, srel1, srel2]
        # # find seasonality, offset, and waning
        # seasonwane[i,:] = [x[3], x[4], x[6]]
        # ageobssect[i,:] = [x[14], x[15], x[-1]]
        # badageobssect[i,:] = [x[17], x[18], x[19]]
        # all_x[i,:] = x
        # all_x[i,2] = np.log(all_x[i,2])
        # # get suppression time
        # suppression[i] = extract_target_value_from_data(pathogen, "suppression_length")
        # # get maximum relative susceptibility
        # from likelihood import run_simulation
        # _, _x, _ = load_optimization_results("emcee_median_", pathogen, seed, lockdown, option1, option2)
        # params = x_to_params(_x, pathogen, lockdown, option1, option2, NAG=NAG)
        # solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
        # population_size = calculate_population_size(solution.ys.T, NAG=NAG)
        # sus = susceptibility(solution, params).sum(axis=1)/population_size.sum(axis=1)
        # rel_sus = sus / sus.mean()
        # excess_sus[i] = rel_sus.max()

    #     print(option1, option2)
    #     inf_matrix, _ = get_infection_matrix(pathogen, seed, lockdown, option1, option2, NAG, CENSUS_AGE_POP, prune=prune, samples=None, hospitalizations=False, prefix="emcee_median_")
    #     transmission_under5[i] = inf_matrix[:,:3].sum() / inf_matrix.sum()
    #     transmission_under18[i] = inf_matrix[:,:4].sum() / inf_matrix.sum()
    # print(transmission_under5)
    # print("Fold-difference under 5:", transmission_under5.max() / transmission_under5.min())
    # print(transmission_under18)
    # print("Fold-difference under 18:", transmission_under18.max() / transmission_under18.min())
    
    # # linear regression of suppression vs r0 and srels
    # from sklearn.linear_model import LinearRegression
    # X = r0srels
    # y = suppression
    # # scale X and y to have mean 0 and std 1
    # X = (X - X.mean(axis=0)) / X.std(axis=0)
    # y = (y - y.mean()) / y.std()
    # reg = LinearRegression().fit(X, y)
    # print(f"Regression coefficients: {reg.coef_}, intercept: {reg.intercept_}")
    # print(f"R^2: {reg.score(X, y)}")

    # # scale X and excess_sus to have mean 0 and std 1
    # y = (excess_sus - excess_sus.mean()) / excess_sus.std()
    # susreg = LinearRegression().fit(X, y)
    # print(f"Regression coefficients (excess susceptibility): {susreg.coef_}, intercept: {susreg.intercept_}")
    # print(f"R^2: {susreg.score(X, y)}")

    # X = (seasonwane - seasonwane.mean(axis=0)) / seasonwane.std(axis=0)
    # y = (suppression - suppression.mean()) / suppression.std()
    # reg2 = LinearRegression().fit(X, y)
    # print(f"Regression coefficients (seasonality, offset, waning): {reg2.coef_}, intercept: {reg2.intercept_}")
    # print(f"R^2: {reg2.score(X, y)}")

    # y = (excess_sus - excess_sus.mean()) / excess_sus.std()
    # susreg2 = LinearRegression().fit(X, y)
    # print(f"Regression coefficients (excess susceptibility): {susreg2.coef_}, intercept: {susreg2.intercept_}")
    # print(f"R^2: {susreg2.score(X, y)}")

    # X = (ageobssect - ageobssect.mean(axis=0)) / ageobssect.std(axis=0)
    # y = (suppression - suppression.mean()) / suppression.std()
    # reg3 = LinearRegression().fit(X, y)
    # print(f"Regression coefficients (age_obs1, age_obs2, age_obs7): {reg3.coef_}, intercept: {reg3.intercept_}")
    # print(f"R^2: {reg3.score(X, y)}")
    # y = (excess_sus - excess_sus.mean()) / excess_sus.std()
    # susreg3 = LinearRegression().fit(X, y)
    # print(f"Regression coefficients (excess susceptibility): {susreg3.coef_}, intercept: {susreg3.intercept_}")
    # print(f"R^2: {susreg3.score(X, y)}")

    # X = (badageobssect - badageobssect.mean(axis=0)) / badageobssect.std(axis=0)
    # y = (suppression - suppression.mean()) / suppression.std()
    # reg4 = LinearRegression().fit(X, y)
    # print(f"Regression coefficients (bad_age_obs1, bad_age_obs2, bad_age_obs3): {reg4.coef_}, intercept: {reg4.intercept_}")
    # print(f"R^2: {reg4.score(X, y)}")
    # y = (excess_sus - excess_sus.mean()) / excess_sus.std()
    # susreg4 = LinearRegression().fit(X, y)
    # print(f"Regression coefficients (excess susceptibility): {susreg4.coef_}, intercept: {susreg4.intercept_}")
    # print(f"R^2: {susreg4.score(X, y)}")

    # # ElasticNet version
    # from sklearn.linear_model import ElasticNet
    # from sklearn.preprocessing import StandardScaler

    # scaler = StandardScaler()
    # all_x_scaled = scaler.fit_transform(all_x)

    # for alpha in 10**(np.linspace(-1, 3, 100)):
    #     en = ElasticNet(alpha=alpha, l1_ratio=1).fit(all_x_scaled, suppression)
    #     if np.sum(np.abs(en.coef_)>0)==4:
    #         print(f"ElasticNet coefficients (all parameters) for alpha={alpha}: {en.coef_}, intercept: {en.intercept_}")
    #         print(f"R^2: {en.score(all_x_scaled, suppression)}")
    #         break
    #     else:
    #         print(alpha, np.sum(np.abs(en.coef_)>0))

    # for alpha in 10**(np.linspace(-5, 0, 100)):
    #     susen = ElasticNet(alpha=alpha, l1_ratio=1).fit(all_x_scaled, excess_sus)
    #     if np.sum(np.abs(susen.coef_)>0)==4:
    #         print(f"ElasticNet coefficients (excess susceptibility) for alpha={alpha}: {susen.coef_}, intercept: {susen.intercept_}")
    #         print(f"R^2: {susen.score(all_x_scaled, excess_sus)}")
    #         break
    #     else:
    #         print(alpha, np.sum(np.abs(susen.coef_)>0))
    # # en3 = ElasticNet(alpha=0.2, l1_ratio=0.5).fit(all_x, suppression)
    # # print(f"ElasticNet coefficients (all parameters): {en3.coef_}, intercept: {en3.intercept_}")
    # # print(f"R^2: {en3.score(all_x, suppression)}")
    # # susen3 = ElasticNet(alpha=0.01, l1_ratio=0.5).fit(all_x, excess_sus)
    # # print(f"ElasticNet coefficients (excess susceptibility): {susen3.coef_}, intercept: {susen3.intercept_}")
    # # print(f"R^2: {susen3.score(all_x, excess_sus)}")

    ### plotting population susceptibility
    # fig, fit_ax = plt.subplots(1, 1, figsize=(3.5, 3.5), layout="constrained", sharex=False, sharey=False)
    # r0_values_by_pathogen = {}
    # immunity_values_by_pathogen = {}
    # for pathogen, option2, seed, color in zip(pathogens, option2s, seeds, colors):
    #     start_time = time.time()
    #     print(f"plotting parameter scatter for {pathogen}...")
    #     r0_values, immunity_values = plot_r0_vs_population_immunity(fit_ax, pathogen, option2, seed, color,n_samples=10000)
    #     r0_values_by_pathogen[pathogen] = r0_values
    #     immunity_values_by_pathogen[pathogen] = immunity_values
    #     end_time = time.time()
    #     print(f"Finished plotting parameter scatter for {pathogen} in {end_time - start_time:.2f} seconds.")
    # x = np.linspace(1, 10, 100)
    # y = 1 - 1 / x
    # fit_ax.plot(x, y, color='k', linewidth=0.5, zorder=0)
    # # label with "1 - 1/R0" in the middle of the line
    # fit_ax.text(3.5, 1-1/3.5, r"$1 - \frac{1}{R_0}$", rotation=0, fontsize=11, color='k', ha='right', va='bottom', zorder=1)
    # # build custom legend with colored squares
    # handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=8) for color in colors]
    # labels = [nice_names.get(pathogen, pathogen) for pathogen in pathogens]
    # fit_ax.legend(handles, labels, frameon=False, fontsize=6, loc='lower right')
    # fit_ax.set_xscale('log')
    # fit_ax.set_xticks([1,2,3,4,5,6,7,8,9,10])
    # fit_ax.set_xticklabels([1,2,3,4,5,6,7,8,9,10])
    # fit_ax.set_xlim(1, 10)
    # fit_ax.set_ylim(0, 0.95)
    # fit_ax.set_xlabel("Basic reproduction number (log scale)")
    # fit_ax.set_ylabel("Seasonal average population immunity to infection")
    # plt.savefig(f"Figures/r0_vs_mean_population_immunity_{lockdown}.png", dpi=300)
    
    # ### pre-2020 table
    # lockdown = "Default"
    # pathogens = ["RSV","Metapneumovirus","Parainfluenza3","Adenovirus","InfluenzaA","InfluenzaB",]
    # option2s = ["2020-01-01maxagep028","2020-01-01fixage0maxagep006","2020-01-01maxagep004","2020-01-01maxagep003","2020-01-01maxagep035","2020-01-01maxagep035",]
    # seeds = [260615, 260624, 260615, 260615, 260615, 260615,]
    # pruners = [100, 100, 400, 400, 200, 100]

    # print_parameter_table(pathogens, option2s, seeds, pruners)

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


    # #### perturbation analysis
    # from likelihood import run_simulation
    # perturbations = np.linspace(0,1,30)*0.5
    # PERIOD = pd.date_range(start=pd.to_datetime('2015-07-04'), end=pd.to_datetime('2025-05-01'), freq='MS')
    # POINTS = np.array(date_to_t(PERIOD))
    # anchor_t = date_to_t(pd.to_datetime("2020-01-01")) - start_idx
    # anchor_idx = int(np.searchsorted(POINTS, anchor_t, side='right'))-1
    # print(anchor_idx)
    # from sim_grid import suppression_duration_single_series

    # samples_per_step = 5000
    
    # durations = np.zeros((len(pathogens), len(perturbations), samples_per_step))

    # for pi, perturbation in enumerate(perturbations):
    #     print(pi)
    #     random_perturbation = np.random.normal(1, perturbation, size=samples_per_step)
    #     for pathogen, seed, option2, prune in zip(pathogens, seeds, option2s, pruners):
    #         _, _, _, p_time_to_obs, _ = pathogen_parameters(pathogen, import_multiplier=1e-9, incidence_data=False, hosp=True, NAG=NAG, dedup=True)
    #         chain = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=prune, prefix="")
    #         x_samples = jnp.asarray(chain[np.random.choice(chain.shape[0], size=samples_per_step, replace=False), :])
    #         r1_idx = 6 + ("RSV" not in pathogen) + ("Influenza" not in pathogen)
    #         r1 = x_samples[:, r1_idx]
    #         x_samples = x_samples.at[:, r1_idx].set(r1 * random_perturbation)
    #         def suppression_duration_for_pathogen(x):
    #             params = x_to_params(x, pathogen, lockdown, option1, option2, NAG=NAG)
    #             solution = run_simulation(params, STATE0, int(POINTS[-1]), POINTS, NAG=NAG)
    #             values = solution.ys.T
    #             trajectory = jnp.diff(values[-NAG:,:], axis=1).T
    #             obs_summed_age = trajectory.sum(axis=1)
    #             duration = suppression_duration_single_series(obs_summed_age, anchor_idx)
    #             return jnp.where(jnp.isnan(duration), 0, duration)
    #         durations[pathogens.index(pathogen), pi, :] = jax.jit(jax.vmap(suppression_duration_for_pathogen))(x_samples)
    # # for each sample and perturbation, find the order of the durations
    # durations_ordinal = np.argsort(durations, axis=0)
    # canonical_order = np.array([3,2,0,1,4,5])
    # # measure distance from canonical order for each perturbation
    # distances = np.mean(durations_ordinal != canonical_order[:, None, None], axis=(0,2))
    # # plot distance against perturbation
    # fig, ax = plt.subplots(figsize=(3.5, 3.5), layout="constrained")
    # ax.plot(perturbations, distances, color='k')
    # ax.set_xlabel("Perturbation to r1")
    # ax.set_ylabel("Difference from canonical order")
    # plt.savefig(f"Figures/suppression_duration_order_sensitivity_r1_correlated.png", dpi=300)
    # distances_f1 = np.genfromtxt("Data/Processed/saved_f1_random_perturbation_distances.csv")
    # distances_seas = np.genfromtxt("Data/Processed/saved_season_random_perturbation_distances.csv")
    # fig, ax = plt.subplots(1,2,figsize=(6.5, 3), layout="constrained", sharey=True)
    # x = 1 + np.linspace(0,1,30)*0.5
    # ax[1].plot(x, distances_f1, color='k')
    # ax[1].set_xlabel("Perturbation to NPI sensitivity ($f$)")
    # ax[1].set_ylabel("Difference from observed order")
    # x = 1 + np.linspace(0,1,100)*0.5
    # ax[0].plot(x, distances_seas, color='k')
    # ax[0].set_xlabel("Perturbation to seasonal forcing amplitude ($A$)")
    # ax[0].set_ylabel("Difference from observed order")
    # plt.savefig(f"Figures/suppression_duration_order_sensitivity_r1_seas.png", dpi=300)


    # #### correlation matrix difference figure
    # fig, axes = plt.subplots(3, 2, figsize=(5, 6.5), layout="constrained", sharex=False, sharey=False)
    # alt_pruners = [100, 100, 400, 400, 200, 100]
    # for index, pathogen, option2, seed, prune, ax in zip(range(6), pathogens, option2s, seeds, pruners, axes.flatten()):
    #     print(f"Difference measures for {pathogen}...")
    #     chain2 = load_mcmc_chain(pathogen, seed, lockdown, option1, option2, just_chain=True, prune=prune, prefix="")
    #     chain1 = load_mcmc_chain(pathogen, [260615, 260624][pathogen=="Metapneumovirus"], ["Default", "ExponentialODipp25"][pathogen=="Metapneumovirus"], option1, "2020-01-01"+option2, just_chain=True, prune=alt_pruners[index], prefix="")
    #     param_names1, _ = parameters_names_bounds(pathogen, "Default", "dedupsac", "2020-01-01"+option2, NAG=NAG)
    #     param_names2, _ = parameters_names_bounds(pathogen, lockdown, "dedupsac", option2, NAG=NAG)
    #     select_param_names = [param_name for param_name in param_names1 if param_name in param_names2]
    #     compare_mcmc_chains(chain1, chain2, param_names1, param_names2, select_param_names)
    #     frob_norm_diff = plot_correlation_matrix_difference(ax, chain1, chain2, param_names1, param_names2, select_param_names)
    #     ax.set_title(short_names.get(pathogen, pathogen)+str(f" ({frob_norm_diff:.3f})"), fontsize=8)
    # plt.savefig(f"Figures/correlation_matrix_difference_DESnooker.png", dpi=300)


    # # ==========================================
    # # Figure 2: Fits and Susceptibility Figure
    # # ==========================================
    # sample_size = 400
    # fig = plt.figure(figsize=(6.5,7), layout="constrained")
    # subfigs = fig.subfigures(4, 1, height_ratios=[2.3, 5, 1.15, 0.1])

    # axes_A = subfigs[0].subplots(2, 3, sharex=True)
    # plot_fits_and_sus(axes_A, sample_size)
    # subfigs[0].supylabel("Monthly incidence\n(per 100,000)", fontsize=8, x=0.03, va='center', ha='center')

    # axes_B = subfigs[1].subplots(7, 6, sharex=False, sharey=False)
    # plot_fits(axes_B, n_samples=sample_size, load_data=False, save_data=True, colors=colors)
    # for ax in axes_B[6, :]:
    #     ax.set_xticklabels([])

    # axes_C = subfigs[2].subplots(1, 6, sharex=False, sharey=False)
    # for pathogen, option2, prune, seed, ax in zip(pathogens, option2s, pruners, seeds, axes_C):
    #     plot_susceptibility(ax, pathogen, option2, prune, seed, n_samples=sample_size)
    #     ax.set_xticks(pd.to_datetime([
    #         "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", 
    #         "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"
    #     ]))
    #     ax.set_xticklabels(["", "2017", "", "2019", "", "2021", "", "2023", "", "2025"], fontsize=6)
    #     ax.tick_params(axis='y', labelsize=6)
    #     ax.set_ylim(0.5, 1.5)
    #     ax.set_yticks([0.5, 1.0, 1.5])
    #     for label in ax.get_xticklabels():
    #         label.set_rotation(45)
    #         label.set_horizontalalignment('right')
    #         label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(5 / 72.0, 3 / 72.0, fig.dpi_scale_trans))
    # axes_C[0].set_ylabel("Relative\nsusceptibility", fontsize=6)
    # legend_handles = [
    #     Patch(facecolor=color, label=name) 
    #     for name, color in zip(nice_age_group_names, hsv_colors)
    # ]
    # subfigs[3].legend(
    #     handles=legend_handles,
    #     loc="center",
    #     ncol=len(nice_age_group_names),
    #     fontsize=6,
    #     frameon=False,
    #     handletextpad=0.4,
    #     columnspacing=1.2
    # )
    # # label panels A, B, C
    # axes_A[0, 0].text(-0.35, 1.15, "A", transform=axes_A[0, 0].transAxes, fontsize=16, fontweight="bold")
    # axes_B[0, 0].text(-0.7, 1.15, "B", transform=axes_B[0, 0].transAxes, fontsize=16, fontweight="bold")
    # axes_C[0].text(-0.7, 1.15, "C", transform=axes_C[0].transAxes, fontsize=16, fontweight="bold")

    # plt.savefig("Figures/fits_figure_combined.pdf", dpi=500, bbox_inches='tight')

    # # # plot_mcmc_corner("InfluenzaB", "Default", "dedupsac", "2020-01-01maxagep035", "260615", prune=0)

    # # # fig, ax = plt.subplots(figsize=(3,3))
    # # # im = plot_suppression_rank_heatmap(ax, flunet_pathogens)
    # # # # ax.set_title("How often does y re-emerge before x?")
    # # # ax.set_ylabel("How often does...")
    # # # ax.set_xlabel("re-emerge after ...?")
    # # # plt.tight_layout()
    # # # plt.savefig(f"Figures/suppression_rank_heatmap.png", dpi=300)

    # # # ## plot MCMC corners and traces for all pathogens
    # # # for pathogen, option2, seed in zip(pathogens, option2s, seeds):
    # # #     print(pathogen, option2, seed)
    # # #     plot_mcmc_corner(pathogen, option2, seed, prune=10000)
    # # #     print(f"plotting MCMC traces for {pathogen}...")
    # # #     fig, axes = plt.subplots(4, 4, figsize=(13.3,7.5), sharex=True)
    # # #     # Calculate this once to avoid repeating the function call
    # # #     n_params = len(parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)[0])
    # # #     plot_mcmc_traces(axes.flatten(), pathogen, option2, seed, prune=0)
    # # #     fig.suptitle(f"MCMC traces for {nice_names.get(pathogen, pathogen)}", fontsize=16)
    # # #     for i, ax in enumerate(axes.flatten()):
    # # #         if i >= n_params:
    # # #             ax.axis('off')
    # # #     for col in range(axes.shape[1]):
    # # #         for row in reversed(range(axes.shape[0])):
    # # #             flat_idx = row * axes.shape[1] + col
    # # #             if flat_idx < n_params:
    # # #                 axes[row, col].tick_params(labelbottom=True)
    # # #                 axes[row, col].set_xlabel("Iteration number")
    # # #                 break
    # # #     plt.tight_layout()
    # # #     plt.savefig(f"Figures/mcmc_traces_{pathogen}_{option2}_{seed}_slide.png", dpi=300)
    # # #     plt.close()

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
    # # for i in range(0, len(countries), 15):
    # #     fig, axes = plt.subplots(min(15, len(countries) - i), 6, figsize=(6.5,9))
    # #     countries_chunk = countries[i:i+15]
    # #     plot_FluNet_chunk(axes, countries_chunk)
    # #     plt.tight_layout()
    # #     plt.savefig(f"Figures//FluNetTimeseries_{countries[i]}_to_{countries[min(i+14, len(countries)-1)]}.png", dpi=300)

    # # ## Generate Figure 2: age-structured fits figure
    # # fig, axes = plt.subplots(7, 6, figsize=(6.5,6.5), layout="constrained")
    # # plot_fits(axes, n_samples=400, load_data=True)
    # # fig.text(0.001, 0.5, 'Estimated incidence of hospitalization per 100k members', va='center', rotation='vertical')
    # # plt.savefig(f"Figures/age_structured_fits.png", dpi=300)

    # # # ## Generate Figure 3: age infection figure
    # fig = plt.figure(figsize=(5, 7), layout="constrained")
    # plot_age_figure(fig, pathogens, colors, option1, option2s, pruners, seeds, lockdown, NAG, CENSUS_AGE_POP, AGE_GROUP_NAMES, age_adjusted=False, logD=True, samples=400, load_data=True, prefix="")
    # plt.savefig(f"Figures/Figure3.png", dpi=300)

    # # Generate Figure 4: immunity cascade figure
    # fig, axes = plt.subplots(6,7, figsize=(4, 5), layout="constrained", sharex=False, sharey=False)
    # plot_immunity_cascade(axes, n_samples=400, save_data=True, kpsc_incidence=True)
    # for ax in axes[-1,:]:
    #     ax.set_xticklabels(["No imm.", "+ I. exp.", "+ D. exp.", "+ D. age"][::-1], fontsize=6, rotation=90, ha='center')
    # plt.savefig(f"Figures/immunity_cascade_{lockdown}_log_twopart.png", dpi=300)


    # # Generate Figure 5: age group heatmaps and line of best fit
    # from sim_grid import generate_2d_heatmap_plot
    # good_simulations = [[pathogen, seed, lockdown, option1, option2, prune] for pathogen, seed, option2, prune in zip(pathogens, seeds, option2s, pruners)]
    # run_save_path = "Outputs/sim_grid_lh_n80000_chunk10000_seed260728_lockdownExponentialODipp25_2d"
    # fig = plt.figure(figsize=(5.25, 4.5))
    # plot_heatmaps_and_best_fit(fig, pathogens, option2s, pruners, seeds, colors, run_save_path, good_simulations, r0_base, fit_line=False)
    # plt.savefig(f"Figures/heatblobs.pdf", dpi=1000)
    # plt.close()