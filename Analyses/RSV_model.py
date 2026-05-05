#### Benjamin John Singer, 2026-05-05
#### Code to simulate RSV transmission, including calibrated parameters

# =========================
# 1. DEFINING THE ODE MODEL
# =========================
import jax
import jax.numpy as jnp

interp_fn = jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=0)

def deltas(t, state, args):
    (FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX, # population parameters
    BETA, WANE, S_REL, I_REL, P_OBS, OBS_AGE, RELATIVE_CONTACT, VAX_RATE, MATERNAL_IMMUNITY, # fit parameters
    REC_UP, REC_SAME, IMPORT_STRENGTH, NAG) = args # pathogen parameters
    N_S = 3
    delta = jnp.zeros((2*N_S+1,NAG))
    maternal = state[0]
    delta_maternal = 0
    shaped_state = state[1:].reshape((2*N_S+1,NAG))
    age_pops = jnp.sum(shaped_state[:2*N_S, :], axis=0).at[0].add(maternal)
    pop_size = jnp.sum(age_pops)
    infectious = shaped_state[1:2*N_S:2, :]
    susceptible = shaped_state[0:2*N_S:2, :]
    # # births
    birth_rate = jnp.interp(t, FULL_POINTS, BIRTH_RATE)
    maternal_immunity = jnp.sum(interp_fn(t, FULL_POINTS, MATERNAL_IMMUNITY) * susceptible[:,4])  / jnp.sum(susceptible[:,4])
    delta_maternal = delta_maternal + maternal_immunity*birth_rate*pop_size
    delta = delta.at[0,0].add((1-maternal_immunity)*birth_rate*pop_size)
    # infections - calculate force of infection
    relative_contact = interp_fn(t, FULL_POINTS, RELATIVE_CONTACT)
    import_strength = jnp.interp(t, FULL_POINTS, IMPORT_STRENGTH)
    CONTACT_t = relative_contact[:, None] * relative_contact[None, :] * CONTACT_MATRIX
    infectious_by_age = jnp.sum(I_REL[:, None] * infectious, axis=0)
    infectious_contact = jnp.dot(CONTACT_t,infectious_by_age)/pop_size
    import_contact = import_strength*jnp.dot(CONTACT_t,age_pops)/pop_size
    force_of_infection = BETA*(infectious_contact + import_contact)
    # multiply by susceptibles in each age group and susceptibility class
    new_infections = S_REL[:, None] * force_of_infection[None, :] * susceptible
    # update deltas for infections
    delta = delta.at[0:2*N_S:2, :].add(-new_infections)
    delta = delta.at[1:2*N_S:2, :].add(new_infections)
    # waning
    waners = WANE[1:, None] * susceptible[1:, :]
    delta = delta.at[2:2*N_S:2, :].add(-waners)
    delta = delta.at[0:2*(N_S-1):2, :].add(waners)
    # recovery
    recers_up = REC_UP[:-1, None] * infectious[:-1, :]
    recers_same = REC_SAME[:, None] * infectious
    delta = delta.at[1:2*(N_S-1):2, :].add(-recers_up)
    delta = delta.at[2:2*N_S:2, :].add(recers_up)
    delta = delta.at[1:2*N_S:2, :].add(-recers_same)
    delta = delta.at[0:2*N_S:2, :].add(recers_same)
    # aging
    agers = AGING_RATE[None, :]*shaped_state[:2*N_S, :]
    delta = delta.at[:2*N_S, :].add(-agers)
    delta = delta.at[:2*N_S, 1:].add(agers[:,:-1])
    # maternal compartment aging
    maternal_agers = AGING_RATE[0] * maternal
    delta_maternal = delta_maternal - maternal_agers
    delta = delta.at[0, 1].add(maternal_agers)
    # vaccination
    vrate = interp_fn(t, FULL_POINTS, VAX_RATE)
    # assume that S_VAX is the last susceptibility class
    vaxxers = vrate[None, :] * susceptible[:-1, :]
    delta = delta.at[0:2*(N_S-1):2, :].add(-vaxxers)
    delta = delta.at[2*N_S-2, :].add(jnp.sum(vaxxers, axis=0))
    # observations
    observed_new_incidence = P_OBS[:, None] * OBS_AGE[None,:] * new_infections
    delta = delta.at[-1, :].add(jnp.sum(observed_new_incidence, axis=0))
    # concatenate maternal immunity delta to flattened delta
    overall_delta = jnp.concatenate((jnp.array([delta_maternal]), delta.flatten()))
    return overall_delta

# =========================
# 2. RUNNING SIMULATIONS
# =========================
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController, RecursiveCheckpointAdjoint
import numpy as np
import pandas as pd
import re

def run_simulation(params, y0, t1, saveat_ts, NAG=7):
    """Run the ODE simulation with the given parameters and initial conditions.
    Args:
        params: tuple of parameters to pass to the ODE function
        y0: initial state (flattened)
        t1: end time for the simulation
        saveat_ts: time points at which to save the solution
        NAG: number of age groups (default 7, can be 8 with split option)
    Returns:
        solution: object containing the simulation results at the specified time points
    """
    sim_params = params + (NAG,)
    term = ODETerm(deltas)
    solver = Dopri5()
    saveat = SaveAt(ts=saveat_ts)
    adjoint = RecursiveCheckpointAdjoint()
    step_controller = PIDController(rtol=1e-5, atol=1e-5)
    dt0 = 0.1
    max_steps = 10000
    solution = diffeqsolve(
                        term, solver,
                        t0=0, t1=t1, dt0=dt0, stepsize_controller=step_controller,
                        saveat=saveat, y0=y0.flatten(), args=sim_params, 
                        max_steps=max_steps, throw=False,
                        adjoint=adjoint,
                        )
    return solution

# jax-safe importations for x_to_params
def x_to_params(x, pathogen, lockdown, option1, option2, fixed_params = None, import_multiplier=1e-9, end_date='2025-05-01', print_params=False, NAG=7):
    """
    Convert the optimization parameters (x) into the full set of parameters needed to run the simulation.
    Args:        
        x: array of optimization parameters
        pathogen, lockdown, option1, option2: strings defining the model configuration
        fixed_params: if not None, a tuple of (FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX) to use instead of the default ones
        import_multiplier: multiplier for the importation strength
        end_date: end date for the simulation (used to define FULL_POINTS)
        print_params: whether to print the parameters after conversion
        NAG: number of age groups (default 7, can be 8 with split option)
    Returns:    
        params: tuple of parameters to pass to the ODE function
    """
    true_NAG = 7
    if "split" in option1:
        NAG = true_NAG = 8
    if fixed_params is None:
        BIRTH_RATE = jnp.asarray(np.genfromtxt('Data/birth_rate_daily.csv', delimiter=','))
        if NAG>7:
            AGING_RATE = 1/jnp.array([3/12*365,9/12*365,4*365,3*365,10*365,22*365,25*365,12.43*365])
        else:
            AGING_RATE = 1/jnp.array([3/12*365,9/12*365,4*365,3*365,32*365,25*365,12.43*365])
        CONTACT_MATRIX = jnp.asarray(pd.read_csv('Data/contact_matrices/KP'+['', '_split'][NAG>7]+'_contact_all_US_Census.csv', delimiter=',', header=None).values)
        REC_UP = jnp.array([1/4.9,1/4.1,0.0])
        REC_SAME = jnp.array([0.0,0.0,1/4.1])
        ARRIVALS = jnp.asarray(np.genfromtxt('Data/arrivals_daily.csv', delimiter=','))
        IMPORT_STRENGTH = import_multiplier*ARRIVALS*jnp.asarray(np.genfromtxt('Data/RSV_positivity_daily.csv', delimiter=','))
    else:
        FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX = fixed_params[:4]
        REC_UP, REC_SAME, IMPORT_STRENGTH = fixed_params[-3:]
    
    N_S = 3
    EPOCH = pd.to_datetime('1970-01-01')
    END = pd.to_datetime(end_date)
    FULL_PERIOD = pd.date_range(start=EPOCH, end=END, freq='D')
    FULL_POINTS = jnp.array(date_to_t(FULL_PERIOD))

    n = 0
    BETA = x[n]
    n+=1
    SEASONALITY = x[n]
    OFFSET = x[n+1]
    MATERNAL_IMMUNITY = jnp.zeros((len(FULL_POINTS), N_S))
    n += 2
    if "SIRS" in option1:
        WANE = jnp.array([0.0,x[n],0.0])
        n += 1
    else:
        WANE = jnp.array([0.0,0.0,x[n]])
        n += 1
    if "SIRS" in option1:
        S_REL = jnp.zeros(N_S)
        S_REL = S_REL.at[0].set(1)
        pobsrel = jnp.ones(N_S)
    elif "RSV" in pathogen:
        S_REL = jnp.array([1,x[n],x[n]*x[n+1]])
        pobsrel = jnp.array([1,0.46,0.31]) # Henderson 1979
        n += 2
    else:
        S_REL = jnp.array([1,x[n],x[n]*x[n+1]])
        pobsrel = jnp.array([1,x[n+2],x[n+2]*x[n+3]])
        n += 4
    if ('flexage' not in option2) and ('maxagep' not in option2):
        P_OBS = x[n]*pobsrel
        n += 1
    else:
        P_OBS = pobsrel
    if "irel" in option1:
        I_REL = jnp.array([1,x[n],x[n]*x[n+1]])
        n += 2
    else:
        I_REL = jnp.array([1,1,1])
    if "maxmimm" in option1:
        MIMM = 1
        MATERNAL_IMMUNITY = MATERNAL_IMMUNITY.at[:,-1].set(MIMM)
    elif "totalmimm" in option1:
        MIMM = 1
        MATERNAL_IMMUNITY = MATERNAL_IMMUNITY.at[:,:].set(MIMM)
    elif "mimm" in option1:
        MIMM = x[n]
        MATERNAL_IMMUNITY = MATERNAL_IMMUNITY.at[:,-1].set(MIMM)
        n += 1
    if "RSV" in pathogen:
        MATERNAL_IMMUNITY = jnp.minimum(1, MATERNAL_IMMUNITY + rsv_maternal_immunity(FULL_POINTS).reshape(-1,1))
    if "Exponential" in lockdown:
        FF = [1,x[n]]
        TT = [date_to_t(EPOCH), date_to_t('2020-03-19')]
        RR = [x[n+1],]
        EXPONENTIAL_CONTACT = exponential_recovery(FULL_POINTS, TT, FF, RR)
        RELATIVE_CONTACT = EXPONENTIAL_CONTACT*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
        n += 2
    if "ODipEqual" in lockdown:
        FO = FF[1]
        dipdates = jnp.array([date_to_t(EPOCH),
                    date_to_t('2021-12-15'),
                    date_to_t('2022-03-01')])
        dipvalues = jnp.array([1, 1-FO, 1])
        ODIP_CONTACT = jax.vmap(lambda t: piecewise(t, dipdates, dipvalues, steepness=0.2))(FULL_POINTS)
        RELATIVE_CONTACT = ODIP_CONTACT[:, None] * RELATIVE_CONTACT if RELATIVE_CONTACT.ndim == 2 else ODIP_CONTACT * RELATIVE_CONTACT
    if ('maxagep' in option2) & ('dynamic' not in option1):
        if "RSV" in pathogen:
            fixed_age_index = 0
        else:
            fixed_age_index = -1
        # search for numbers after maxagep in option2, that number divided by 100 is the value of OBS_AGE for the fixed age group
        match = re.search(r'maxagep(\d+)', option2)
        obs_age_max = int(match.group(1)) / (10 ** len(match.group(1)))
        OBS_AGE = jnp.array([0.0]*NAG)
        OBS_AGE = OBS_AGE.at[fixed_age_index].set(obs_age_max)
        # at all other indexes, use x[n:n+NAG-1] in order
        OBS_AGE = OBS_AGE.at[1+fixed_age_index:NAG+fixed_age_index].set(obs_age_max * x[n:n+NAG-1])
        n += NAG-1
    if (("RSV" in pathogen) and ("nvax" not in option1)) or ("rsvvax" in option1):
        protection_param = S_REL*P_OBS
        max_eff0 = 1 - protection_param[-1]
        max_eff1 = (protection_param[-2]-protection_param[-1])/protection_param[-2]
        VAX_RATE = rsv_eff_vax_rate(FULL_POINTS, max_eff0, max_eff1)
        if true_NAG == 8:
            VAX_RATE = jnp.concatenate((VAX_RATE[:, :5], jnp.tile(VAX_RATE[:, 4:5], (1, NAG-7)), VAX_RATE[:, 5:]), axis=1)
        elif true_NAG == 65:
            VAX_RATE = jnp.concatenate((jnp.tile(VAX_RATE[:, 0:1], (1, 3)), jnp.tile(VAX_RATE[:, 1:2], (1, 9)), jnp.tile(VAX_RATE[:, 2:3], (1, 12*4)), VAX_RATE[:, 3:5], jnp.tile(VAX_RATE[:, 4:5], (1, 1)), VAX_RATE[:, 5:]), axis=1)
    else:
        VAX_RATE = jnp.zeros((len(FULL_POINTS),NAG))

    # if relative contact is 1-dimensional, copy it across all age groups
    if RELATIVE_CONTACT.ndim == 1:
        RELATIVE_CONTACT = jnp.sqrt(jnp.tile(RELATIVE_CONTACT.reshape(-1,1), (1,NAG)))

    params = (FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX,
                BETA, WANE, S_REL, I_REL, P_OBS, OBS_AGE, RELATIVE_CONTACT, VAX_RATE, MATERNAL_IMMUNITY,
                REC_UP, REC_SAME, IMPORT_STRENGTH)
    
    if print_params:
        param_names = ["BETA","WANE","SEASONALITY","OFFSET","S_REL","I_REL","P_OBS","OBS_AGE"]
        if "daycare" in option2:
            param_names += ["DAYCARE"]
        if lockdown != "Taube":
            param_names += ["FF"]
            if "ODip" in lockdown:
                param_names += ["FO"]
            if "Mobility" not in lockdown:
                param_names += ["TT"]
            if "Exponential" in lockdown or "Sigmoid" in lockdown:
                param_names += ["RR"]
        for i in range(len(param_names)):
            if param_names[i] == "TT" and "Mobility" not in lockdown:
                print("TT: " + [t_to_date(t).strftime('%Y-%m-%d') for t in eval(param_names[i])].__str__())
            else:
                print(param_names[i]+": "+eval(param_names[i]).__str__())
        if "mimm" in option1 or "maxmimm" in option1:
            print("MATERNAL_IMMUNITY: "+MIMM.__str__())
    
    return params
    
def t_to_date(t, start_date=pd.to_datetime("1970-01-01")):
    """
    Convert time index to date
    """
    return start_date + pd.DateOffset(days=t)

def date_to_t(date, start_date=pd.to_datetime("1970-01-01")):
    """
    Convert date to time index
    """
    date_time = pd.to_datetime(date)
    return (date_time - start_date).days

def piecewise(t, ts, fs, steepness=0.2):
    """Smooth approximation of a stepwise function using tanh transitions, to avoid problems with JAX"""
    result = fs[0]
    for i in range(1, len(ts)):
        # Smooth step using tanh
        transition = 0.5 * (1 + jnp.tanh(steepness * (t - ts[i])))
        result = result * (1 - transition) + fs[i] * transition
    return result

def exponential_recovery(t, ts, fs, rs, steepness=0.2):
    """Sudden tanh reduction at ts[i] with exponential recovery to fs[0]"""
    result = fs[0]
    for i in range(1, len(ts)):
        # Sudden tanh reduction at ts[i]
        reduction = 0.5 * (1 + jnp.tanh(steepness * (t - ts[i])))
        # Exponential recovery back to fs[0]
        recovery = jnp.exp(-rs[i-1] * jnp.maximum(0, t - ts[i]))
        # Combine: reduce to fs[i], then recover toward fs[0]
        transition = fs[i] + (fs[0] - fs[i]) * (1 - recovery)
        result = result * (1 - reduction) + transition * reduction
    return result

def get_jax_arrays_from_vax_csv(file, age_group_names=['<3m','3-11m','1-4y','5-7y','8-39y','40-64y','>=65y']):
    vax_df = pd.read_csv(file, parse_dates=['month_start'])
    vax_df.loc[:,'time'] = (vax_df['month_start'] - pd.to_datetime('1970-01-01')).dt.days
    if age_group_names is not None:
        vax_pivot = vax_df.pivot(index='time', columns='age_group', values='rate')
        # Reorder columns to match age_group_names order
        vax_pivot = vax_pivot.reindex(columns=age_group_names)
    else:
        vax_pivot = vax_df[["time", "cumulative_proportion_vaccinated"]].set_index("time")
    vax_idx = jnp.array(vax_pivot.index)
    vax_np = jnp.array(vax_pivot.values)
    return vax_idx, vax_np

#### RSV vax rates #####
RSV_VAX_RATE_IDX, RSV_VAX_RATE_NP = get_jax_arrays_from_vax_csv('Data/RSV_vaccination_rates_by_month_and_age_group.csv')
RSV_VAX_EFF = 0.73 # CDC website
# nirsevimab rates
NIRSEVIMAB_RATE_IDX, NIRSEVIMAB_RATE_NP = get_jax_arrays_from_vax_csv('Data/Nirsevimab_rates_by_month_and_age_group.csv')
NIRSEVIMAB_EFF = 0.98 # Hsiao et al., Pediatrics 2025
def rsv_eff_vax_rate(t_arr, max_eff0, max_eff1):
    # RSV vaccination rate
    vax_indices = jnp.searchsorted(RSV_VAX_RATE_IDX, t_arr, side='right')
    rate_vax = RSV_VAX_RATE_NP[vax_indices]
    # Nirsevimab administration rate
    nirsev_indices = jnp.searchsorted(NIRSEVIMAB_RATE_IDX, t_arr, side='right')
    rate_nirsev = NIRSEVIMAB_RATE_NP[nirsev_indices]
    # adjusted efficacy
    adj_eff_vax = jnp.maximum(RSV_VAX_EFF / max_eff1, 1.0)
    adj_eff_nirsev = jnp.maximum(NIRSEVIMAB_EFF / max_eff0, 1.0)
    # Effective rates
    eff_rate_vax = rate_vax * adj_eff_vax
    eff_rate_nirsev = rate_nirsev * adj_eff_nirsev
    total_rate = eff_rate_vax + eff_rate_nirsev
    return total_rate

#### RSV maternal immunity rates #####
MATERNAL_PVAX_IDX, MATERNAL_PVAX_NP = get_jax_arrays_from_vax_csv('Data/RSV_pregnant_vaccination_by_month.csv', age_group_names=None)
MATERNAL_VAX_EFF = 0.82 # Kampmann et al., NEJM 2023
def rsv_maternal_immunity(t_arr):
    vax_indices = jnp.searchsorted(MATERNAL_PVAX_IDX, t_arr, side='right')
    p = MATERNAL_PVAX_NP[vax_indices].squeeze()
    eff_p = p * MATERNAL_VAX_EFF
    return eff_p

# ===================================
# 3. PROCESSING AND PLOTTING OUTPUTS
# ===================================
from matplotlib import pyplot as plt
from matplotlib import cm as colormaps

def calculate_expected_obs(values, p_time_to_obs, length, NAG=7):
    """Calculate expected observations from the simulated values, using convolution with the probability of detection over time.
    Args:
        values: array of shape (time_points, state_variables) containing the simulated state variables over time
        p_time_to_obs: array of shape (max_time_to_obs,) containing the probability of detection at each time after infection
        length: number of time points to calculate expected observations for
        NAG: number of age groups (default 7, can be 8 with split option)
    Returns:
        expected_obs: array of shape (length,) containing the expected observations at each time point
    """
    trajectory = jnp.diff(values[-NAG:,:],axis=1).T
        # convolution of trajectory with probability of detection at each day after infection to get expected observations on each day
    p_time_to_obs_flipped = jnp.flip(p_time_to_obs.flatten())
    def obs_convolution(x):
        return jnp.convolve(x, p_time_to_obs_flipped, mode='same')
        # the expected observations for a given date are the observations on each day i days prvious multiplied by the probability of detection i days after infection
    expected_obs = jax.vmap(obs_convolution, in_axes=1, out_axes=1)(trajectory)
    expected_obs = jax.nn.softplus(expected_obs[-length:]*100)/100
    return expected_obs

def lockdown_incidence_plot(
    ax,state0,params,label='Observed cases',color='#648FFF',linewidth=1,alpha=1,
    by_age=False,AGE_GROUP_NAMES=None,select_age_group=None,NAG=7,
    solution=None,obs=None,points=None,times=None,
    start_t=date_to_t('2015-10-01'),end_t=date_to_t('2025-05-01'),factor=1,p_time_to_obs=[1],
):
    """Plot the incidence curve from the simulation, optionally by age group.
    Args:
        ax: matplotlib axis to plot on
        state0: initial state (flattened) for the simulation
        params: tuple of parameters to pass to the ODE function
        label: label for the plot legend
        color: color for the plot
        linewidth: linewidth for the plot
        alpha: transparency for the plot
        by_age: whether to plot separate lines for each age group
        AGE_GROUP_NAMES: list of names for the age groups (required if by_age is True)
        select_age_group: if not None, index or name of the single age group to plot (overrides by_age)
        NAG: number of age groups (default 7, can be 8 with split option)
        solution: if not None, a precomputed solution object from run_simulation to use instead of running a new simulation
        obs: if not None, an array of shape (time_points, NAG) containing the expected observations to plot instead of calculating from the simulation
        points: if not None, an array of time points to use for the x-axis and to extract values from the solution (must be the same as the saveat_ts used in run_simulation if solution is provided)
        times: if not None, an array of time points corresponding to the solution values (overrides points)
        start_t: start time index for plotting
        end_t: end time index for plotting
        factor: multiplier for the expected observations (e.g., to convert to cases per 100,000 population)
        p_time_to_obs: array of shape (max_time_to_obs,) containing the probability of detection at each time after infection, used to calculate expected observations from the simulation if obs is not provided
    Returns:
        mx: maximum y-value plotted (after applying factor), useful for setting y-axis limits
    """
    if solution is None:
        solution = run_simulation(params, state0, t1=points[-1], saveat_ts=points)
    if times is None:
        times = solution.ts
    values = solution.ys.T
    dates = [t_to_date(t) for t in times]
    start_index = np.argmin(times <= start_t)
    end_index = np.argmin(times <= end_t)
    if end_index <= start_index:
        end_index = len(times)
    expected_obs = calculate_expected_obs(values, p_time_to_obs, length=len(times), NAG=NAG) if obs is None else obs

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
        obs = factor * expected_obs

        if selected_age_idx is None:
            for i_age in range(NAG):
                ax.plot(
                    dates[(start_index + 1):end_index],
                    obs[start_index:end_index, i_age] / pop_size_by_age[start_index:end_index, i_age],
                    label=AGE_GROUP_NAMES[i_age] if AGE_GROUP_NAMES is not None else f"Age {i_age}",
                    color=hsv_colors[i_age],
                    linewidth=linewidth,
                    alpha=alpha,
                )
            mx = 1.1 * np.max(np.max(obs / pop_size_by_age, axis=1)[start_index:end_index])
        else:
            series = obs[start_index:end_index, selected_age_idx] / pop_size_by_age[start_index:end_index, selected_age_idx]
            age_label = (
                AGE_GROUP_NAMES[selected_age_idx]
                if AGE_GROUP_NAMES is not None
                else f"Age {selected_age_idx}"
            )
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
        ax.plot(
            dates[(start_index + 1):end_index],
            obs[start_index:end_index],
            label=label,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
        mx = 1.1 * np.max(np.array(obs[start_index:end_index]))
    return mx

def calculate_population_size(values, N_S=3, NAG=7):
    """
    Calculate population size from model output
    """
    population_size = jnp.sum(values[1:-NAG].reshape(2*N_S, NAG, -1), axis=0).T
    population_size = population_size.at[:,0].add(values[0,:])
    return population_size

# ===========================================
# 4. EXAMPLE PLOT WITH CALIBRATED PARAMETERS
# ===========================================

if __name__ == "__main__":
    ### Define model parameters
    NAG, N_S = 8, 3 # eight age groups and three susceptibility classes
    x = jnp.array([[0.20298816, 0.04762436, 0.2037047,  0.00330361, 0.47083195, 0.16601894,
                    0.26167706, 0.00282628, 0.3368089,  0.14488223, 0.0653761,  0.00694061,
                    0.00939794, 0.03575595, 0.35374895]])
    params = x_to_params(x[0], pathogen="RSV", lockdown="ExponentialODipEqual", option1="dedupsplit", option2="maxagep028",
                         print_params=True)
    p_time_to_obs = jnp.asarray(pd.read_csv("Data/RSV_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
    ### Define initial conditions
    if NAG == 7:
        AGE_GROUP_NAMES = ['<3m','3-11m','1-4y','5-7y','8-39y','40-64y','>=65y']
        CENSUS_AGE_POP = jnp.asarray([11089.338, 33268.016, 180948.38, 146224.97, 1698971.1, 1259997.4, 693042.4])
    else:
        AGE_GROUP_NAMES = ['<3m','3-11m','1-4y','5-7y','8-17y','18-39y','40-64y','>=65y']
        CENSUS_AGE_POP = jnp.asarray([11089.338, 33268.016, 180948.38, 146224.97, 512718.0, 1186252.9, 1259997.4, 693042.4])
    ## Initial conditions
    STATE0 = jnp.zeros((2*3+1,NAG))
    STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
    STATE0 = STATE0.at[1,:].set(1)
    # flatten initial state and add maternal immunity compartment
    STATE0 = STATE0.flatten()
    STATE0 = jnp.concatenate((jnp.array([0]), STATE0))
    # time points for simulation
    START = pd.to_datetime('2015-07-04') 
    END = pd.to_datetime('2025-05-01')
    PERIOD = pd.date_range(start=START, end=END, freq='D')
    POINTS = jnp.array(date_to_t(PERIOD))
    ### Run simulation from 1970-01-01 to END, saving only after START
    import time
    start_time = time.time()
    solution = jax.jit(run_simulation, static_argnums=(4,))(params, STATE0, POINTS[-1], POINTS, NAG)
    print(f"JIT-compiled simulation completed in {time.time() - start_time:.2f} seconds.")
    print("Note that the first run will be slower due to JIT compilation. Use jax.vmap across multiple parameter sets for much faster execution.")
    observations = calculate_expected_obs(solution.ys.T, p_time_to_obs, length=len(POINTS), NAG=NAG)
    ### Plot results
    fig = plt.figure(figsize=(13,7))
    
    # First row: single large panel with total trajectory
    ax_total = fig.add_subplot(3,1,1)
    lockdown_incidence_plot(ax_total, STATE0, params, label='Simulated cases', color='#648FFF', linewidth=2, alpha=1,
                            by_age=False, AGE_GROUP_NAMES=AGE_GROUP_NAMES, select_age_group=None, NAG=NAG,
                            solution=solution, obs=observations, points=POINTS,
                            start_t=date_to_t('2015-10-01'), end_t=date_to_t('2025-05-01'), factor=100000,
                            p_time_to_obs=p_time_to_obs)
    ax_total.set_title('Total RSV Incidence')
    ax_total.set_xlabel('Date')
    ax_total.set_ylabel('Incidence per 100,000')
    
    gs = fig.add_gridspec(2, 4, top=0.58, bottom=0.08)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(2)]
    hsv_colors = colormaps.hsv(-0.02+np.arange(NAG)/NAG)
    hsv_colors[3] = colormaps.hsv((3/NAG)+0.28/NAG)
    # Second and third rows: individual age groups (0-7)
    for i_age in range(NAG):
        row = i_age // 4
        col = (i_age % 4)
        ax = axes[row][col]
        lockdown_incidence_plot(ax, STATE0, params, label=AGE_GROUP_NAMES[i_age], color=hsv_colors[i_age], linewidth=2, alpha=1,
                                by_age=True, AGE_GROUP_NAMES=AGE_GROUP_NAMES, select_age_group=i_age, NAG=NAG,
                                solution=solution, obs=observations, points=POINTS,
                                start_t=date_to_t('2015-10-01'), end_t=date_to_t('2025-05-01'), factor=100000,
                                p_time_to_obs=p_time_to_obs)
        ax.set_title(f'{AGE_GROUP_NAMES[i_age]}', fontsize=10)
        if row:
            ax.set_xlabel('Date')
            ax.set_xticklabels([label.get_text() if i % 2 == 0 else '' for i, label in enumerate(ax.get_xticklabels())])
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        if col:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Incidence per 100,000')
    
    plt.savefig('Figures/RSV_simulation_example.png')