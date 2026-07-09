import jax
import jax.numpy as jnp
import numpy as np
import jax.scipy as jsp
from matplotlib import cm as colormaps
hsv_colors = colormaps.hsv(-0.02+np.arange(7)/7)
hsv_colors[3] = colormaps.hsv((3/7)+0.04)
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController, ConstantStepSize, DirectAdjoint, RecursiveCheckpointAdjoint

from JAX_ODEs import deltas
N_C = 2
# NAG = 7
N_S = 3
from utils import calculate_population_size, sum_age_to
from plotting import calculate_observations_per_season_jax, get_season_start_jax

def run_simulation(params, y0, t1, saveat_ts, constant_step=False, hessian=False, NAG=7):
    # add NAG to end of params to pass to ODE solver
    sim_params = params + (NAG,)
    term = ODETerm(deltas)
    solver = Dopri5()
    saveat = SaveAt(ts=saveat_ts)
    if hessian:
        adjoint = DirectAdjoint()
    else:
        adjoint = RecursiveCheckpointAdjoint()
    if constant_step:
        step_controller = ConstantStepSize()
        # if constant_step is a float, use that as the step size, otherwise use 0.05
        if isinstance(constant_step, float):
            dt0 = constant_step
        else:
            dt0 = 0.05
        max_steps = int(t1/dt0) + 1
    else:
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

## POINTS must start  (at least) len(p_time_to_obs) days before the first observation to avoid issues from jnp.roll behaviour
def SIS_likelihood(data, daily_hospitalization_rates, params, POINTS, STATE0, p_time_to_obs, incidence_data=False, obs_age=None, mask=[3135,3288], solution=None, constant_step=False, hessian=False, return_sum=True, NAG=7, AGE_GROUPS=None, max_month=None):
    # run simulation
    if solution is None:
        t1 = int(POINTS[-1])
        values = run_simulation(params, STATE0, t1, POINTS, constant_step=constant_step, hessian=hessian, NAG=NAG)
        values = values.ys.T
    else:
        values = solution.ys.T

    population_size_inital = calculate_population_size(values, N_S=N_S, NAG=NAG)

    if incidence_data or (obs_age is None):
        expected_obs_initial = calculate_expected_obs(values, p_time_to_obs, len(data), NAG=NAG)

    if (AGE_GROUPS is not None) and (max_month is not None):
        population_size = sum_age_to(population_size_inital, max_month, AGE_GROUPS)
        expected_obs = sum_age_to(expected_obs_initial, max_month, AGE_GROUPS)
        NAG = len(AGE_GROUPS)
    else:
        population_size = population_size_inital
        expected_obs = expected_obs_initial

    if incidence_data:
        incidence = jnp.round(data*population_size[-len(data):])
        # exclude date range from likelihood calculation
        masked_incidence = jnp.ones((len(data) - (mask[1] - mask[0]),NAG))
        masked_expected_obs = jnp.ones((len(data) - (mask[1] - mask[0]),NAG))
        masked_incidence = masked_incidence.at[:mask[0]].set(incidence[:mask[0]]).at[mask[0]:].set(incidence[mask[1]:])
        masked_expected_obs = masked_expected_obs.at[:mask[0]].set(expected_obs[:mask[0]]).at[mask[0]:].set(expected_obs[mask[1]:])
        # calculate Poisson likelihood of observed incidence given expected incidence
        likelihood = jsp.stats.poisson.logpmf(masked_incidence, masked_expected_obs)
    else:
        if obs_age is not None:
            # trajectory is total proportion infected over time
            infectious = jnp.sum(values[1:].reshape((2*N_S+1, NAG, -1))[1:2*N_S:2], axis=0).T
            expected_infectious = jax.nn.softplus(infectious[-len(data):]*100)/100
            expected_ratio = jnp.divide(expected_infectious, population_size[-len(data):] * obs_age)
            expected_positivity = jnp.clip(expected_ratio, 1e-10, 0.99)
        else:
            # probability of getting a positive test in hospital is expected_obs / population size over time
            expected_ratio = jnp.divide(expected_obs, population_size[-len(data):])
            # then condition by baseline probabilty of hospitalization
            expected_positivity = jnp.clip(jnp.divide(expected_ratio, jnp.maximum(daily_hospitalization_rates[-len(data):], 1e-10)),
                                           1e-10,0.99)
        # exclude date range from likelihood calculation
        masked_tests = jnp.ones((len(data)- (mask[1] - mask[0]),NAG,2))
        masked_expected_positivity = jnp.ones((len(data)- (mask[1] - mask[0]),NAG))
        masked_tests = masked_tests.at[:mask[0]].set(data[:mask[0]]).at[mask[0]:].set(data[mask[1]:])
        masked_expected_positivity = masked_expected_positivity.at[:mask[0]].set(expected_positivity[:mask[0]]).at[mask[0]:].set(expected_positivity[mask[1]:])
        # calculate binomial likelihood of observed positives given expected proportion positive and total tests
        likelihood = jsp.stats.binom.logpmf(masked_tests[...,1], masked_tests[...,0], masked_expected_positivity)
    if return_sum:
        return jnp.sum(likelihood)
    else:
        return likelihood
    
def calculate_expected_obs(values, p_time_to_obs, length, NAG=7):
    trajectory = jnp.diff(values[-NAG:,:],axis=1).T
        # convolution of trajectory with probability of detection at each day after infection to get expected observations on each day
    p_time_to_obs_flipped = jnp.flip(p_time_to_obs.flatten())
    def obs_convolution(x):
        return jnp.convolve(x, p_time_to_obs_flipped, mode='same')
        # the expected observations for a given date are the observations on each day i days prvious multiplied by the probability of detection i days after infection
    expected_obs = jax.vmap(obs_convolution, in_axes=1, out_axes=1)(trajectory)
    expected_obs = jax.nn.softplus(expected_obs[-length:]*100)/100
    return expected_obs

def peaks_and_times_likelihood(obs_per_season, peak_times, params, POINTS, STATE0, p_time_to_obs, incidence_data=False, obs_age=None, mask=[3135,3288], solution=None, hessian=False, return_sum=True, NAG=7):
    # run simulation
    if solution is None:
        t1 = int(POINTS[-1])
        solution = run_simulation(params, STATE0, t1, POINTS, hessian=hessian, NAG=NAG)
    values = solution.ys.T
    times = solution.ts

    population_size = calculate_population_size(values, N_S=N_S, NAG=NAG)
    obs_per_season = obs_per_season * population_size[-1]

    expected_obs = calculate_expected_obs(values, p_time_to_obs, len(times), NAG=NAG)
    cut_times = times[:-1]
    expected_obs_per_season = calculate_observations_per_season_jax(expected_obs, age_groups=True)
    # Assign each time point to a season (numeric season id/start)
    season_ids = jax.vmap(get_season_start_jax)(cut_times)
    unique_seasons = 16684 + 365 * jnp.arange(10)  # Assuming seasons start on day 259 of each year
    # For each season, find the time index of the peak expected observation (per age group)
    def season_peak_times(season_id):
        season_mask = season_ids == season_id                           # (T,)
        masked_obs = jnp.where(season_mask[:, None], expected_obs, -jnp.inf)  # (T, NAG)
        peak_idx = jnp.argmax(masked_obs, axis=0)                      # (NAG,)
        return times[peak_idx]                                          # (NAG,)
    expected_peak_times = jax.vmap(season_peak_times)(unique_seasons)   # (n_seasons, NAG)
    # calculate likelihood based on how close expected_obs_per_season is to obs_per_season and how close expected_peak_times is to peak_times
    season_likelihood = -jnp.sum((expected_obs_per_season - obs_per_season)**2)
    peak_time_likelihood = -jnp.sum((expected_peak_times - peak_times)**2)
    total_likelihood = season_likelihood / jnp.sum(obs_per_season**2) + peak_time_likelihood / (365**2 * 10)
    if return_sum:
        return total_likelihood
    else:
        return season_likelihood, peak_time_likelihood