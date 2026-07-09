# import jax.numpy as jnp
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

def date_to_t(date, start_date=pd.to_datetime("1970-01-01")):
    """
    Convert date to time index
    """
    date_time = pd.to_datetime(date)
    return (date_time - start_date).days

def STATIC(t):
    return 1

def STEP(t,t_lockdown,duration,reduction):
    return 1-reduction if t < t_lockdown + duration and t > t_lockdown else 1

def RAMP(t,t_lockdown,duration,recovery_duration,reduction):
    return 1-reduction if t < t_lockdown + duration and t > t_lockdown else 1 + reduction*((t-t_lockdown-duration)/recovery_duration - 1) if t > t_lockdown and t < t_lockdown + duration + recovery_duration else 1

def piecewise(t, ts, fs, steepness=0.2):
    """Smooth approximation using tanh transitions, to avoid problems with JAX"""
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

def exponential_in_and_out(t, ts, fs, rs, steepness=0.2):
    """Gradual exponential reduction accelerating toward fs[1] by ts[1], then exponential recovery from ts[2]."""
    t = jnp.asarray(t)
    dt_01 = jnp.maximum(1e-6, ts[2] - ts[1])
    k = -jnp.log(1e-2) / dt_01
    time_to_reduction = jnp.maximum(0, ts[2] - t)
    reduced_curve = fs[0] - (fs[0] - fs[1]) * jnp.exp(-k * time_to_reduction)

    # Phase 2: exponential recovery back to fs[0] from ts[2]
    time_since_recovery = jnp.maximum(0, t - ts[2])
    recovered_curve = fs[0] + (fs[1] - fs[0]) * jnp.exp(-rs[0] * time_since_recovery)
    
    # Smooth transition at ts[2] using tanh
    transition = 0.5 * (1 + jnp.tanh(steepness * (t - ts[2])))
    result = reduced_curve * (1 - transition) + recovered_curve * transition
    return result

def exponential_recovery_byage(t, ts, fs, rs, age_partition, steepness=0.2, NAG=7):
    """Exponential recovery with different rates for two age groups"""
    result = jnp.ones((len(t), NAG)) * fs[0]
    
    for i in range(1, len(ts)):
        # Sudden tanh reduction at ts[i]
        reduction = 0.5 * (1 + jnp.tanh(steepness * (t - ts[i])))
        
        # Exponential recovery for first age group
        recovery_1 = jnp.exp(-rs[i-1, 0] * jnp.maximum(0, t - ts[i]))
        transition_1 = fs[i] + (fs[0] - fs[i]) * (1 - recovery_1)
        
        # Exponential recovery for second age group
        recovery_2 = jnp.exp(-rs[i-1, 1] * jnp.maximum(0, t - ts[i]))
        transition_2 = fs[i] + (fs[0] - fs[i]) * (1 - recovery_2)
        
        # Apply to respective age groups
        reduction_expanded = jnp.expand_dims(reduction, axis=1)
        result = result.at[:, :age_partition].set(
            result[:, :age_partition] * (1 - reduction_expanded) + jnp.expand_dims(transition_1, axis=1) * reduction_expanded
        )
        result = result.at[:, age_partition:].set(
            result[:, age_partition:] * (1 - reduction_expanded) + jnp.expand_dims(transition_2, axis=1) * reduction_expanded
        )
    
    return result

def sigmoid_recovery(t, ts, fs, rs, steepness=0.2):
    """Sudden tanh reduction at ts[i] with sigmoidal recovery to fs[0]"""
    result = fs[0]
    # Sudden tanh reduction at ts[i]
    reduction = (1 + jnp.tanh(steepness * (t - ts[1])))
    # Sigmoidal recovery back to fs[0]
    recovery = (1 + jnp.exp(-rs[0]*(ts[2]-ts[1]))) / (1 + jnp.exp(rs[0] * (t - ts[2])))
    # Combine: reduce to fs[i], then recover toward fs[0]
    transition = fs[1] + (fs[0] - fs[1]) * (1-recovery)
    result = result * (1 - reduction) + transition * reduction
    return result

MOBILITY2020 = pd.read_csv('Data/Raw/Google_mobility_reports/2020_US_Region_Mobility_Report.csv', delimiter=',')
MOBILITY2021 = pd.read_csv('Data/Raw/Google_mobility_reports/2021_US_Region_Mobility_Report.csv', delimiter=',')
MOBILITY2022 = pd.read_csv('Data/Raw/Google_mobility_reports/2022_US_Region_Mobility_Report.csv', delimiter=',')
MOBILITY = pd.concat([MOBILITY2020, MOBILITY2021, MOBILITY2022])
MOBILITY_CA = MOBILITY.loc[MOBILITY['iso_3166_2_code'] == 'US-CA']
MOBILITY_CA = MOBILITY_CA.sort_values(by='date')
MOBILITY_CA.index = (pd.to_datetime(MOBILITY_CA['date'])-pd.to_datetime('1970-01-01')).dt.days
MOBILITY_CHANGE_NONRESIDENTIAL = MOBILITY_CA[['retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline']].mean(axis=1)/100

MOBILITY_START = MOBILITY_CHANGE_NONRESIDENTIAL.index[0] + date_to_t('1970-01-01')
MOBILITY_END = MOBILITY_CHANGE_NONRESIDENTIAL.index[-1] + date_to_t('1970-01-01')
MOBILITY_CHANGE_JAX = jnp.array(MOBILITY_CHANGE_NONRESIDENTIAL)