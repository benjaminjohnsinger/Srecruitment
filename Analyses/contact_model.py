# import jax.numpy as jnp
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

# @jit
def STATIC(t):
    return 1

# @jit
def STEP(t,t_lockdown,duration,reduction):
    return 1-reduction if t < t_lockdown + duration and t > t_lockdown else 1

# @jit
def RAMP(t,t_lockdown,duration,recovery_duration,reduction):
    return 1-reduction if t < t_lockdown + duration and t > t_lockdown else 1 + reduction*((t-t_lockdown-duration)/recovery_duration - 1) if t > t_lockdown and t < t_lockdown + duration + recovery_duration else 1

# @jit
# def piecewise(t,ts,fs):
#     # assume five elements in ts and fs
#     return jnp.where(
#         (t >= ts[0]) & (t < ts[1]), fs[0],
#         jnp.where(
#             (t >= ts[1]) & (t < ts[2]), fs[1],
#             jnp.where(
#                 (t >= ts[2]) & (t < ts[3]), fs[2],
#                 jnp.where(
#                     (t >= ts[3]) & (t < ts[4]), fs[3],
#                     fs[4]
#                 )
#             )
#         )
#     )

@jax.jit
def piecewise(t, ts, fs, steepness=0.2):
    """Smooth approximation using sigmoid transitions, to avoid problems with JAX"""
    result = fs[0]
    for i in range(1, len(ts)):
        # Smooth step using tanh
        transition = 0.5 * (1 + jnp.tanh(steepness * (t - ts[i])))
        result = result * (1 - transition) + fs[i] * transition
    return result

if __name__ == "__main__":
    # plot piecewise cm
    import matplotlib.pyplot as plt
    def date_to_t(date, start_date=pd.to_datetime("1970-01-01")):
        """
        Convert date to time index
        """
        date_time = pd.to_datetime(date)
        return (date_time - start_date).days
    fig, ax = plt.subplots(figsize=(10, 5))
    t_values = jnp.linspace(17500, date_to_t("2024-10-01"), 1000)
    ts = jnp.array([date_to_t("1970-01-01"),
        date_to_t('2020-03-19'), # Newsom announces stay-at-home order
        date_to_t('2021-04-27'), # CDC amends mask guidance to allow vaccinated individuals to go maskless
        date_to_t('2021-12-15'), # CDC reinstates mask guidance
        date_to_t('2022-03-01')]) # End of mask mandate in California
    fs = jnp.array([1,0.99,1,0.995,1])
    ax.plot(t_values, piecewise(t_values, ts, fs, steepness=0.2), label='Piecewise Contact Model', color='blue')
    ax.plot(t_values, piecewise(t_values, ts, fs, steepness=0.1), label='Piecewise Contact Model', color='green')
    ax.plot(t_values, piecewise(t_values, ts, fs, steepness=0.01), label='Piecewise Contact Model', color='red')
    plt.show()

# MOBILITY2020 = pd.read_csv('Data/Raw/Google_mobility_reports/2020_US_Region_Mobility_Report.csv', delimiter=',')
# MOBILITY2021 = pd.read_csv('Data/Raw/Google_mobility_reports/2021_US_Region_Mobility_Report.csv', delimiter=',')
# MOBILITY2022 = pd.read_csv('Data/Raw/Google_mobility_reports/2022_US_Region_Mobility_Report.csv', delimiter=',')
# MOBILITY = pd.concat([MOBILITY2020, MOBILITY2021, MOBILITY2022])
# MOBILITY_CA = MOBILITY.loc[MOBILITY['iso_3166_2_code'] == 'US-CA']
# MOBILITY_CA = MOBILITY_CA.sort_values(by='date')
# MOBILITY_CA.index = (pd.to_datetime(MOBILITY_CA['date'])-pd.to_datetime('1970-01-01')).dt.days
# MOBILITY_WORK = 1+MOBILITY_CA['workplaces_percent_change_from_baseline']/100
# MOBILITY_WORK_MA = MOBILITY_WORK.rolling(window=28).mean()
# MOBILITY_WORK_MA = MOBILITY_WORK_MA.bfill()
# RELATIVE_CONTACT_WORK = 1.3169 - 4.7718*MOBILITY_WORK_MA + 5.7062*MOBILITY_WORK_MA**2
# IDX = jnp.array(MOBILITY_CA.index)
# RELATIVE_CONTACT_WORK_NP = jnp.array(RELATIVE_CONTACT_WORK)
# # print(RELATIVE_CONTACT_WORK_NP)
# # @jit
# def google_prestige_work(t, RELATIVE_CONTACT_WORK_NP=RELATIVE_CONTACT_WORK_NP, IDX=IDX):
#     # if t<18355: # if before first dip below baseline
#     #     return 1
#     # elif t<18952: # if before first recovery to 95% of baseline
#     # if t < max(IDX):
#     return jnp.minimum(1,RELATIVE_CONTACT_WORK_NP[jnp.argmin(IDX<=t)])
#     # else:
#     #     return RELATIVE_CONTACT_WORK_NP[-1]
#     # else:
#     #     return 1
# # print([RELATIVE_CONTACT_WORK_NP[jnp.argmin(IDX<=t)] for t in [18383,18506,18809,19024,19631]])