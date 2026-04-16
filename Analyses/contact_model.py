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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    TT = [0, 18340]
    FF = [1, 0.5]
    RR = [0.007,]
    EPOCH = pd.to_datetime('1970-01-01')
    END = pd.to_datetime("2025-05-01")
    FULL_PERIOD = pd.date_range(start=EPOCH, end=END, freq='D')
    FULL_POINTS = jnp.array(date_to_t(FULL_PERIOD))
    ec = exponential_recovery(FULL_POINTS, TT, FF, RR)
    SEASONALITY = 0.9
    OFFSET = 0.3
    rc = ec*(1+SEASONALITY*jnp.cos(2*jnp.pi*((FULL_POINTS-274)/365-OFFSET)))
    plt.plot(FULL_POINTS, rc)
    plt.show()

    # print(date_to_t('2020-01-01'))
    # print(MOBILITY_START, MOBILITY_END)
    # import matplotlib.pyplot as plt
    # contact_factor = 1 + 1*MOBILITY_CHANGE_JAX
    # FULL_POINTS = jnp.arange(19632)
    # MOBILITY_CONTACT = jnp.ones(len(FULL_POINTS))
    # MOBILITY_CONTACT = MOBILITY_CONTACT.at[MOBILITY_START:MOBILITY_END+1].set(contact_factor)
    # plt.plot(FULL_POINTS, MOBILITY_CONTACT)
    # plt.show()

    # SC_COUNTIES = ['Kern County', 'Ventura County', 'Los Angeles County', 'Orange County', 'Riverside County', 'San Bernardino County', 'San Diego County'] # https://southerncalifornia.permanente.org/

    # TAUBE_CONTACTS = pd.read_csv('Data/Raw/baseline_contact_by_county_week.csv', delimiter=',', encoding='latin1')
    # # TAUBE_CONTACTS = TAUBE_CONTACTS.loc[TAUBE_CONTACTS['name'].isin(SC_COUNTIES)]
    # # TAUBE_CONTACTS = TAUBE_CONTACTS.loc[TAUBE_CONTACTS['state'] == 'CA']
    # TAUBE_CONTACTS['week'] = pd.to_datetime(TAUBE_CONTACTS['week'])
    # TAUBE_CONTACTS = TAUBE_CONTACTS.sort_values(by='week')
    # # get rid of NAs
    # TAUBE_CONTACTS = TAUBE_CONTACTS.dropna(subset=['contact_fit', 'samp_size'])
    # # for each week, average "contact_fit" weighted by "samp_size"
    # TAUBE_CONTACTS = TAUBE_CONTACTS.groupby('week').apply(lambda x: np.average(x['contact_fit'], weights=x['samp_size'])).reset_index()
    # TAUBE_CONTACTS.index = (TAUBE_CONTACTS['week']-pd.to_datetime('1970-01-01')).dt.days
    # # fill days between weeks
    # # TAUBE_CONTACTS = TAUBE_CONTACTS.reindex(pd.date_range(start=TAUBE_CONTACTS.index[0], end=TAUBE_CONTACTS.index[-1], freq='D'))
    # # TAUBE_CONTACTS = TAUBE_CONTACTS.interpolate(method='linear')
    # TAUBE_CONTACTS_JAX = jnp.array(TAUBE_CONTACTS.iloc[:,1])
    # # # plot piecewise cm

    # import matplotlib.pyplot as plt
    # # plt.plot(TAUBE_CONTACTS.iloc[:,0], TAUBE_CONTACTS.iloc[:,1]/13.85 - 1, label='Contact Survey Data', color='black')
    # # # one week rolling average of mobility change
    # # MOBILITY_CHANGE_NONRESIDENTIAL = MOBILITY_CHANGE_NONRESIDENTIAL.rolling(window=7, center=True).mean()
    # plt.plot(MOBILITY_CHANGE_NONRESIDENTIAL.index, MOBILITY_CHANGE_NONRESIDENTIAL.values, label='Google Mobility Data', color='orange')
    # # # plt.legend()
    # plt.show()

    # # fit quadratic model of mobility to contact data
    # coeffs = np.polyfit(MOBILITY_CHANGE_NONRESIDENTIAL.loc[TAUBE_CONTACTS.index].values, TAUBE_CONTACTS.iloc[:,1]/13.85 - 1, 2)
    # print("Fitted quadratic coefficients:", coeffs)
    # # plot fitted model
    # coeffs[2] += 1
    # plt.plot(MOBILITY_CHANGE_NONRESIDENTIAL.index, coeffs[0]*MOBILITY_CHANGE_NONRESIDENTIAL.values**2 + coeffs[1]*MOBILITY_CHANGE_NONRESIDENTIAL.values + coeffs[2], label='Fitted Quadratic Model', color='blue')
    # plt.legend()
    # plt.xlabel('Date')
    # plt.ylabel('Relative Contact Rate Change')
    # plt.title('Contact Rate Change vs Google Mobility Data')
    # plt.show()


    # def date_to_t(date, start_date=pd.to_datetime("1970-01-01")):
    #     """
    #     Convert date to time index
    #     """
    #     date_time = pd.to_datetime(date)
    #     return (date_time - start_date).days
    # fig, ax = plt.subplots(figsize=(10, 5))
    # t_values = jnp.linspace(17500, date_to_t("2024-10-01"), 1000)
    # ts = jnp.array([date_to_t("1970-01-01"),
    #     date_to_t('2020-03-19'), # Newsom announces stay-at-home order
    #     date_to_t('2021-04-27'), # CDC amends mask guidance to allow vaccinated individuals to go maskless
    #     date_to_t('2021-12-15'), # CDC reinstates mask guidance
    #     date_to_t('2022-03-01')]) # End of mask mandate in California
    # fs = jnp.array([1,0.99,1,0.995,1])
    # ax.plot(t_values, piecewise(t_values, ts, fs, steepness=0.2), label='Piecewise Contact Model', color='blue')
    # ax.plot(t_values, piecewise(t_values, ts, fs, steepness=0.1), label='Piecewise Contact Model', color='green')
    # ax.plot(t_values, piecewise(t_values, ts, fs, steepness=0.01), label='Piecewise Contact Model', color='red')
    # plt.show()
