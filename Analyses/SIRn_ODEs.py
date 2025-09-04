## Ordinary differential equations defining SIR models with n susceptibility classes
## BJS September 2024

# import jax.numpy as jnp
from numba import jit
N_C = 3 # three types of compartment: susceptible, infected, recovered

# @jit
## Differential equations
def single_pathogen_deltas(t, state, NAG, N_S, AGING_RATE, birth_rate, WANE_UP, WANE_SAME, REC, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV, BCOV, arrivals, regional_positivity, IMPORT_RATE, BETA, SEASONALITY, OFFSET, contact):
    delta = jnp.zeros(state.shape)
    pop_size = jnp.sum(state)
    print(state)
    # age_pops = jnp.array([jnp.sum(state[range(i_age,(N_C*N_S+1)*NAG,NAG)],axis=0) for i_age in range(NAG)])
    age_pops = jnp.zeros(NAG)
    for i_age in range(NAG):
        for i in range(N_S):
            for j in range(N_C):
                # Sum the population in each age group across all susceptibility classes and compartments
                age_pops[i_age] += state[(N_C*i+j)*NAG + i_age]
    infectious_states = jnp.zeros((N_S, NAG))
    for j in range(N_S):
        infectious_states[j] = state[(3*j+2)*NAG:(3*j+3)*NAG]
    for i in range(N_S):
        # Susceptibile class i = birth - infection + waning + aging in - aging out +/- vaccination
        # WANE_UP[-1] is zero
        delta[(3*i+1)*NAG:(3*i+2)*NAG] = birth_vax(t,BCOV,S_VAX,NAG,N_S,N_C)[(3*i+1)*NAG:(3*i+2)*NAG]*birth_rate(t)*pop_size*jnp.concatenate((jnp.ones(1),jnp.zeros(NAG-1)))\
            - S_REL[i]*S_AGE*BETA*(jnp.dot(contact(t,SEASONALITY,OFFSET),jnp.sum(IMPORT_RATE*regional_positivity(t)*arrivals(t)*age_pops+infectious_states*I_REL,axis=0))/pop_size)*state[(3*i+1)*NAG:(3*i+2)*NAG]\
            + WANE_UP[i-1]*state[(3*i)*NAG:(3*i+1)*NAG] + WANE_SAME[i]*state[(3*i+3)*NAG:(3*i+4)*NAG]\
            - AGING_RATE*state[(3*i+1)*NAG:(3*i+2)*NAG] + jnp.concatenate((jnp.zeros(1), AGING_RATE[:-1]*state[(3*i+1)*NAG:(3*i+2)*NAG-1]))
        # Infectious class i = infection - recovery + aging in - aging out - vaccination + importations
        delta[(3*i+2)*NAG:(3*i+3)*NAG] = S_REL[i]*S_AGE*BETA*(jnp.dot(contact(t,SEASONALITY,OFFSET),jnp.sum(IMPORT_RATE*regional_positivity(t)*arrivals(t)+infectious_states*I_REL,axis=0))/pop_size)*state[(3*i+1)*NAG:(3*i+2)*NAG]\
            - REC[i]*state[(3*i+2)*NAG:(3*i+3)*NAG]\
            - AGING_RATE*state[(3*i+2)*NAG:(3*i+3)*NAG] + jnp.concatenate((jnp.zeros(1), AGING_RATE[:-1]*state[(3*i+2)*NAG:(3*i+3)*NAG-1]))\
        # Recovered class i = recovery - waning + aging in - aging out - vaccination
        delta[(3*i+3)*NAG:(3*i+4)*NAG] = REC[i]*state[(3*i+2)*NAG:(3*i+3)*NAG]\
            - (WANE_UP[i]+WANE_SAME[i])*state[(3*i+3)*NAG:(3*i+4)*NAG]\
            - AGING_RATE*state[(3*i+3)*NAG:(3*i+4)*NAG] + jnp.concatenate((jnp.zeros(1), AGING_RATE[:-1]*state[(3*i+3)*NAG:(3*i+4)*NAG-1]))
    return delta

# test that single_pathogen_deltas runs
if __name__ == "__main__":
    NAG = 7
    N_S = 3
    from Parameters.census_population import AGING_RATE
    from demography import birth_rate
    WANE_UP = jnp.array([1/365, 1/365, 0.0])
    WANE_SAME = jnp.array([0.0, 0.0, 1/365])
    REC = jnp.array([1/4.9, 1/4.1, 1/4.1])
    S_REL = jnp.array([1, 0.25, 0.025])
    S_AGE = jnp.ones(NAG)
    I_REL = jnp.array([[1], [1], [1]])
    P_OBS = jnp.array([1, 0.46, 0.31]) * 0.03
    S_VAX = 2
    from vaccination import birth_vax, flu_rate
    from Parameters.RSV import BCOV, ACOV, regional_positivity
    from mobility_and_import import arrivals
    CONTACT = jnp.genfromtxt('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', dtype=jnp.float64)
    IMPORT_RATE = 0.01
    BETA = 0.5
    SEASONALITY = 0.2
    OFFSET = 0.2
    # @jit
    def contact(t, seasonality, offset):
        return (1 + seasonality * jnp.cos(2 * jnp.pi * ((t - 274) / 365 - offset))) * CONTACT

    params = {
        "NAG": NAG,
        "N_S": N_S,
        "AGING_RATE": AGING_RATE,
        "birth_rate": birth_rate,
        "WANE_UP": WANE_UP,
        "WANE_SAME": WANE_SAME,
        "REC": REC,
        "S_REL": S_REL,
        "S_AGE": S_AGE,
        "I_REL": I_REL,
        "P_OBS": P_OBS,
        "birth_vax": birth_vax,
        "all_vax": flu_rate,
        "S_VAX": S_VAX,
        "ACOV": ACOV,
        "BCOV": BCOV,
        "regional_positivity": regional_positivity,
        "arrivals": arrivals,
        "IMPORT_RATE": IMPORT_RATE,
        "BETA": BETA,
        "SEASONALITY": SEASONALITY,
        "OFFSET": OFFSET,
        "contact": contact
    }

    from Parameters.census_population import CENSUS_AGE_POP
    STATE0 = jnp.zeros((1+N_S*3)*(NAG))
    STATE0[NAG:2*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
    STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

    # t = 0
    # delta = single_pathogen_deltas(t, STATE0, NAG, N_S, AGING_RATE, birth_rate,
    #                                WANE_UP, WANE_SAME, REC, S_REL, S_AGE,
    #                                I_REL, P_OBS, birth_vax, flu_rate,
    #                                S_VAX, ACOV, BCOV, arrivals,
    #                                regional_positivity,
    #                                IMPORT_RATE, BETA,
    #                                SEASONALITY, OFFSET, contact)
    # print(delta)
    # try solve_ivp
    from scipy.integrate import solve_ivp
    from utils import date_to_t
    import pandas as pd
    result = solve_ivp(
        single_pathogen_deltas,
        (date_to_t('1970-01-01'), date_to_t('2023-10-01')),
        STATE0,
        args=params.values(),
        t_eval=jnp.array(date_to_t(pd.date_range(pd.to_datetime('2015-10-01'), pd.to_datetime('2023-10-01'), freq='D'))),
        method='RK45'
    )
    print(result)