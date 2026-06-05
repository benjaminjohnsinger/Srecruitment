## BJS June 2025
## JAX and NumPyro compatible ODES

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
    maternal_immunity = jnp.sum(interp_fn(t, FULL_POINTS, MATERNAL_IMMUNITY) * susceptible[:,NAG-3])  / jnp.sum(susceptible[:,NAG-3])
    delta_maternal = delta_maternal + maternal_immunity*birth_rate*pop_size
    delta = delta.at[0,0].add((1-maternal_immunity)*birth_rate*pop_size)
    # infections - calculate force of infection
    relative_contact = interp_fn(t, FULL_POINTS, RELATIVE_CONTACT)
    import_strength = jnp.interp(t, FULL_POINTS, IMPORT_STRENGTH)
    CONTACT_t = relative_contact[:, None] * relative_contact[None, :] * CONTACT_MATRIX
    infectious_by_age = jnp.sum(I_REL * infectious, axis=0)
    infectious_contact = jnp.dot(CONTACT_t, infectious_by_age/age_pops)
    import_contact = import_strength*jnp.sum(CONTACT_t, axis=1)
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
    if NAG == 65:
        maternal_agers = (1 / ((3 / 12) * 365)) * maternal
        delta_maternal = delta_maternal - maternal_agers
        delta = delta.at[0, 3].add(maternal_agers)
    else:
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

if __name__ == "__main__":
    import jax
    import numpy as np
    import scipy as sp
    import pandas as pd
    from matplotlib import pyplot as plt
    from Parameters.census_population import AGING_RATE
    from Parameters.census_population import CENSUS_AGE_POP
    from utils import date_to_t, t_to_date
    from contact_model import piecewise
    import time

    CONTACT_MATRIX = np.asarray(pd.read_csv('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', header=None).values)
    BIRTH_RATE = np.genfromtxt('Data/Processed/birth_rate_daily.csv', delimiter=',')
    ARRIVALS = np.genfromtxt('Data/Processed/arrivals_daily.csv', delimiter=',')
    POSITIVITY = np.genfromtxt('Data/Processed/RSV_positivity_daily.csv', delimiter=',')


    # # example parameters for testing
    NAG = 7
    N_S = 3
    DAYS = len(BIRTH_RATE)
    FULL_POINTS = jnp.arange(DAYS)
    # BIRTH_RATE = np.ones(DAYS)*np.mean(BIRTH_RATE) # np.mean(BIRTH_RATE)
    # ARRIVALS = np.ones(DAYS)*np.mean(ARRIVALS) # np.mean(ARRIVALS)
    # POSITIVITY = np.ones(DAYS)*np.mean(POSITIVITY) # np.mean(POSITIVITY)
    BETA = 3.09801199e-01
    # BETA = 0.4427481
    WANE = np.array([0.0, 0.0, 4.12833152e-03])
    # WANE = np.array([0.0, 0.0, 0.002982292])
    S_REL = np.array([1.0, 2.51378508e-01, 2.51378508e-01 * 1.93695805e-01])
    P_OBS = np.array([1, 0.46, 0.31])
    OBS_AGE = np.array([1.41156737e-03,1.42463892e-03,2.78237498e-03,1.34302008e-04,8.83682889e-05,3.16598659e-04,2.38996273e-03])
    REC_UP = np.array([1/4.9,1/4.1,0.0])
    REC_SAME = np.array([0.0,0.0,1/4.1])
    IMPORT_STRENGTH = 0.01*ARRIVALS/np.max(ARRIVALS)*POSITIVITY/30.44
    # IMPORT_STRENGTH = 1e-5*ARRIVALS*POSITIVITY
    # plt.plot(IMPORT_STRENGTH, label='Import Strength')
    # plt.show()

    SEASONALITY = 8.88627411e-02
    OFFSET = 1.82381144e-01
    # SEASONALITY = 0.10698847
    # OFFSET = 0.1400197 
    TT = np.array([date_to_t('1970-01-01'), date_to_t('2020-03-19'), date_to_t('2021-03-18'), date_to_t('2021-09-14'), date_to_t('2022-02-23')])
    FF = [1, 0.79711585, 0.94333315, 0.79516564, 0.95091717]
    PIECEWISE_CONTACT = np.array([piecewise(t, TT, FF, steepness=0.2) for t in range(DAYS)])
    RELATIVE_CONTACT = PIECEWISE_CONTACT*(1.0 + SEASONALITY*np.cos(2*np.pi*((np.arange(DAYS)-274)/365 - OFFSET)))
    # RELATIVE_CONTACT = np.ones(DAYS)
    # protection_param = S_REL * P_OBS
    # max_eff0 = 1 - protection_param[-1]
    # max_eff1 = (protection_param[-2]-protection_param[-1])/protection_param[-2]
    # from new_vax import rsv_eff_vax_rate
    # VAX_RATE = rsv_eff_vax_rate(FULL_POINTS, max_eff0, max_eff1)
    VAX_RATE = jnp.zeros((DAYS, NAG))
    # plt.plot(RELATIVE_CONTACT*np.sum(CONTACT_MATRIX), label='Relative Contact')
    # plt.savefig('Figures/new_contact_test.png')

    # from new_vax import rsv_maternal_immunity
    # MATERNAL_IMMUNITY = rsv_maternal_immunity(FULL_POINTS)
    MATERNAL_IMMUNITY = jnp.zeros(DAYS)

    # # dayrange = np.arange(date_to_t("2016-10-01"),date_to_t("2017-10-01"))
    # # fig, ax = plt.subplots(1,4, figsize=(12, 3))
    # # ax[0].plot(ARRIVALS[dayrange]/(30.44*np.max(ARRIVALS)), label='Arrivals')
    # # ax[0].set_title('Arrivals')
    # # ax[1].plot(POSITIVITY[dayrange], label='Positivity')
    # # ax[1].set_title('Positivity')
    # # ax[2].plot((RELATIVE_CONTACT*np.sum(CONTACT_MATRIX))[dayrange], label='Import Strength')
    # # ax[2].set_title('Relative Contact')
    # # ax[3].plot(BIRTH_RATE[dayrange], label='Import Strength')
    # # ax[3].set_title('Birth Rate')
    # # plt.tight_layout()
    # # plt.savefig('Figures/new_model_rates_test.png',dpi=300)

    # # initial state - everyone susceptible except one in each age group
    STATE0 = jnp.zeros((2*N_S+1,NAG))
    STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
    STATE0 = STATE0.at[1,:].set(1)
    # # flatten initial state and add maternal immunity compartment
    STATE0 = STATE0.flatten()
    STATE0 = jnp.concatenate((jnp.array([0]), STATE0))
    params = (FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX,
              BETA, WANE, S_REL, P_OBS, OBS_AGE, RELATIVE_CONTACT, VAX_RATE, MATERNAL_IMMUNITY,
              REC_UP, REC_SAME, IMPORT_STRENGTH)
    
    # start_time = time.time()
    # result = sp.integrate.solve_ivp(deltas,(0,DAYS-1),STATE0.flatten(),args=(params,),t_eval=jnp.arange(15000,DAYS),method='RK45')
    # print("Time:", time.time() - start_time)
    # values = result.y
    # times = result.t

    # import jax
    # jax_deltas = jax.jit(deltas)
    # start_time = time.time()
    # result_jax = sp.integrate.solve_ivp(jax_deltas,(0,DAYS-1),STATE0.flatten(),args=(params,),t_eval=jnp.arange(15000,DAYS),method='RK45')
    # print("JAX Time:", time.time() - start_time)
    # values = result_jax.y
    # times = result_jax.t
    from utils import parameters_from_DE
    import pickle
    pathogen = "InfluenzaA"
    option1 = "NA"
    option2 = "flexage"
    seed = 251103

    paramst, param_names, bounds, incidence, p_time_to_obs = parameters_from_DE(pathogen, "FlexStepwise", option1, option2, str(seed))
    # S_REL = paramst[6]
    # P_OBS = paramst[7]
    # protection_param = S_REL * P_OBS
    # max_eff0 = 1 - protection_param[-1]
    # max_eff1 = (protection_param[-2]-protection_param[-1])/protection_param[-2]
    # from new_vax import flu_eff_vax_rate
    # VAX_RATE = flu_eff_vax_rate(FULL_POINTS, max_eff1)

    # # set MATERNAL_IMMUNITY in paramst
    # paramst = (paramst[0], paramst[1], paramst[2], paramst[3],
    #            paramst[4], paramst[5], paramst[6], paramst[7], paramst[8], paramst[9], paramst[10], MATERNAL_IMMUNITY,
    #            paramst[12], paramst[13], paramst[14])
    
    # with open("Data/Processed/results"+str(seed)[:6]+"/DE_opt_"+pathogen+"FlexStepwise"+option1+option2+str(seed)+".pickle","rb") as f:
    #     opt = pickle.load(f)
    # print(opt.x)

    
    from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController

    term = ODETerm(deltas)
    solver = Dopri5()
    saveat = SaveAt(ts=jnp.arange(date_to_t('2015-10-01')-90,date_to_t('2025-05-01')))
    step_controller = PIDController(rtol=1e-5, atol=1e-5)
    print("Starting Diffrax solve...")
    time0 = time.time()
    solution = diffeqsolve(
                        term, solver,
                        t0=0, t1=DAYS-1, dt0=None, stepsize_controller=step_controller,
                        saveat=saveat, y0=STATE0, args=paramst, 
                        max_steps=100000,  
                        )
    print("Diffrax Time:", time.time() - time0)
    times = solution.ts
    values = solution.ys.T
    
    from fit_MCMC import SIS_likelihood
    incidence = np.asarray(pd.read_csv("Data/Processed/KPSC_ARI_RSV_incidence_age_daily.csv",index_col=0))
    p_time_to_obs = np.asarray(pd.read_csv("Data/Processed/RSV_incubation_admittance_distribution.csv",delimiter=',', header=None).values)
    POINTS = jnp.arange(date_to_t('2015-10-01')-90,date_to_t('2025-05-01'))
    likelihood = SIS_likelihood(incidence, params, POINTS, STATE0, p_time_to_obs, age=True, incidence=True, start_t=date_to_t(pd.to_datetime('1970-01-01')), overdispersion=False, solution=solution)
    print(likelihood)

    p_time_to_obs_flipped = jnp.flip(p_time_to_obs.flatten())
    def obs_convolution(x):
        return jnp.convolve(x, p_time_to_obs_flipped, mode='same')
    cumulative_observations = solution.ys[:,-NAG:]  # shape (DAYS, NAG)
    observations = jnp.diff(cumulative_observations, axis=0)  # shape (DAYS-1, NAG)
    expected_obs = jax.vmap(obs_convolution, in_axes=1, out_axes=1)(observations)

    # are there any negative values?
    print("Min:", jnp.min(values), jnp.argmin(values)%((2*N_S+1)*NAG)//NAG, jnp.argmin(values)%((2*N_S+1)*NAG)%NAG)
    
    maternal = values[0,:]
    other_values = values[1:,:]
    shaped_other_values = other_values.reshape((2*N_S+1,NAG,-1))
    shaped_values = shaped_other_values.at[0,0,:].add(maternal)
    age_pops = jnp.sum(shaped_values[:2*N_S, :, :], axis=0)

    from matplotlib import cm as colormaps
    fig,ax = plt.subplots(1,1, figsize=(6,6))
    hsv_colors = colormaps.hsv(-0.02+np.arange(7)/7)
    hsv_colors[3] = colormaps.hsv((3/7)+0.04)
    # plt.plot(result.t, np.diff(np.sum(result.y[-NAG:,:],axis=0)))
    # plt.plot(times, values[-NAG:,:].T)
    # ax[0].plot(times, maternal, color='black', label='Maternally Immune')
    for i in range(NAG):
        # plt.plot(times, values[i,:]+values[2*NAG+i,:]+values[4*NAG+i,:], color=hsv_colors[i], label="S age "+str(i))
        # plt.plot(times, values[NAG+i,:]+values[3*NAG+i,:]+values[5*NAG+i,:], '--', color=hsv_colors[i], label="I age "+str(i))
        ax.plot(times[1:], 30.44*10000*expected_obs[:,i]/age_pops[i,1:], color=hsv_colors[i], label="Obs age "+str(i))
    # ax.plot(times[1:], RELATIVE_CONTACT[times[1:].astype(int)]*0.1, color='black', label='Relative Contact')
    # x ticks at 1st of january every year
    xticks = np.array([date_to_t('20'+str(year)+'-01-01') for year in range(16,25)])
    xticklabels = [t_to_date(t).strftime('%Y') for t in xticks]
    ax.set_xticks(xticks, xticklabels)


    plt.savefig('Figures/InfluenzaA_DEparamtest_w_InfluenzaAvax.png', dpi=300)