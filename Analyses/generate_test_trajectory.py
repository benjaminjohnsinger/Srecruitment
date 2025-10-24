import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

N_C, N_S, NAG = 2, 3, 7
from Parameters.census_population import *
from Parameters.times_and_contacts import *
FULL_PERIOD = pd.date_range(start=EPOCH, end=END, freq='D')
FULL_POINTS = jnp.array(date_to_t(FULL_PERIOD))
POINTS = jnp.array(date_to_t(PERIOD))

## Initial conditions
STATE0 = jnp.zeros((2*N_S+1,NAG))
STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP-1)
STATE0 = STATE0.at[1,:].set(1)
# # flatten initial state and add maternal immunity compartment
STATE0 = STATE0.flatten()
STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

from utils import x_to_params
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController
from JAX_ODEs import deltas
import time

def generate_test_trajectory(x, seed, pathogen, lockdown, option1, option2):
    params = x_to_params(x, pathogen, lockdown, option1, option2, print_params=False)
    np.random.seed(seed)
    term = ODETerm(deltas)
    solver = Dopri5()
    saveat = SaveAt(ts=jnp.arange(date_to_t('2015-10-01')-90,date_to_t('2025-05-01')))
    step_controller = PIDController(rtol=1e-5, atol=1e-5)
    print("Starting Diffrax solve...")
    time0 = time.time()
    solution = diffeqsolve(
                        term, solver,
                        t0=0, t1=int(POINTS[-1]), dt0=None, stepsize_controller=step_controller,
                        saveat=saveat, y0=STATE0.flatten(), args=params, 
                        max_steps=None,  
                        )
    print("ODE integration time:",time.time()-time0)
    values = solution.ys.T
    times = solution.ts

    age_pops = np.array([np.sum(values[range(1+i_age,N_C*N_S*NAG,NAG),1:],axis=0) for i_age in range(NAG)]).T

    # apply poisson noise to values
    new_cases = jnp.diff(values[-NAG:],axis=1).T
    noisy_cases = np.random.poisson(lam=jnp.maximum(new_cases,0))
    noisy_incidence = noisy_cases/age_pops[np.argmax(POINTS>=date_to_t('2015-07-04')):np.argmax(POINTS>=date_to_t('2025-05-01'))]
    noisy_cases = noisy_cases.astype(int)[-3501:,:]
    noisy_incidence = noisy_incidence[-3501:,:]
    cropped_period = PERIOD[-len(noisy_incidence):]
    np.savetxt("Data/Processed/KPSC_ARI_"+pathogen+str(seed)+"_cases_age_daily.csv", np.column_stack((cropped_period.astype(str), noisy_cases)), delimiter=",", fmt="%s", header="Date,<3m,3-11m,1-4y,5-7y,8-39y,40-64y,>=65y", comments='')
    np.savetxt("Data/Processed/KPSC_ARI_"+pathogen+str(seed)+"_incidence_age_daily.csv", np.column_stack((cropped_period.astype(str), noisy_incidence)), delimiter=",", fmt="%s", header="Date,<3m,3-11m,1-4y,5-7y,8-39y,40-64y,>=65y", comments='')

    # monthly aggreagated version
    monthly_noisy_cases = np.zeros((len(pd.date_range(start=cropped_period[0], end=cropped_period[-1], freq='ME')),NAG))
    monthly_age_pops = np.zeros((len(pd.date_range(start=cropped_period[0], end=cropped_period[-1], freq='ME')),NAG))
    for i, month_start in enumerate(pd.date_range(start=cropped_period[0], end=cropped_period[-1], freq='ME')):
        month_end = month_start + pd.offsets.MonthEnd(1)
        mask = (cropped_period >= month_start) & (cropped_period <= month_end)
        monthly_noisy_cases[i,:] = np.sum(noisy_cases[mask,:], axis=0)
        monthly_age_pops[i,:] = np.mean(age_pops[np.argmax(POINTS>=date_to_t(month_start)):np.argmax(POINTS>=date_to_t(month_end))+1,:], axis=0)
    monthly_noisy_incidence = monthly_noisy_cases/monthly_age_pops
    np.savetxt("Data/Processed/KPSC_ARI_"+pathogen+str(seed)+"_cases_age_monthly.csv", np.column_stack((pd.date_range(start=cropped_period[0], end=cropped_period[-1], freq='ME').astype(str), monthly_noisy_cases)), delimiter=",", fmt="%s", header="Date,<3m,3-11m,1-4y,5-7y,8-39y,40-64y,>=65y", comments='')
    np.savetxt("Data/Processed/KPSC_ARI_"+pathogen+str(seed)+"_incidence_age_monthly.csv", np.column_stack((pd.date_range(start=cropped_period[0], end=cropped_period[-1], freq='ME').astype(str), monthly_noisy_incidence)), delimiter=",", fmt="%s", header="Date,<3m,3-11m,1-4y,5-7y,8-39y,40-64y,>=65y", comments='')

if __name__ == "__main__":
    lockdown = "FlexStepwise"
    option1 = "NA"
    option2 = "flexage"
    import_multiplier = 1.0
    np.random.seed(251024)

    pathogen = "0test"
    x = [0.5,0.1,0.1,0.001
    ,0.1,0.5,0.6,0.5
    ,0.9,0.5,0.4,0.8,0.7,0.8,0.1
    ,2e-03,1e-03,3e-03,2e-04,1e-04,3e-04,4e-03]
    # save parameters
    np.savetxt("Data/Processed/"+pathogen+"_parameters.txt", np.array(x))

    for seed in range(4):
        generate_test_trajectory(x, seed, pathogen, lockdown, option1, option2)
        monthly_noisy_cases = pd.read_csv("Data/Processed/KPSC_ARI_"+pathogen+str(seed)+"_cases_age_monthly.csv", delimiter=',', header=0, parse_dates=['Date'], index_col='Date')
        from plotting import hsv_colors
        fig, ax = plt.subplots(figsize=(6.5,4))
        for i in range(7):
            ax.plot(monthly_noisy_cases.iloc[:,i], label=AGE_GROUP_NAMES[i], color=hsv_colors[i])
        plt.savefig("Figures/"+pathogen+str(seed)+"_trajectory_cases_age_monthly.png")
        plt.close()