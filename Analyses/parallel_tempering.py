## BJS April 2025
## MCMC with parallel tempering
import numpy as np
import pandas as pd
from scipy import stats
from multiprocessing import Process, Queue, cpu_count
from fit_MCMC import SIS_likelihood
import matplotlib.pyplot as plt
from utils import real_to_p, p_to_real, scalars_to_params, age_detection, pathogen_parameters, date_to_t
import sys
import pickle
import time
from Parameters.census_population import CENSUS_AGE_POP

NAG=7

def run_chain(temperature, q_in, q_out, n_iterations, initial_state, proposal_width, compute_energy, swap_interval=1, save_interval=1, samples_queue=None):
    """Run a single Markov chain at a specific temperature"""
    
    current_state = initial_state
    current_energy = compute_energy(current_state)
    accepted = 0
    
    for i in range(n_iterations):
        # Generate proposal
        proposal = current_state + np.random.normal(0, proposal_width, size=len(current_state))
        proposal_energy = compute_energy(proposal)
        
        # Metropolis acceptance criterion with temperature scaling
        delta_e = proposal_energy - current_energy
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / temperature):
            current_state = proposal
            current_energy = proposal_energy
            accepted += 1
        
        # Every swap_interval iterations, try to swap with adjacent temperatures
        if i % swap_interval == 0:
            # Send current state and energy to coordinator
            q_out.put((temperature, current_state, current_energy))
            
            # Get potentially new state from coordinator
            swap_result = q_in.get()
            if swap_result is not None:  # None means no swap
                current_state, current_energy = swap_result
        
        # Periodically save samples (for the chain at temperature=1.0)
        if temperature == 1.0 and i % save_interval == 0:
            samples_queue.put(current_state)
    
    # Return acceptance rate and final state
    return accepted / n_iterations

def coordinator(n_chains, temperatures, q_chains_out, q_chains_in, n_swaps):
    """Coordinate swap attempts between chains"""
    
    for _ in range(n_swaps):
        # Collect states from all chains
        states = {}
        for _ in range(n_chains):
            temp, state, energy = q_chains_out.get()
            states[temp] = (state, energy)
        
        # Try swaps between adjacent temperatures
        for i in range(n_chains - 1):
            t1, t2 = temperatures[i], temperatures[i+1]
            state1, energy1 = states[t1]
            state2, energy2 = states[t2]
            
            # Compute acceptance probability
            delta = (1/t1 - 1/t2) * (energy1 - energy2)
            if delta < 0 or np.random.random() < np.exp(-delta):
                # Accept swap
                states[t1], states[t2] = states[t2], states[t1]
        
        # Send states back to chains
        for temp in temperatures:
            q_chains_in[temp].put(states[temp])

def neg_log_likelihood(state, variables, params, data, POINTS, STATE0):
    if "OVERDISPERSION" in variables:
        overdispersion = np.exp(state[-1]-5)
        vbls = variables[:-1]
        scalars = real_to_p(state[:-1])
    else:
        overdispersion = None
        vbls = variables
        scalars = real_to_p(state)
    scalar_dict = {variables[j]:scalars[j] for j in range(len(vbls))}
    proposal_params = scalars_to_params(scalar_dict,params)
    proposal_OBS_AGE = age_detection(NAG,*scalars[-3:])
    # log_likelihood_priors = log_priors(scalars)
    log_likelihood_proposal = SIS_likelihood(data, proposal_params, POINTS, STATE0, proposal_OBS_AGE, p_time_to_obs, overdispersion=overdispersion)
    return -log_likelihood_proposal

# Parameters
pathogen, seed, lockdown, option1, option2, import_cap = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6])

np.random.seed(seed)

params, p_time_to_obs, incidence = pathogen_parameters(pathogen, lockdown)
N_S, NAG = params["N_S"], params["NAG"]

START = pd.to_datetime('2015-07-04') 
END = pd.to_datetime('2023-10-01')
PERIOD = pd.date_range(start=START, end=END, freq='D')
POINTS = np.array(date_to_t(PERIOD))
incidence.index = pd.to_datetime(incidence.index)
incidence = incidence.loc[START+pd.Timedelta(days=89):END]

## Initial conditions
STATE0 = np.zeros((2*N_S+2)*NAG)
STATE0[NAG:2*NAG] = CENSUS_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

variables = ["WANE","SEASONALITY","OFFSET","BETA","P_OBS","OBS_AGE_YOUNG","OBS_AGE_OLD","OBS_AGE_YOUNG_OLD"]
if option1 == "nb":
    variables.append("OVERDISPERSION")
if option2 != "setimport":
    variables.append("IMPORT_RATE")
else:
    params["IMPORT_RATE"] = import_cap
if (pathogen == 'RSV') or (option2 == 'nr'):
    variables.extend(["S_REL1","S_REL2","D_REL1","D_REL2"])
else:
    variables.extend(["EXTRA_IMMUNITY","FIRST_IMMUNITY","FIRST_DIS_INF_FACTOR"])
if lockdown == "FlexStepwise":
    variables.extend(["DT1","DT2","DT3","F1","F2","F3","F4"])

ordered_variables = np.array([var for var in ["WANE","SEASONALITY","OFFSET","BETA","IMPORT_RATE","EXTRA_IMMUNITY","FIRST_IMMUNITY","FIRST_DIS_INF_FACTOR","S_REL1","S_REL2","D_REL1","D_REL2","P_OBS","DT1","DT2","DT3","F1","F2","F3","F4","OVERDISPERSION","AGE_OBS_YOUNG","AGE_OBS_OLD","AGE_OBS_YOUNG_OLD"]
if var in variables])

# define compute energy by setting values in neg_log_likelihood
def compute_energy(state):
    return neg_log_likelihood(state, variables, params, incidence, POINTS, STATE0)

if __name__ == "__main__":
    np.random.seed(seed)
    # Configuration
    n_chains = cpu_count()  # Number of chains
    temperatures = np.exp(np.linspace(0, 3, n_chains))  # from 1.0 to ~20.1
    n_iterations = 50
    swap_interval = 10
    save_interval = 1
    proposal_width = 0.1
    # initial state is differential evolution result
    with open("Data/Processed/results"+str(seed)+"/DE_opt_"+pathogen+lockdown+option1+option2+str(seed)+".pickle","rb") as f:
        opt = pickle.load(f)
    initial_state = p_to_real(opt.x)

    start = time.time()

    # Create queues for communication
    q_chains_out = Queue()  # From chains to coordinator
    q_chains_in = {temp: Queue() for temp in temperatures}  # From coordinator to chains
    samples_queue = Queue()  # For collecting samples
    
    # Start chain processes
    processes = []
    for temp in temperatures:
        p = Process(
            target=run_chain,
            args=(temp, q_chains_in[temp], q_chains_out, n_iterations, initial_state, proposal_width, compute_energy, swap_interval, save_interval, samples_queue)
        )
        processes.append(p)
        p.start()
    
    # Start coordinator process
    n_swaps = n_iterations // swap_interval
    coord_process = Process(
        target=coordinator,
        args=(n_chains, temperatures, q_chains_out, q_chains_in, n_swaps)
    )
    coord_process.start()
    
    # Collect samples (from temperature=1.0 chain)
    samples = []
    expected_samples = n_iterations // save_interval
    for _ in range(expected_samples):
        samples.append(samples_queue.get())
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    coord_process.join()

    print(time.time()-start)
    
    # Convert samples to numpy array
    samples = np.array(samples)
    # calculate acceptance rates
    acceptance_rates = np.sum(samples[1:] != samples[:-1], axis=0) / n_iterations
    print(f"Acceptance rates: {acceptance_rates}")

    
    # Analysis of results
    print(f"Mean: {real_to_p(np.mean(samples, axis=0))}")
    print(f"Median: {real_to_p(np.median(samples, axis=0))}")
    print(f"Std: {np.std(samples, axis=0)}")
    print(f"IQR: {np.percentile(samples, 75, axis=0) - np.percentile(samples, 25, axis=0)}")
    # # 2d density plot
    # plt.hexbin(samples[:, 0], samples[:, 1], gridsize=50, cmap='viridis')
    # plt.show()