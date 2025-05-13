## BJS April 2025
## MCMC with parallel tempering and adaptive sampling
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

def run_chain(temperature, q_in, q_out, n_iterations, initial_state, proposal_widths, compute_energy, swap_interval=1, save_interval=1, samples_queue=None):
    """Run a single Markov chain at a specific temperature with adaptive proposal widths"""
    
    current_state = initial_state
    current_energy = compute_energy(current_state)
    n_params = len(current_state)
    acceptance = np.zeros((n_iterations, n_params))
    
    for i in range(n_iterations):
        # Use adaptive proposal widths after initial burn-in
        if i > 20:
            # Calculate acceptance rates over last 20 iterations for each parameter
            acceptance_rates = np.mean(acceptance[i-20:i], axis=0)
            # Update proposal widths (targeting 23% acceptance rate)
            update = (acceptance_rates - 0.23) / np.log(i+1)
            proposal_widths = proposal_widths * np.exp(-update)
            
            # Optional: Print progress every 100 iterations
            if i % 100 == 0 and temperature == 1.0:
                print(f"Chain {temperature:.2f}, Iter {i}, Accept rates: {[f'{x:.2f}' for x in acceptance_rates]}, "
                      f"Widths: {[f'{x:.2g}' for x in proposal_widths]}")
        
        # Random order for parameters
        param_order = np.random.permutation(n_params)
        
        # Update each parameter individually (Gibbs-style)
        for param_idx in param_order:
            # Create proposal by changing only one parameter
            proposal = current_state.copy()
            proposal[param_idx] += np.random.normal(0, proposal_widths[param_idx])
            
            # Compute energy of proposal
            proposal_energy = compute_energy(proposal)
            
            # Metropolis acceptance criterion with temperature scaling
            delta_e = proposal_energy - current_energy
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / temperature):
                current_state = proposal
                current_energy = proposal_energy
                acceptance[i, param_idx] = 1
        
        # Every swap_interval iterations, try to swap with adjacent temperatures
        if i % swap_interval == 0:
            # Send current state and energy to coordinator
            q_out.put((temperature, current_state, current_energy))
            
            # Get potentially new state from coordinator
            swap_result = q_in.get()
            if swap_result is not None:  # None means no swap
                current_state, current_energy = swap_result
        
        # Periodically save samples (only for the chain at temperature=1.0)
        if temperature == 1.0 and i % save_interval == 0:
            samples_queue.put((current_state, proposal_widths))
    
    # Return mean acceptance rate for final diagnostics
    return np.mean(acceptance, axis=0)

def coordinator(n_chains, temperatures, q_chains_out, q_chains_in, n_swaps):
    """Coordinate swap attempts between chains"""
    
    # Track swap statistics
    swap_attempts = np.zeros((n_chains-1))
    swap_accepts = np.zeros((n_chains-1))
    
    for swap_idx in range(n_swaps):
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
            swap_attempts[i] += 1
            
            if delta < 0 or np.random.random() < np.exp(-delta):
                # Accept swap
                states[t1], states[t2] = states[t2], states[t1]
                swap_accepts[i] += 1
        
        # Send states back to chains
        for temp in temperatures:
            q_chains_in[temp].put(states[temp])
        
        # Periodically print swap statistics
        if swap_idx % 10 == 0 and swap_idx > 0:
            swap_rates = swap_accepts / swap_attempts
            print(f"Swap {swap_idx}/{n_swaps}: Rates between chains: {[f'{r:.2f}' for r in swap_rates]}")
    
    # Return final swap acceptance rates
    return swap_accepts / swap_attempts

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
    n_iterations = int(1e5)  # Increased for better adaptation
    swap_interval = 100
    save_interval = 5
    
    # initial state is differential evolution result
    with open(f"Data/Processed/results{seed}/DE_opt_{pathogen}{lockdown}{option1}{option2}{seed}.pickle", "rb") as f:
        opt = pickle.load(f)
    initial_state = p_to_real(opt.x)
    
    # Initialize adaptive proposal widths (one per parameter)
    proposal_widths = np.ones(len(initial_state)) * 0.1
    
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
            args=(temp, q_chains_in[temp], q_chains_out, n_iterations, initial_state, proposal_widths.copy(), compute_energy, swap_interval, save_interval, samples_queue)
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
    proposal_history = []
    expected_samples = n_iterations // save_interval
    print(f"Sampling started, expecting {expected_samples} samples")
    
    for i in range(expected_samples):
        sample, width = samples_queue.get()
        samples.append(sample)
        proposal_history.append(width)
        
        # Print periodic updates
        if (i+1) % 100 == 0:
            current_time = time.time() - start
            completion = (i+1) / expected_samples
            eta = (current_time / completion) - current_time if completion > 0 else float('inf')
            print(f"Collected {i+1}/{expected_samples} samples ({completion:.1%}), " 
                  f"elapsed: {current_time:.1f}s, ETA: {eta:.1f}s")
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    coord_process.join()

    runtime = time.time() - start
    print(f"Completed in {runtime:.2f} seconds")
    
    # Convert samples to numpy array
    samples = np.array(samples)
    proposal_history = np.array(proposal_history)
    
    # Calculate parameter-wise transition rates
    transitions = np.sum(samples[1:] != samples[:-1], axis=0)
    acceptance_rates = transitions / (n_iterations * (len(samples) - 1) / expected_samples)
    
    # Save results
    results = {
        'samples': samples,
        'proposal_history': proposal_history,
        'acceptance_rates': acceptance_rates,
        'runtime': runtime,
        'temperatures': temperatures,
        'variables': variables
    }
    
    output_path = f"Data/Processed/results{seed}/PT_adaptive_{pathogen}{lockdown}{option1}{option2}{seed}.pickle"
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    
    # Analysis of results
    print(f"Acceptance rates by parameter: {[f'{r:.3f}' for r in acceptance_rates]}")
    print(f"Mean: {real_to_p(np.mean(samples, axis=0))}")
    print(f"Median: {real_to_p(np.median(samples, axis=0))}")
    print(f"Std: {np.std(samples, axis=0)}")
    print(f"IQR: {np.percentile(samples, 75, axis=0) - np.percentile(samples, 25, axis=0)}")
    
    # Final proposal widths
    print(f"Final proposal widths: {[f'{w:.2g}' for w in proposal_history[-1]]}")
    
    # Plot proposal width evolution for each parameter
    plt.figure(figsize=(15, 10))
    for i in range(len(proposal_history[0])):
        plt.plot(proposal_history[:, i], label=f'Param {i} ({variables[i] if i < len(variables) else "unknown"})')
    plt.xlabel('Sample')
    plt.ylabel('Proposal Width')
    plt.title('Adaptation of Proposal Widths')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f"Data/Processed/results{seed}/PT_adaptive_widths_{pathogen}{lockdown}{option1}{option2}{seed}.png")