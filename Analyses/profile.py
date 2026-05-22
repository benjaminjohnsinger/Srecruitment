### BJS May 2026
### Calculating profile likelihood

import argparse
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jaxopt

from fit_opt import get_likelihood
from utils import parameters_names_bounds, load_optimization_results
from Parameters.census_population import AGE_GROUP_NAMES_split, CENSUS_AGE_POP_split
def main(pathogen, seed, lockdown, option1, option2, param_name):
    NAG = 7 + ("split" in option1)
    prefix = "jaxopt_polish"

    # Fetch parameter names and bounds first to determine the index
    names, bounds = parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)
    
    # Dynamically find the index of the requested parameter
    try:
        # Convert to list in case it returns a numpy array or tuple
        param_idx = list(names).index(param_name)
    except ValueError:
        print(f"Error: Parameter '{param_name}' not found. Available parameters are: {names}", flush=True)
        sys.exit(1)

    print(f"Profiling parameter '{param_name}' (Index: {param_idx}) for {pathogen}...", flush=True)

    # Load likelihood function and optimization results
    likelihood, N = get_likelihood(
        pathogen, lockdown, option1, option2, 1e-9, 
        constant_step=0.5, normalize=False, NAG=NAG, CENSUS_AGE_POP=CENSUS_AGE_POP_split
    )

    _, x, _ = load_optimization_results(prefix, pathogen, seed, lockdown, option1, option2)
    ml = likelihood(x)
    print(f"Likelihood at DE solution: {ml}", flush=True)

    # Dynamically extract bounds, excluding the parameter we are profiling
    lower_bounds = np.delete(bounds[:, 0], param_idx)
    upper_bounds = np.delete(bounds[:, 1], param_idx)
    bounds_tuple = (lower_bounds, upper_bounds)

    # Generalize the likelihood wrapper using jnp.insert
    def fix_param_likelihood(x_rest, fixed_val):
        x_fixed = jnp.insert(x_rest, param_idx, fixed_val)
        return likelihood(x_fixed)

    perts = jnp.array([-0.05, -0.025, -0.01, -0.005, 0, 0.005, 0.01, 0.025, 0.05])
    np.random.seed(260522)

    lbfgsb = jaxopt.ScipyBoundedMinimize(
        fun=fix_param_likelihood,
        method="L-BFGS-B",
        tol=1e-7,
    )

    # Prepare the initial 'rest' array (everything except the parameter being profiled)
    x_initial_rest = jnp.delete(x, param_idx)
    param_opt_val = x[param_idx]

    start_time = time.time()
    results = []
    
    for p in perts:
        fixed_val = param_opt_val * (1 + p)
        print(f"Optimizing with {param_name} fixed at {fixed_val:.5f}...", flush=True)
        result = lbfgsb.run(x_initial_rest, fixed_val=fixed_val, bounds=bounds_tuple).params
        results.append(result)
        
    results = jnp.array(results)
    end_time = time.time()

    print(f"Optimization for {len(perts)} values completed in {end_time - start_time:.2f} seconds", flush=True)

    # Reconstruct the full parameter arrays
    final_xs_list = [jnp.insert(res, param_idx, param_opt_val * (1 + p)) for res, p in zip(results, perts)]
    final_xs = jnp.array(final_xs_list)
    final_likelihoods = jax.jit(jax.vmap(likelihood))(final_xs)
    
    print(f"Final parameters for each perturbed {param_name}:", final_xs, flush=True)
    print(f"Final likelihoods for each perturbed {param_name}:", final_likelihoods, flush=True)

    # Fit a parabola to the final_likelihoods and pert values
    coeffs = np.polyfit(perts, final_likelihoods, 2)
    a, b, c = coeffs
    
    # Original script uses index 3 for calculation baseline
    original_likelihood = final_likelihoods[3]
    delta = b**2 - 4*a*(c - original_likelihood - 1.92)
    
    if delta >= 0:
        root1 = (-b + np.sqrt(delta)) / (2*a)
        root2 = (-b - np.sqrt(delta)) / (2*a)
        print(f"Estimated {param_name} values where likelihood crosses original + 1.92: {root1:.5f}, {root2:.5f}", flush=True)
        
        # Calculate profile likelihood at these points
        liks = []
        for pert in [root1, root2]:
            fixed_val = param_opt_val * (1 + pert)
            result = lbfgsb.run(x_initial_rest, fixed_val=fixed_val, bounds=bounds_tuple).params
            x_fixed = jnp.insert(result, param_idx, fixed_val)
            lik = likelihood(x_fixed)
            liks.append(lik)
            print(f"Profile likelihood at {param_name}={fixed_val:.5f}: {lik:.2f}", flush=True)
            
        full_perts = np.concatenate((perts, [root1, root2]))
        full_likelihoods = np.concatenate((final_likelihoods, [liks[0], liks[1]]))
    else:
        print("Parabola does not cross original likelihood + 1.92, cannot estimate confidence interval.", flush=True)
        print(f"Parabola coefficients: a={a}, b={b}, c={c}", flush=True)
        full_perts = perts
        full_likelihoods = final_likelihoods

    # Sort arrays for plotting
    sorted_indices = np.argsort(full_perts)
    full_perts = full_perts[sorted_indices]
    full_likelihoods = full_likelihoods[sorted_indices]

    # Plot
    plt.plot(full_perts, full_likelihoods, marker='o')
    plt.xlabel(f"Perturbation to {param_name}")
    plt.ylabel("Final likelihood")
    plt.title(f"Sensitivity of likelihood to perturbations in {param_name}")
    plt.axhline(ml + 1.92, color='k', linestyle='--', label=f"Original {param_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Figures/sensitivity_of_likelihood_to_{param_name}_perturbations_{pathogen}.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    pathogen, seed, lockdown, option1, option2, param_name = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
    main(pathogen, seed, lockdown, option1, option2, param_name)
