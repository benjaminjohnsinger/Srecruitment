### BJS April 2026
import jax
print(f"Devices found: {jax.devices()}")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import time
import pickle
import sys
import os

# import optax
import jaxopt

from utils import *

from fit_opt import get_likelihood, likelihood_threshold

if __name__ == "__main__":
    pathogen, seed, lockdown, option1, option2, import_multiplier, opt_size, opt_rate1, opt_rate2 = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9])

    NAG = 7 + ("split" in option1)
    if NAG == 7:
        from Parameters.census_population import CENSUS_AGE_POP
    elif NAG == 8:
        from Parameters.census_population import CENSUS_AGE_POP_split as CENSUS_AGE_POP

    names, bounds = parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)
    if int(str(seed)[:6]) < 260406:
        hosp = False
    else:
        hosp = True
    likelihood, N = get_likelihood(pathogen, lockdown, option1, option2, import_multiplier, constant_step=2, hosp=hosp, CENSUS_AGE_POP=CENSUS_AGE_POP, NAG=NAG)
    
    np.random.seed(seed)

    # # Load DE results (x is already in the bounded, physical space)
    # prefix, x, neg_log_likelihood = load_optimization_results("", pathogen, seed, lockdown, option1, option2)

    # # 1. Format the bounds for jaxopt: a tuple of (lower_bounds, upper_bounds)
    # lower_bounds = bounds[:, 0]
    # upper_bounds = bounds[:, 1]
    # bounds_tuple = (lower_bounds, upper_bounds)

    # # 2. Define the objective function natively (no logistic transforms needed)
    # def bounded_likelihood(params):
    #     lik = likelihood(params)
    #     # remove NAs to prevent gradient explosion
    #     return jnp.where(jnp.isnan(lik), likelihood_threshold * 10, lik)

    # print(f"Initial likelihood: {bounded_likelihood(x)}, DE reported: {neg_log_likelihood}")
    # print("Starting L-BFGS-B optimization via jaxopt...")
    # start_time = time.time()

    # # 3. Initialize and run ScipyBoundedMinimize
    # # method="l-bfgs-b" is the default for bounded Scipy minimization
    # lbfgsb = jaxopt.ScipyBoundedMinimize(
    #     fun=bounded_likelihood,
    #     method="L-BFGS-B",
    #     maxiter=opt_size,
    #     options={
    #         "maxls": 100,       # More robust line search (helps when gradients are extreme)
    #         "ftol": opt_rate1,      # Stop only if the function stops changing at a microscopic level
    #         "gtol": opt_rate2        # Force the optimizer to hunt for a true flat gradient
    #     }
    # )
    
    # # Run the optimizer
    # res = lbfgsb.run(init_params=x, bounds=bounds_tuple)
    
    # final_x = res.params
    # final_likelihood = res.state.fun_val

    # print(f"L-BFGS-B optimization completed in {time.time() - start_time:.2f} seconds.")
    # print(f"Final parameters: {final_x}")
    # print(f"Final likelihood: {final_likelihood}")
    # print(f"Likelihood improvement: {neg_log_likelihood - final_likelihood}")
    # print(f"Norm of proportional parameter changes: {jnp.linalg.norm((final_x - x) / (bounds[:,1] - bounds[:,0]))}")
    # print(f"Optax optimization completed in {time.time() - start_time:.2f} seconds.")

    # # # # save results to disk
    results_file = "Data/Processed/results"+str(seed)[:6]+"/jaxopt_polish_"+pathogen+lockdown+option1+option2+str(seed)+".pickle"
    # with open(results_file, "wb") as f:
    #     pickle.dump({
    #         "final_x": final_x
    #     }, f)
    # load results from disk
    with open(results_file, "rb") as f:
        results = pickle.load(f)
    print(f"Loaded results from disk: {results.keys()}")
    final_x = results["final_x"]
    
    # calculate Hessian natively on the physical parameters
    true_likelihood, N = get_likelihood(pathogen, lockdown, option1, option2, import_multiplier, normalize=False, hosp=hosp, constant_step=0.05, hessian=True, CENSUS_AGE_POP=CENSUS_AGE_POP, NAG=NAG)
    
    def hessian_likelihood_physical(params):
        lik = true_likelihood(params)
        return jnp.where(jnp.isnan(lik), likelihood_threshold * 10, lik)

    # Calculate gradients and Hessian at the physical optimum
    # print(f"Final gradient norm: {jnp.linalg.norm(jax.grad(hessian_likelihood_physical)(final_x))}")
    hessian = jax.hessian(hessian_likelihood_physical)(final_x)
    
    # estimate uncertainty from Hessian
    try:
        cov_matrix = jnp.linalg.inv(hessian)
        print(f"Covariance Matrix Diagonal:\n{jnp.diag(cov_matrix)}")
        param_uncertainty = jnp.sqrt(jnp.diag(cov_matrix))
        print(f"Parameter uncertainty (std): {param_uncertainty}")
    except jnp.linalg.LinAlgError:
        print("Hessian is not invertible, cannot estimate parameter uncertainty.")