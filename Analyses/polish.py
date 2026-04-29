### BJS April 2026

import jax
print(f"Devices found: {jax.devices()}")
import jax.numpy as jnp
import numpy as np
import pandas as pd
import time
import pickle
import sys
import os

import optax

from utils import *

from fit_opt import get_likelihood, likelihood_threshold

if __name__ == "__main__":
    pathogen, seed, lockdown, option1, option2, import_multiplier, opt_size, opt_rate1, opt_rate2 = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9])

    NAG = 7 + ("split" in option1)
    if NAG == 7:
        from Parameters.census_population import CENSUS_AGE_POP
    elif NAG == 8:
        from Parameters.census_population import CENSUS_AGE_POP_split as CENSUS_AGE_POP

    _, bounds = parameters_names_bounds(pathogen, lockdown, option1, option2)
    if int(str(seed)[:6]) < 260406:
        hosp = False
    else:
        hosp = True
    likelihood, N = get_likelihood(pathogen, lockdown, option1, option2, import_multiplier, hosp=hosp, CENSUS_AGE_POP=CENSUS_AGE_POP, NAG=NAG)
    def logistic_transform(x):
        return bounds[:, 0] + 1 / (1 + jnp.exp(-x)) * (bounds[:, 1] - bounds[:, 0])
    def inverse_logistic_transform(y):
        return -jnp.log((bounds[:, 1] - bounds[:, 0]) / (y - bounds[:, 0]) - 1)
    def new_likelihood(x):
        x_transformed = logistic_transform(x)
        lik =  likelihood(x_transformed)
        # remove nas
        lik = jnp.where(jnp.isnan(lik), likelihood_threshold*10, lik)
        return lik
    np.random.seed(seed)

    prefix, x, neg_log_likelihood = load_optimization_results("", pathogen, seed, lockdown, option1, option2)

    x = inverse_logistic_transform(x) # transform to unconstrained space for optimization
    print(new_likelihood(x), neg_log_likelihood)

    schedule = optax.exponential_decay(init_value=opt_rate1, transition_steps=100, decay_rate=opt_rate2, staircase=True)
    solver = optax.apply_if_finite(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=schedule)
        ),
        max_consecutive_errors=5,
    )

    def single_step(x, opt_state):
        neglogL, grad = jax.value_and_grad(new_likelihood)(x)
        update, opt_state = solver.update(grad, opt_state, x)
        x = optax.apply_updates(x, update)
        return x, opt_state, neglogL
    def scan_body(carry, step_index):
            x, opt_state = carry
            x, opt_state, neglogL = single_step(x, opt_state)
            return (x, opt_state), neglogL
    @jax.jit
    def run_optimization(x):
        opt_state = solver.init(x)

        initial_carry = (x, opt_state)
        final_carry, neglogL_history = jax.lax.scan(scan_body, initial_carry, jnp.arange(opt_size))
        final_x, _ = final_carry
        return final_x, neglogL_history
    
    print("Starting optax optimization...")
    start_time = time.time()
    final_x, neglogL_history = run_optimization(x)
    final_x.block_until_ready()
    print(f"Optax optimization completed in {time.time() - start_time:.2f} seconds.")

    final_likelihood = neglogL_history[-1]
    
    print(f"Final parameters: {logistic_transform(final_x)}")
    print(f"Final likelihood: {final_likelihood}")
    print(f"Likelihood improvement: {neg_log_likelihood - final_likelihood}")
    print(f"Norm of proportional parameter changes: {jnp.linalg.norm((final_x - x) / (bounds[:,1] - bounds[:,0]))}")
    
    # # save results to disk
    results_file = "Data/Processed/results"+str(seed)[:6]+"/polish_"+pathogen+lockdown+option1+option2+str(seed)+".pickle"
    with open(results_file, "wb") as f:
        pickle.dump({
            "final_x": final_x, 
            "neglogL_history": neglogL_history
        }, f)
    # # load results from disk
    # with open(results_file, "rb") as f:
    #     results = pickle.load(f)
    # print(f"Loaded results from disk: {results.keys()}")
    # final_x = results["final_x"]
    
    # calculate Hessian
    true_likelihood, N = get_likelihood(pathogen, lockdown, option1, option2, import_multiplier, normalize=False, hosp=hosp, hessian=True, CENSUS_AGE_POP=CENSUS_AGE_POP, NAG=NAG)
    def hessian_likelihood(x):
        x_transformed = logistic_transform(x)
        lik =  true_likelihood(x_transformed)
        # remove nas
        lik = jnp.where(jnp.isnan(lik), likelihood_threshold*10, lik)
        return lik
    print(f"Final gradient norm: {jnp.linalg.norm(jax.grad(hessian_likelihood)(final_x))}")
    hessian = jax.hessian(hessian_likelihood)(final_x)
    # print(f"Hessian matrix:\n{hessian}")
    # estimate uncertainty from Hessian
    try:
        cov_matrix = jnp.linalg.inv(hessian)
        print(jnp.diag(cov_matrix))
        param_uncertainty = jnp.sqrt(jnp.diag(cov_matrix))
        print(f"Parameter uncertainty (std): {param_uncertainty}")
    except jnp.linalg.LinAlgError:
        print("Hessian is not invertible, cannot estimate parameter uncertainty.")