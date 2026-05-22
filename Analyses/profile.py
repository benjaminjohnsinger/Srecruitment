### BJS May 2026
### Calculating profile likelihood

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt
from fit_opt import get_likelihood
import jaxopt

from utils import parameters_names_bounds, load_optimization_results

pathogen = "RSV"
seed = 260505
lockdown = "ExponentialODipEqual"
option1 = "dedupsplit"
option2 = "maxagep028"
NAG = 8
N_S = 3
prefix = "jaxopt_polish"
from Parameters.census_population import AGE_GROUP_NAMES_split, CENSUS_AGE_POP_split


likelihood, N = get_likelihood(pathogen, lockdown, option1, option2, 1e-9, constant_step=0.5, normalize=False, NAG=NAG, CENSUS_AGE_POP=CENSUS_AGE_POP_split)

_, x, _ = load_optimization_results(prefix, pathogen, seed, lockdown, option1, option2)

ml = likelihood(x)
print(f"Likelihood at DE solution: {ml}")

names, bounds = parameters_names_bounds(pathogen, lockdown, option1, option2, NAG=NAG)
lower_bounds = bounds[1:, 0]
upper_bounds = bounds[1:, 1]
bounds_tuple = (lower_bounds, upper_bounds)

def fix_beta_likelihood(x, beta=0.29562485):
    x_fixed = jnp.concatenate((jnp.array([beta]), x))
    return likelihood(x_fixed)

perts = jnp.array([-0.02, -0.01, -0.005, -0.001, 0, 0.001, 0.005, 0.01, 0.02])

np.random.seed(260521)

lbfgsb = jaxopt.ScipyBoundedMinimize(
    fun=fix_beta_likelihood,
    method="L-BFGS-B",
    tol = 1e-7,
)

start_time = time.time()
results = []
for p in perts:
    print(f"Optimizing with beta fixed at {x[0] + p:.5f}...")
    result = lbfgsb.run(x[1:], beta=x[0]+p, bounds=bounds_tuple).params
    results.append(result)
results = jnp.array(results)
end_time = time.time()


print(f"Optimization for {len(perts)} values completed in {end_time - start_time:.2f} seconds")
final_xs = jnp.concatenate(((x[0]+perts)[:, None], results), axis=1)
final_likelihoods = jax.jit(jax.vmap(likelihood))(final_xs)
print("Final parameters for each perturbed beta:", final_xs)
print("Final likelihoods for each perturbed beta:", final_likelihoods)

# add zero to middle of perts and x[1:] to middle of results for plotting
lenperts = len(perts)
perts = jnp.concatenate((perts[:lenperts//2], jnp.array([0.0]), perts[lenperts//2:]))
# results = jnp.concatenate((results[:lenperts//2], jnp.array([x[1:]]), results[lenperts//2:]))
final_xs = jnp.concatenate((final_xs[:lenperts//2], jnp.array([x]), final_xs[lenperts//2:]))
final_likelihoods = jnp.concatenate((final_likelihoods[:lenperts//2], jnp.array([ml]), final_likelihoods[lenperts//2:]))
# fit a parabola to the final_likelihoods and pert values
coeffs = np.polyfit(perts, final_likelihoods, 2)
# estimate where the parabola crosses the original likelihood + 1.92 (i.e. the 95% confidence interval for a chi-squared distribution with 1 degree of freedom)
a, b, c = coeffs
original_likelihood = final_likelihoods[3]
delta = b**2 - 4*a*(c - original_likelihood - 1.92)
if delta >= 0:
    root1 = (-b + np.sqrt(delta)) / (2*a)
    root2 = (-b - np.sqrt(delta)) / (2*a)
    print(f"Estimated beta values where likelihood crosses original + 1.92: {root1:.5f}, {root2:.5f}")
else:
    print("Parabola does not cross original likelihood + 1.92, cannot estimate confidence interval for beta")
    print(f"Parabola coefficients: a={a}, b={b}, c={c}")
    plt.plot(perts, final_likelihoods, marker='o')
    plt.show()
# calculate profile likelihood at these points
liks = []
for pert in [root1, root2]:
    beta_value = x[0] + pert
    # optimize the other parameters with beta fixed at this value
    result = lbfgsb.run(x[1:], beta=beta_value, bounds=bounds_tuple).params
    x_fixed = jnp.concatenate((jnp.array([beta_value]), result))
    lik = likelihood(x_fixed)
    liks.append(lik)
    print(f"Profile likelihood at beta={beta_value:.5f}: {lik:.2f}")
# add these to the plot
full_perts = np.concatenate((perts, [root1, root2]))
full_likelihoods = np.concatenate((final_likelihoods, [liks[0], liks[1]]))
# sort by pert values
sorted_indices = np.argsort(full_perts)
full_perts = full_perts[sorted_indices]
full_likelihoods = full_likelihoods[sorted_indices]
plt.plot(full_perts, full_likelihoods, marker='o')
# plt.plot(perts, final_likelihoods, marker='o')
plt.xlabel("Perturbation to beta")
plt.ylabel("Final likelihood")
plt.title("Sensitivity of likelihood to perturbations in beta")
plt.axhline(ml+1.92, color='k', linestyle='--', label="Original beta")
plt.legend()
plt.tight_layout()
plt.savefig(f"Figures/sensitivity_of_likelihood_to_beta_perturbations_{pathogen}.png", dpi=300, bbox_inches='tight')
