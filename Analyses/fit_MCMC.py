import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy as sp
import re
import jax.scipy as jsp
import matplotlib.pyplot as plt
from matplotlib import cm as colormaps
import time
from functools import partial
hsv_colors = colormaps.hsv(-0.02+np.arange(7)/7)
hsv_colors[3] = colormaps.hsv((3/7)+0.04)
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController, DirectAdjoint, RecursiveCheckpointAdjoint

from Parameters.census_population import CENSUS_AGE_POP, AGING_RATE
from JAX_ODEs import deltas
N_C = 2
NAG = 7
N_S = 3
from utils import date_to_t, parameters_from_DE, x_to_params, calculate_population_size
from Gemini_vaccination import FluRatePreprocessor
import time

def run_simulation(params, y0, t1, saveat_ts, hessian=False):
    term = ODETerm(deltas)
    solver = Dopri5()
    saveat = SaveAt(ts=saveat_ts)
    step_controller = PIDController(rtol=1e-5, atol=1e-5)
    if hessian:
        adjoint = DirectAdjoint()
    else:
        adjoint = RecursiveCheckpointAdjoint()
    solution = diffeqsolve(
                        term, solver,
                        t0=0, t1=t1, dt0=0.1, stepsize_controller=step_controller,
                        saveat=saveat, y0=y0.flatten(), args=params, 
                        max_steps=10000, throw=False,
                        adjoint=adjoint,
                        )
    return solution

## POINTS must start  (at least) len(p_time_to_obs) days before the first observation to avoid issues from jnp.roll behaviour
def SIS_likelihood(data, daily_hospitalization_rates, params, POINTS, STATE0, p_time_to_obs, incidence_data=False, obs_age=None, mask=[3135,3288], solution=None, hessian=False, return_sum=True):
    # run simulation
    if solution is None:
        t1 = int(POINTS[-1])
        values = run_simulation(params, STATE0, t1, POINTS, hessian=hessian)
        values = values.ys.T
    else:
        values = solution.ys.T

    population_size = calculate_population_size(values, N_S=N_S, NAG=NAG)

    if incidence_data:
        expected_obs = calculate_expected_obs(values, p_time_to_obs, len(data))
        incidence = jnp.round(data*population_size[-len(data):])
        # exclude date range from likelihood calculation
        masked_incidence = jnp.ones((len(data) - (mask[1] - mask[0]),NAG))
        masked_expected_obs = jnp.ones((len(data) - (mask[1] - mask[0]),NAG))
        masked_incidence = masked_incidence.at[:mask[0]].set(incidence[:mask[0]]).at[mask[0]:].set(incidence[mask[1]:])
        masked_expected_obs = masked_expected_obs.at[:mask[0]].set(expected_obs[:mask[0]]).at[mask[0]:].set(expected_obs[mask[1]:])
        # calculate Poisson likelihood of observed incidence given expected incidence
        likelihood = jsp.stats.poisson.logpmf(masked_incidence, masked_expected_obs)
    else:
        if obs_age is not None:
            # trajectory is total proportion infected over time
            infectious = jnp.sum(values[1:].reshape((2*N_S+1, NAG, -1))[1:2*N_S:2], axis=0).T
            expected_infectious = jax.nn.softplus(infectious[-len(data):]*100)/100
            expected_ratio = jnp.divide(expected_infectious, population_size[-len(data):] * obs_age)
            expected_positivity = jnp.clip(expected_ratio, 1e-10, 0.99)
        else:
            expected_obs = calculate_expected_obs(values, p_time_to_obs, len(data))
            # probability of getting a positive test in hospital is expected_obs / population size over time
            expected_ratio = jnp.divide(expected_obs, population_size[-len(data):])
            # then condition by baseline probabilty of hospitalization
            expected_positivity = jnp.clip(jnp.divide(expected_ratio, jnp.maximum(daily_hospitalization_rates[-len(data):], 1e-10)),
                                           1e-10,0.99)
        # exclude date range from likelihood calculation
        masked_tests = jnp.ones((len(data)- (mask[1] - mask[0]),NAG,2))
        masked_expected_positivity = jnp.ones((len(data)- (mask[1] - mask[0]),NAG))
        masked_tests = masked_tests.at[:mask[0]].set(data[:mask[0]]).at[mask[0]:].set(data[mask[1]:])
        masked_expected_positivity = masked_expected_positivity.at[:mask[0]].set(expected_positivity[:mask[0]]).at[mask[0]:].set(expected_positivity[mask[1]:])
        # calculate binomial likelihood of observed positives given expected proportion positive and total tests
        likelihood = jsp.stats.binom.logpmf(masked_tests[...,1], masked_tests[...,0], masked_expected_positivity)
    if return_sum:
        return jnp.sum(likelihood)
    else:
        return likelihood

def calculate_expected_obs(values, p_time_to_obs, length):
    trajectory = jnp.diff(values[-NAG:,:],axis=1).T
        # convolution of trajectory with probability of detection at each day after infection to get expected observations on each day
    p_time_to_obs_flipped = jnp.flip(p_time_to_obs.flatten())
    def obs_convolution(x):
        return jnp.convolve(x, p_time_to_obs_flipped, mode='same')
        # the expected observations for a given date are the observations on each day i days prvious multiplied by the probability of detection i days after infection
    expected_obs = jax.vmap(obs_convolution, in_axes=1, out_axes=1)(trajectory)
    expected_obs = jax.nn.softplus(expected_obs[-length:]*100)/100
    return expected_obs

def fit_transform(target_means, target_cov, bounds):
    n_params = target_means.shape[0]

    # low, high, and scale of bounds
    low = bounds[:,0]
    high = bounds[:,1]
    scale = high - low

    # indices for lower-triangular Cholesky factor
    tril_indices = jnp.tril_indices(n_params)

    def objective_func(unconstrained_params_flat):
        # unpack into loc and Cholesky L
        loc = unconstrained_params_flat[:n_params]
        L_elements = unconstrained_params_flat[n_params:]
        L = jnp.zeros((n_params,n_params)).at[tril_indices].set(L_elements)
        unconstrained_cov = L @ L.T

        # define transform with vettorized loc and scale
        transform = dist.transforms.ComposeTransform([
                    dist.transforms.SigmoidTransform(),
                    dist.transforms.AffineTransform(loc=low, scale=scale)
                    ])
        base_dist = dist.MultivariateNormal(loc=loc, covariance_matrix=unconstrained_cov)
        final_dist = dist.TransformedDistribution(base_dist, transform)

        # calculate mean and cov of final distribution
        samples = final_dist.sample(jax.random.PRNGKey(0), sample_shape=(250_000,))
        actual_mean = jnp.mean(samples, axis=0)
        actual_cov = jnp.cov(samples, rowvar=False)

        # calculate and return flat error vector
        mean_error = actual_mean - target_means
        cov_error = actual_cov[tril_indices] - target_cov[tril_indices]
        error_vec = jnp.concatenate([mean_error, cov_error])
        if jnp.any(jnp.isnan(error_vec)) or jnp.any(jnp.isinf(error_vec)):
            return jnp.ones_like(error_vec)*1e6
        return error_vec

    # initial guess for loc and L
    scaled_means = (target_means - low) / scale
    initial_loc_guess = jnp.log(scaled_means / (1 - scaled_means))
    initial_L_guess = jnp.linalg.cholesky(jnp.diag(jnp.diag(target_cov)))
    initial_L_elements = initial_L_guess[tril_indices]
    initial_guess = jnp.concatenate([initial_loc_guess, initial_L_elements])

    # optimize
    print(f"Solving transformed priors for {n_params} parameters...")
    # solution = sp.optimize.root(objective_func, x0=initial_guess, method='hybr')
    solution = jsp.optimize.least_squares(objective_func, x0=initial_guess, method='trf', xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=200)

    if not solution.success:
        raise RuntimeError("Optimization failed: " + solution.message)
    
    found_params_flat = solution.x
    unconstrained_loc = found_params_flat[:n_params]
    L_elements = found_params_flat[n_params:]
    L = jnp.zeros((n_params,n_params)).at[tril_indices].set(L_elements)
    unconstrained_cov = L @ L.T
    return unconstrained_loc, unconstrained_cov
    

def prior_distribution(filename, bounds, n=1000, dist_type="multilog", varlim=None, pathogen="RSV", option1="NA"):
    DE_output = pd.read_csv(filename).values
    # remove columns 1,2,3 (pathogen, lockdown, seed)
    DE_output = np.delete(DE_output, [1,2,3], axis=1)
    # enforce numerical type
    DE_output = DE_output.astype(float)
    de = jnp.asarray(DE_output)
    # sort by first column (best likelihood)
    de = de[jnp.argsort(de[:,0]),:]
    if varlim is not None:
        n_ds = 6 + ("Influenza" in pathogen) + ("wane" in option1) + (("mimm" in option1) & ("maxmimm" not in option1)) + 2 * ((pathogen != "RSV") & ("Influenza" not in pathogen))
        if varlim == "dynamicpathogen":
            de = de[:,:n_ds+1]
            bounds = bounds[:n_ds]
        elif varlim == "dynamic":
            de = de[:,:n_ds+8]
            bounds = bounds[:n_ds+7]
        elif varlim == "pathogen":
            de_cols1 = de[:,:n_ds+1]
            de_cols2 = de[:,n_ds+8:]
            de = jnp.concatenate((de_cols1, de_cols2), axis=1)
            bounds = jnp.concatenate((bounds[:n_ds], bounds[n_ds+7:]))

    if dist_type == "multilog":
        # fit lognormal priors
        de_mbound = de[:,1:] - bounds[:,0]
        # exclude zeros
        de_mbound = de_mbound[jnp.all(de_mbound > 0, axis=1)]
        # take first n samples
        de_mbound = de_mbound[:n,:]
        fit_means = jnp.mean(jnp.log(de_mbound), axis=0)
        fit_cov = jnp.cov(jnp.log(de_mbound), rowvar=False)

        # fit multivariate normal in the log space
        prior_dist = dist.MultivariateNormal(loc=fit_means, covariance_matrix=fit_cov)
        # constrain to less than the log of the upper bound - lower bound
        prior_dist.support = dist.constraints.less_than(jnp.log(bounds[:,1] - bounds[:,0]))
    elif dist_type == "sigmoid":
        mean = jnp.mean(de[:n,1:], axis=0)
        cov = jnp.cov(de[:n,1:], rowvar=False)

        unconstrained_loc, unconstrained_cov = fit_transform(mean, cov, bounds)

        base_dist = dist.MultivariateNormal(loc=unconstrained_loc, covariance_matrix=unconstrained_cov)
        transform = dist.transforms.ComposeTransform([
                    dist.transforms.SigmoidTransform(),
                    dist.transforms.AffineTransform(loc=bounds[:,0], scale=bounds[:,1]-bounds[:,0])
                    ])
        prior_dist = dist.TransformedDistribution(base_dist, transform)

    return prior_dist, fit_means

def fit_MCMC(pathogen, lockdown, option1, option2, seed, import_multiplier = 1e-9, samples=1000, varlim=None, ts_length=3500, mask=[3135,3288]):
    params, _, bounds, tests, p_time_to_obs = parameters_from_DE(pathogen, lockdown, option1, option2, seed)
    daily_hospitalization_rates = jnp.asarray(pd.read_csv('Data/Processed/KPSC_ARI_hospitalization_rates_by_day_age_group.csv',index_col=0).fillna(0).values)
    p_time_to_obs_flipped = jnp.flip(p_time_to_obs.flatten())
    def obs_convolution(x):
        return jnp.convolve(x, p_time_to_obs_flipped, mode='same')
    @jax.jit
    def trajectory(state0, params, term=ODETerm(deltas), solver=Dopri5(), step_controller=PIDController(rtol=1e-5, atol=1e-5), startdate='2015-10-01', enddate='2025-05-01'):
        saveat = SaveAt(ts=jnp.arange(date_to_t(startdate)-90, date_to_t(enddate)))
        solution = diffeqsolve(
                        term, solver,
                        t0=0, t1=date_to_t(enddate)-1, dt0=None, stepsize_controller=step_controller,
                        saveat=saveat, y0=state0.flatten(), args=params, 
                        max_steps=100000,  
                        )
        return solution
    STATE0_shaped = jnp.zeros((2*N_S+1,NAG))
    STATE0_shaped = STATE0_shaped.at[0,:].set(CENSUS_AGE_POP-1)
    STATE0_shaped = STATE0_shaped.at[1,:].set(1)
    STATE0 = jnp.concatenate((jnp.array([0]), STATE0_shaped.flatten()))
    if re.match(r'\d{6}', lockdown):
        prior_lockdown = "FlexStepwise"
    else:
        prior_lockdown = lockdown
    prior_dist, prior_means = prior_distribution(f"Outputs/DE_outputs/{pathogen}{seed}{prior_lockdown}{option1}{option2}1e-92010.7.csv", bounds,
        n=1000, dist_type="multilog", varlim = varlim, pathogen=pathogen, option1=option1)
    if varlim is not None:
        option1 = option1 + varlim
        n_ds = 6 + ("Influenza" in pathogen) + ("wane" in option1) + (("mimm" in option1) & ("maxmimm" not in option1)) + 2 * ((pathogen != "RSV") & ("Influenza" not in pathogen))
        if varlim == "dynamicpathogen":
            bounds = bounds[:n_ds]
        elif varlim == "dynamic":
            bounds = bounds[:n_ds+7]
        elif varlim == "pathogen":
            bounds = jnp.concatenate((bounds[:n_ds], bounds[n_ds+7:]))
    def model(obs_tests=None):
        masked_obs_tests = jnp.ones((ts_length - (mask[1] - mask[0]),NAG,2))
        masked_expected_positivity = jnp.ones((ts_length - (mask[1] - mask[0]),NAG))

        # sample parameters from prior
        sample = numpyro.sample("params", prior_dist)
        x_sampled = jnp.exp(sample) + bounds[:,0]
        sim_params = x_to_params(x_sampled, pathogen, lockdown, option1, option2, fixed_params=params, import_multiplier=import_multiplier)
        solution = trajectory(STATE0, sim_params)
        values = solution.ys.T
        
        # Compute expected positivity using binomial likelihood from SIS_likelihood
        trajectory_diff = jnp.diff(values[-NAG:,:],axis=1).T
        expected_obs = jax.vmap(obs_convolution, in_axes=1, out_axes=1)(trajectory_diff)
        expected_obs = jax.nn.softplus(expected_obs[-ts_length:]*100)/100
        
        # Compute population size and expected positivity
        population_size = calculate_population_size(values, N_S=N_S, NAG=NAG)
        expected_ratio = jnp.divide(expected_obs, population_size[-ts_length:])
        expected_positivity = jnp.minimum(0.99, jnp.divide(expected_ratio, jnp.maximum(daily_hospitalization_rates[-ts_length:], 1e-10)))
        
        # Remove masked values by explicitly indexing
        masked_obs_tests = masked_obs_tests.at[:mask[0]].set(obs_tests[:mask[0]]).at[mask[0]:].set(obs_tests[mask[1]:])
        masked_expected_positivity = masked_expected_positivity.at[:mask[0]].set(expected_positivity[:mask[0]]).at[mask[0]:].set(expected_positivity[mask[1]:])
        
        numpyro.sample("obs_tests", dist.Binomial(masked_obs_tests[...,0], masked_expected_positivity), obs=masked_obs_tests[...,1])
    nuts_kernel = NUTS(model,
                        init_strategy=numpyro.infer.init_to_value(values={"params": prior_means}),
                        max_tree_depth=6,
                        dense_mass=True)
    mcmc = MCMC(nuts_kernel, num_warmup=jnp.minimum(samples,1000), num_samples=samples)
    mcmc.run(jax.random.PRNGKey(seed), obs_tests=tests)
    return mcmc
    

if __name__ == "__main__":
    import seaborn as sns

    def plot_histogram(samples, param_name, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        sns.histplot(samples, kde=True, stat='density', bins=30, ax=ax)
        ax.set_title(f'Posterior Distribution of {param_name}')
        ax.set_xlabel(param_name)
        ax.set_ylabel('Density')
        ax.axvline(x=jnp.mean(samples), color='red', linestyle='--', label='Mean')
        # ax.axvline(x=true_value, color='green', linestyle='--', label='Original')
        ax.legend()
        return ax
    
    def plot_likelihoods(posterior_samples_params, params, incidence, p_time_to_obs, downsample=False):
        if downsample:
            posterior_samples_params = posterior_samples_params[np.random.choice(posterior_samples_params.shape[0], size=downsample, replace=False)]
        n = posterior_samples_params.shape[1]//2 + posterior_samples_params.shape[1]%2
        fig, ax = plt.subplots(n, 2, figsize=(12, 3*n))
        age_pops = jnp.asarray(pd.read_csv("Data/Processed/age_pops_daily.csv").values)
        STATE0_shaped = jnp.zeros((2*N_S+1,NAG))
        STATE0_shaped = STATE0_shaped.at[0,:].set(CENSUS_AGE_POP-1)
        STATE0_shaped = STATE0_shaped.at[1,:].set(1)
        STATE0 = jnp.concatenate((jnp.array([0]), STATE0_shaped.flatten()))
        sim_params = jax.vmap(lambda x: x_to_params(x, pathogen, "FlexStepwise", "pathogen", "flexage", vax_preprocessor=FluRatePreprocessor(jnp.arange(0, date_to_t('2025-05-01')), age_pops, AGING_RATE), fixed_params=params, import_multiplier=1e-9))(posterior_samples_params)
        likelihoods = jax.vmap(lambda p: SIS_likelihood(incidence, p, jnp.arange(date_to_t('2015-07-03'), date_to_t('2025-05-01')), STATE0, p_time_to_obs, age=True, incidence=True))(sim_params)
        likelihoods = jnp.array(likelihoods)
        for i in range(n):
            for j in range(2):
                idx = i*2 + j
                if idx < posterior_samples_params.shape[1]:
                    plt.plot(posterior_samples_params[:,idx], likelihoods, 'o', alpha=0.1)
                    ax[i,j].set_xlabel(param_names[idx])
                    ax[i,j].set_ylabel('Log Likelihood')
        plt.tight_layout()
        plt.savefig("Figures/NumPyro_test_likelihoods_"+pathogen+"_sp1000.png", dpi=300)

    def plot_trajectories(posterior_samples_params, params, restricted_bounds, incidence, p_time_to_obs, option1="pathogen", downsample=False, ax=None, age=False, monthly=False, noisy=False):
        if downsample:
            posterior_samples_params = posterior_samples_params[np.random.choice(posterior_samples_params.shape[0], size=downsample, replace=False)]
        age_pops = jnp.asarray(pd.read_csv("Data/Processed/age_pops_daily.csv").values)
        cases = jnp.round(incidence*age_pops[-len(incidence):])
        p_time_to_obs_flipped = jnp.flip(p_time_to_obs.flatten())
        def obs_convolution(x):
            return jnp.convolve(x, p_time_to_obs_flipped, mode='same')
        @jax.jit
        def trajectory(state0, params, term=ODETerm(deltas), solver=Dopri5(), step_controller=PIDController(rtol=1e-5, atol=1e-5), startdate='2015-10-01', enddate='2025-05-01'):
            saveat = SaveAt(ts=jnp.arange(date_to_t(startdate)-90, date_to_t(enddate)))
            solution = diffeqsolve(
                            term, solver,
                            t0=0, t1=date_to_t(enddate)-1, dt0=None, stepsize_controller=step_controller,
                            saveat=saveat, y0=state0.flatten(), args=params, 
                            max_steps=100000,  
                            )
            # trajectory is np.diff over time of last NAG elements of solution
            cumulative_observations = solution.ys[:,-NAG:]
            observations = jnp.diff(cumulative_observations, axis=0)
            expected_obs = jax.vmap(obs_convolution, in_axes=1, out_axes=1)(observations)
            softplus_obs = jax.nn.softplus(expected_obs*100)/100
            return softplus_obs
        STATE0_shaped = jnp.zeros((2*N_S+1,NAG))
        STATE0_shaped = STATE0_shaped.at[0,:].set(CENSUS_AGE_POP-1)
        STATE0_shaped = STATE0_shaped.at[1,:].set(1)
        STATE0 = jnp.concatenate((jnp.array([0]), STATE0_shaped.flatten()))
        transformed_samples = jnp.exp(posterior_samples_params) + restricted_bounds[:,0]
        sim_params = jax.vmap(lambda x: x_to_params(x, pathogen, "FlexStepwise", option1, "flexage", vax_preprocessor=FluRatePreprocessor(jnp.arange(0, date_to_t('2025-05-01')), age_pops, AGING_RATE), fixed_params=params, import_multiplier=1e-9))(transformed_samples)
        trajectories = jax.vmap(lambda p: trajectory(STATE0, p)) (sim_params)
        if noisy:
            trajectories = sp.stats.poisson.rvs(trajectories)
            lst = ":"
        else:
            lst = "-"
        if monthly:
            transformed_cases = np.zeros((cases.shape[0]//30 + 1, cases.shape[1]))
            for i in range(cases.shape[0]//30):
                transformed_cases[i] = np.sum(cases[i*30:(i+1)*30], axis=0)
            case_times = np.arange(15, cases.shape[0]+15, 30)
        if age:
            for i in range(7):
                if not monthly:
                    if noisy:
                        transformed_cases_age = cases[:,i]
                    else:
                        transformed_cases_age = np.convolve(cases[:,i], np.ones(7)/7, mode='same')
                    case_times = np.arange(cases.shape[0])
                else:
                    transformed_cases_age = transformed_cases[:,i]
                age_trajectories = trajectories[:,:,i]
                ax[i].plot((30**monthly)*age_trajectories.T[-len(cases):], color='#DC267F', alpha=0.1*(0.3**noisy))
                ax[i].plot(case_times,transformed_cases_age, color='black', label='Observed Cases', linestyle=lst)
        else:
            if monthly:
                transformed_cases = transformed_cases.sum(axis=1)
            else:
                transformed_cases = np.convolve(cases.sum(axis=1), np.ones(7)/7, mode='same')
                case_times = np.arange(cases.shape[0])
            ax.plot(case_times, transformed_cases, color='black', label='Observed Cases (7-day MA)')
            ax.plot((30**monthly)*trajectories.sum(axis=2).T[-len(cases):], color='red', alpha=0.1)

    import pickle
    from matplotlib.patches import Patch
    print(jax.local_device_count())
    start = time.time()
    lockdown = "FlexStepwise"
    option1 = "NA"
    seeds = [260217, 260217, 260217, 260217, 2602173, 2511042, ]
    pathogens = ["RSV", "InfluenzaA", "Adenovirus", "Metapneumovirus", "Parainfluenza3", "InfluenzaB" ]
    for pathogen, seed in zip(pathogens, seeds):
        print(pathogen, time.time()-start)
        mcmc = fit_MCMC(pathogen, lockdown, option1, "flexage", seed, import_multiplier=1e-9, samples=10, varlim="pathogen")
        mcmc.print_summary()
        # save samples
        posterior_samples = mcmc.get_samples()
        with open("Data/Processed/MCMC_outputs/MCMC_"+pathogen+lockdown+option1+"flexage"+str(seed)+"_pathogen_samples_sp100_mass.pickle", "wb") as f:
            pickle.dump(posterior_samples, f)
        # with open("Data/Processed/MCMC_outputs/MCMC_"+pathogen+"FlexStepwise"+option1+"flexage"+str(seed)+"_pathogen_samples_sp100_mass.pickle", "rb") as f:
        #     posterior_samples = pickle.load(f)
        param_samples = posterior_samples['params']
        params, param_names, bounds, tests, p_time_to_obs = parameters_from_DE(pathogen, "FlexStepwise", option1, "flexage", seed)
        # plot_likelihoods(param_samples, params, incidence, p_time_to_obs, downsample=100)
        # prior_dist, prior_means = prior_distribution(f"Data/Processed/DE_outputs/DE_{pathogen}FlexStepwise0.005flexage250709_sorted.csv",
        #     bounds, n=1000, dist_type="multilog", varlim = "pathogen", pathogen=pathogen, option1=option1)
        # param_samples = prior_dist.sample(jax.random.PRNGKey(0), sample_shape=(1000,))
        # plot histograms of each parameter
        n = param_samples.shape[1]//2 + param_samples.shape[1]%2
        n_ds = 6 + ("Influenza" in pathogen) + ("wane" in option1) + (("mimm" in option1) & ("maxmimm" not in option1)) + 2 * ((pathogen != "RSV") & ("Influenza" not in pathogen))
        bounds = jnp.concatenate((bounds[:n_ds], bounds[n_ds+7:]))
        fig, ax = plt.subplots(n, 2, figsize=(12, 3*n))
        transformed_samples = jnp.exp(param_samples) + bounds[:,0]
        param_names = param_names[:n_ds] + param_names[n_ds+7:]
        for i in range(n):
            for j in range(2):
                idx = i*2 + j
                if idx < len(param_names):
                    plot_histogram(transformed_samples[:,idx], param_names[idx], ax=ax[i,j])
        plt.tight_layout()
        plt.savefig("Figures/NumPyro_test_pathogen_variables_"+pathogen+option1+str(seed)+"_sp100_mass.png", dpi=300)
        plt.close()
        fig, axes = plt.subplots(3,3,figsize=(13.3,7.5))
        axes = axes.flatten()
        plot_trajectories(param_samples, params, bounds, incidence, p_time_to_obs, option1=option1+"pathogen", downsample=100, ax=axes, age=True, monthly=True, noisy=False)
        ## title axes
        age_names = ["<3m", "3–11m", "1–4y", "5–7y", "8–39y", "40–64y", "<=65y"]
        for i in range(7):
            axes[i].set_title(f'{age_names[i]}')

        # plot_trajectories(param_samples, params, bounds, incidence, p_time_to_obs, downsample=False, ax=ax)
        # plt.xlabel('Days since 1970-01-01')
        # plt.ylabel('Number of Cases')
        # plt.title(f'Posterior Predictive Trajectories for {pathogen}')
        # plt.legend()
        plt.suptitle(f'Posterior Predictive Trajectories for {pathogen} (Monthly Cases)', fontsize=16)
        plt.tight_layout()
        plt.savefig("Figures/NumPyro_test_trajectories_"+pathogen+option1+str(seed)+lockdown+"sp100_mass_monthly.png", dpi=300)

    # ## 2d contour plot comparisons
    # fig, ax = plt.subplots(figsize=(6.5,6.5))
    # # # ax.set_xlim(0.05, 0.55)
    # # ax.set_xlim(0, 0.01)
    # # # ax.set_ylim(0, 0.011)
    # # ax.set_ylim(0.1, 1)
    # pathogen_colors = ["#648FFF","#DC267F","#FFB000","#785EF0"]
    # for pathogen_i in range(4):
    #     pathogen = ["RSV", "Metapneumovirus", "Parainfluenza3", "Adenovirus"][pathogen_i]
    #     REC = [1/4.9,1/3,1/3,1/3][pathogen_i]
    #     # 2d histograms of beta vs S_REL2
    #     with open("Data/Processed/MCMC_outputs/MCMC_"+pathogen+"FlexStepwiseNAflexage"+str(seed)+"_pathogen_samples_sp100_mass.pickle", "rb") as f:
    #         posterior_samples = pickle.load(f)
    #     params, param_names, bounds, incidence, p_time_to_obs = parameters_from_DE(pathogen, "FlexStepwise", "NA", "flexage", seed)
    #     n_ds = 6 + ("Influenza" in pathogen) + ("wane" in option1) + (("mimm" in option1) & ("maxmimm" not in option1)) + 2 * ((pathogen != "RSV") & ("Influenza" not in pathogen))
    #     bounds = jnp.concatenate((bounds[:n_ds], bounds[n_ds+7:]))
    #     transformed_samples = jnp.exp(posterior_samples['params']) + bounds[:,0]
    #     param_names = param_names[:n_ds] + param_names[n_ds+7:]
    #     beta_idx = param_names.index('BETA')
    #     s_rel2_idx = param_names.index('WANE2')
    #     beta_samples = transformed_samples[:,beta_idx]/REC*13
    #     s_rel2_samples = transformed_samples[:,s_rel2_idx]
    #     # plot multiple seaborn-style contour plots on the same axis
    #     sns.kdeplot(x=beta_samples, y=s_rel2_samples, levels=5, fill=True, alpha=0.3, ax=ax, color=pathogen_colors[pathogen_i])
    # # custom legend
    # legend_elements = [Patch(facecolor=pathogen_colors[i], edgecolor='k', label=pathogen) for i, pathogen in enumerate(["RSV", "Metapneumovirus", "Parainfluenza3", "Adenovirus"])]
    # ax.legend(handles=legend_elements, title="Pathogen", loc='upper right')
    # ax.set_xlabel('R0')
    # ax.set_ylabel('WANE')
    # plt.tight_layout()
    # plt.savefig("Figures/NumPyro"+str(seed)+"_R0_vs_WANEs.png", dpi=300)