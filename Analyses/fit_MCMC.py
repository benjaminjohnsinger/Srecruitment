import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm as colormaps
import time
hsv_colors = colormaps.hsv(-0.02+np.arange(7)/7)
hsv_colors[3] = colormaps.hsv((3/7)+0.04)
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController

from Parameters.census_population import CENSUS_AGE_POP, AGING_RATE
from JAX_ODEs import deltas
N_C = 2
NAG = 7
N_S = 3
from utils import date_to_t, parameters_from_DE, x_to_params
from Gemini_vaccination import FluRatePreprocessor
import time
import types

## POINTS must start  (at least) len(p_time_to_obs) days before the first observation to avoid issues from jnp.roll behaviour
def SIS_likelihood(data, params, POINTS, STATE0, p_time_to_obs, age=True, incidence=True, start_t=date_to_t(pd.to_datetime('1970-01-01')), overdispersion=False, solution=None):
    # run simulation
    if solution is None:
        term = ODETerm(deltas)
        solver = Dopri5()
        saveat = SaveAt(ts=POINTS)
        step_controller = PIDController(rtol=1e-5, atol=1e-5)
        solution = diffeqsolve(
                            term, solver,
                            t0=0, t1=int(POINTS[-1]), dt0=None, stepsize_controller=step_controller,
                            saveat=saveat, y0=STATE0.flatten(), args=params, 
                            max_steps=None,  
                            )
        values = solution.ys.T
    else:
        values = solution.ys.T

    # convert into observed cases
    trajectory = np.diff(values[-NAG:,:],axis=1).T

    # format data into cases, rescaled appropriately by population age distribution
    if incidence:
        if age:
            cases = np.round(data*np.array([np.sum(values[range(i,N_S*N_C*NAG,NAG),len(p_time_to_obs):],axis=0) for i in range(NAG)]).T)
        else:
            cases = data*np.sum(values,axis=0)
    else:
        cases = data.copy()

    # the expected observations for a given date are the observations on each day i days prvious multiplied by the probability of detection i days after infection
    expected_obs = np.sum([np.roll(trajectory,i,axis=0)*p_time_to_obs[i] for i in range(len(p_time_to_obs))],axis=0)
    # cut off the first few days of the trajectory since they are not used in the likelihood (and the roll function is wrapping around)
    expected_obs = np.maximum(0,expected_obs[-len(cases):])
    
    # calculate the log likelihood
    if overdispersion:
        p = overdispersion/(overdispersion+expected_obs)
        likelihood = sp.stats.nbinom.logpmf(cases,overdispersion,p).sum()
    else:
        likelihood = sp.stats.poisson.logpmf(cases,expected_obs).sum()
    return likelihood

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
    solution = sp.optimize.least_squares(objective_func, x0=initial_guess, method='trf', xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=200)

    if not solution.success:
        raise RuntimeError("Optimization failed: " + solution.message)
    
    found_params_flat = solution.x
    unconstrained_loc = found_params_flat[:n_params]
    L_elements = found_params_flat[n_params:]
    L = jnp.zeros((n_params,n_params)).at[tril_indices].set(L_elements)
    unconstrained_cov = L @ L.T
    return unconstrained_loc, unconstrained_cov
    

def prior_distribution(filename, bounds, n=1000, dist_type="multilog", varlim=None, pathogen="RSV"):
    de = jnp.asarray(pd.read_csv(filename).values)
    # sort by first column (best likelihood)
    de = de[jnp.argsort(de[:,0]),:]
    if varlim is not None:
        n_ds = 6 + ("Influenza" in pathogen) + 2 * ((pathogen != "RSV") & ("Influenza" not in pathogen))
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

def fit_MCMC(pathogen, lockdown, option1, option2, seed, import_multiplier = 1e-9, vax_preprocessor=None, samples=1000, varlim=None):
    params, param_names, bounds, incidence, p_time_to_obs = parameters_from_DE(pathogen, lockdown, option1, option2, seed)
    age_pops = jnp.asarray(pd.read_csv("Data/Processed/age_pops_daily.csv").values)
    cases = jnp.round(incidence*age_pops[-len(incidence):])
    p_time_to_obs_flipped = jnp.flip(p_time_to_obs.flatten())
    def obs_convolution(x):
        return jnp.convolve(x, p_time_to_obs_flipped, mode='same')
    @jax.jit
    def trajectory(state0, params, term=ODETerm(deltas), solver=Dopri5(), step_controller=PIDController(rtol=1e-5, atol=1e-5), startdate='2015-10-01', enddate='2023-10-01'):
        saveat = SaveAt(ts=jnp.arange(date_to_t(startdate)-90, date_to_t(enddate)))
        solution = diffeqsolve(
                        term, solver,
                        t0=0, t1=date_to_t(enddate)-1, dt0=None, stepsize_controller=step_controller,
                        saveat=saveat, y0=state0.flatten(), args=params, 
                        max_steps=100000,  
                        )
        # trajectory is np.diff over time of last NAG elements of solution
        cumulative_observations = solution.ys[:,-NAG:]  # shape (DAYS, NAG)
        observations = jnp.diff(cumulative_observations, axis=0)  # shape (DAYS-1, NAG)
        expected_obs = jax.vmap(obs_convolution, in_axes=1, out_axes=1)(observations)
        return expected_obs
    STATE0_shaped = jnp.zeros((2*N_S+1,NAG))
    STATE0_shaped = STATE0_shaped.at[0,:].set(CENSUS_AGE_POP-1)
    STATE0_shaped = STATE0_shaped.at[1,:].set(1)
    STATE0 = jnp.concatenate((jnp.array([0]), STATE0_shaped.flatten()))
    prior_dist, prior_means = prior_distribution(f"Data/Processed/DE_outputs/DE_{pathogen}{lockdown}{option1}{option2}{seed}_sorted.csv", bounds,
        n=1000, dist_type="multilog", varlim = varlim, pathogen=pathogen)
    if varlim is not None:
        option1 = varlim
        n_ds = 6 + ("Influenza" in pathogen) + 2 * ((pathogen != "RSV") & ("Influenza" not in pathogen))
        if varlim == "dynamicpathogen":
            bounds = bounds[:n_ds]
        elif varlim == "dynamic":
            bounds = bounds[:n_ds+7]
        elif varlim == "pathogen":
            bounds = jnp.concatenate((bounds[:n_ds], bounds[n_ds+7:]))
    if "Influenza" in pathogen:
        if vax_preprocessor is None:
            vax_preprocessor = FluRatePreprocessor(jnp.arange(0, date_to_t('2023-10-01')), age_pops, AGING_RATE)
    def model(obs_cases=None):
        # sample parameters from prior
        sample = numpyro.sample("params", prior_dist)
        x_sampled = jnp.exp(sample) + bounds[:,0]
        sim_params = x_to_params(x_sampled, pathogen, lockdown, option1, option2, vax_preprocessor=vax_preprocessor, fixed_params=params, import_multiplier=import_multiplier)
        expected_obs = trajectory(STATE0, sim_params)[-len(obs_cases):]
        # very sharp softplus to avoid issues with Poisson likelihood while keeping close to original
        softplus_obs = jax.nn.softplus(expected_obs*1000)/1000
        numpyro.sample("obs_cases", dist.Poisson(softplus_obs), obs=obs_cases)
    nuts_kernel = NUTS(model,
                        init_strategy=numpyro.infer.init_to_value(values={"params": prior_means}),
                        max_tree_depth=6)
    mcmc = MCMC(nuts_kernel, num_warmup=jnp.minimum(samples,1000), num_samples=samples)
    mcmc.run(jax.random.PRNGKey(seed), obs_cases=cases)
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
    
    def plot_trajectories(posterior_samples, params, restricted_bounds, incidence, p_time_to_obs, downsample=False, ax=None):
        if downsample:
            posterior_samples = {k: v[np.random.choice(v.shape[0], size=downsample, replace=False)] for k, v in posterior_samples.items()}
        age_pops = jnp.asarray(pd.read_csv("Data/Processed/age_pops_daily.csv").values)
        cases = jnp.round(incidence*age_pops[-len(incidence):])
        p_time_to_obs_flipped = jnp.flip(p_time_to_obs.flatten())
        def obs_convolution(x):
            return jnp.convolve(x, p_time_to_obs_flipped, mode='same')
        @jax.jit
        def trajectory(state0, params, term=ODETerm(deltas), solver=Dopri5(), step_controller=PIDController(rtol=1e-5, atol=1e-5), startdate='2015-10-01', enddate='2023-10-01'):
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
            softplus_obs = jax.nn.softplus(expected_obs*1000)/1000
            return softplus_obs
        STATE0_shaped = jnp.zeros((2*N_S+1,NAG))
        STATE0_shaped = STATE0_shaped.at[0,:].set(CENSUS_AGE_POP-1)
        STATE0_shaped = STATE0_shaped.at[1,:].set(1)
        STATE0 = jnp.concatenate((jnp.array([0]), STATE0_shaped.flatten()))
        transformed_samples = jnp.exp(posterior_samples['params']) + restricted_bounds[:,0]
        sim_params = jax.vmap(lambda x: x_to_params(x, pathogen, "FlexStepwise", "pathogen", "flexage", vax_preprocessor=FluRatePreprocessor(jnp.arange(0, date_to_t('2023-10-01')), age_pops, AGING_RATE), fixed_params=params, import_multiplier=1e-9))(transformed_samples)
        trajectories = jax.vmap(lambda p: trajectory(STATE0, p)) (sim_params)
        rolling_average_cases = jnp.convolve(cases.sum(axis=1), jnp.ones(14)/14, mode='same')
        ax.plot(rolling_average_cases, color='black', label='Observed Cases')
        ax.plot(trajectories.sum(axis=2).T[-len(cases):], color='red', alpha=0.3)

    import pickle
    print(jax.local_device_count())
    start = time.time()
    for pathogen in ["RSV", "Metapneumovirus", "InfluenzaA", "InfluenzaB", "Parainfluenza3", "Adenovirus"]:
        print(pathogen, time.time()-start)
        mcmc = fit_MCMC(pathogen, "FlexStepwise", "0.005", "flexage", 2507092, import_multiplier=1e-9, samples=1000, varlim="pathogen")
        mcmc.print_summary()
        # save samples
        posterior_samples = mcmc.get_samples()
        with open("Data/Processed/MCMC_outputs/MCMC_"+pathogen+"FlexStepwise0.005flexage250709_pathogen_samples_sp1000.pickle", "wb") as f:
            pickle.dump(posterior_samples, f)
        # plot histograms of each parameter
        sns.set_style("whitegrid")
        n = posterior_samples['params'].shape[1]//2 + posterior_samples['params'].shape[1]%2
        fig, ax = plt.subplots(n, 2, figsize=(12, 3*n))
        params, param_names, bounds, incidence, p_time_to_obs = parameters_from_DE(pathogen, "FlexStepwise", "0.005", "flexage", 2507092)
        n_ds = 6 + ("Influenza" in pathogen) + 2 * ((pathogen != "RSV") & ("Influenza" not in pathogen))
        bounds = jnp.concatenate((bounds[:n_ds], bounds[n_ds+7:]))
        transformed_samples = jnp.exp(posterior_samples['params']) + bounds[:,0]
        param_names = param_names[:n_ds] + param_names[n_ds+7:]
        for i in range(n):
            for j in range(2):
                idx = i*2 + j
                if idx < len(param_names):
                    plot_histogram(transformed_samples[:,idx], param_names[idx], ax=ax[i,j])
        plt.tight_layout()
        plt.savefig("Figures/NumPyro_test_pathogen_variables_"+pathogen+"_sp1000.png", dpi=300)
        plt.close()
        fig, ax = plt.subplots(figsize=(13.3,7.5))
        plot_trajectories(posterior_samples, params, bounds, incidence, p_time_to_obs, downsample=100, ax=ax)
        plt.xlabel('Days since 1970-01-01')
        plt.ylabel('Number of Cases')
        plt.title(f'Posterior Predictive Trajectories for {pathogen}')
        plt.legend()
        plt.tight_layout()
        plt.savefig("Figures/NumPyro_test_trajectories_"+pathogen+"_sp1000.png", dpi=300)

    ## 2d contour plot comparisons
    # fig, ax = plt.subplots(figsize=(6.5,6.5))
    # ax.set_xlim(0.05, 0.4)
    # ax.set_ylim(0, 0.01)
    # for pathogen in ["RSV", "Metapneumovirus", "Parainfluenza3", "Adenovirus"]:
    #     #2d histograms of beta vs S_REL2
    #     with open("Data/Processed/MCMC_outputs/MCMC_"+pathogen+"FlexStepwise0.005flexage250709_pathogen_samples.pickle", "rb") as f:
    #         posterior_samples = pickle.load(f)
    #     params, bounds_dict, incidence, p_time_to_obs = parameters_from_DE(pathogen, "FlexStepwise", "0.005", "flexage", 2507092)
    #     bounds = jnp.array(list(bounds_dict.values()))
    #     n_ds = 6 + ("Influenza" in pathogen) + 2 * ((pathogen != "RSV") & ("Influenza" not in pathogen))
    #     bounds = jnp.concatenate((bounds[:n_ds], bounds[n_ds+7:]))
    #     transformed_samples = jnp.exp(posterior_samples['params']) + bounds[:,0]
    #     param_names = list(bounds_dict.keys())
    #     param_names = param_names[:n_ds] + param_names[n_ds+7:]
    #     beta_idx = param_names.index('BETA')
    #     s_rel2_idx = param_names.index('WANE')
    #     beta_samples = transformed_samples[:,beta_idx]
    #     s_rel2_samples = transformed_samples[:,s_rel2_idx]
    #     # plot multiple seaborn-style contour plots on the same axis
    #     sns.kdeplot(x=beta_samples, y=s_rel2_samples, levels=5, fill=True, alpha=0.3, ax=ax)
    # # custom legend
    # from matplotlib.patches import Patch
    # legend_elements = [Patch(facecolor=hsv_colors[i], edgecolor='k', label=pathogen) for i, pathogen in enumerate(["RSV", "Metapneumovirus", "Parainfluenza3", "Adenovirus"])]
    # ax.legend(handles=legend_elements, title="Pathogen", loc='lower right')
    # ax.set_xlabel('BETA')
    # ax.set_ylabel('WANE')
    # plt.tight_layout()
    # plt.savefig("Figures/NumPyro_test_BETA_vs_WANEs.png", dpi=300)