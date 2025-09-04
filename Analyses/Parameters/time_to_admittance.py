## BJS Jan 2025
## Processing distributions of incubation and admittance times for RSV and influenza

# import jax.numpy as jnp

incubation_median_RSV = 4.4
incubation_dispersion_RSV = 1.24
incubation_distribution_RSV = sp.stats.lognorm(jnp.log(incubation_dispersion_RSV),scale=incubation_median_RSV)

admittance_logmean_RSV = 1.85
admittance_logsd_RSV = 0.762
admittance_distribution_RSV = sp.stats.lognorm(admittance_logsd_RSV,scale=jnp.exp(admittance_logmean_RSV))

incubation_median_fluA = 1.4 # Lessler 2009
incubation_dispersion_fluA = 1.51
incubation_distribution_fluA = sp.stats.lognorm(jnp.log(incubation_dispersion_fluA),scale=incubation_median_fluA)

incubation_median_fluB = 0.6
incubation_dispersion_fluB = 1.51
incubation_distribution_fluB = sp.stats.lognorm(jnp.log(incubation_dispersion_fluB),scale=incubation_median_fluB)

admittance_mu_flu = 1.92 
admittance_sigma_flu = 0.914
admittance_Q_flu = 0.126
a = admittance_Q_flu**(-2)
c = admittance_Q_flu/admittance_sigma_flu
scale = jnp.exp(admittance_mu_flu-jnp.log(a)/c)
admittance_distribution_flu = sp.stats.gengamma(a,c,scale=scale)

# # generate 100k samples of incubation plus admittance
jnp.random.seed(241125)
N = int(1e8)
infection_to_admittance = incubation_distribution_fluB.rvs(N) + admittance_distribution_flu.rvs(N)
jnp.savetxt('Data/Processed/Influenza_B_incubation_admittance_distribution.csv',n,delimiter=',')
