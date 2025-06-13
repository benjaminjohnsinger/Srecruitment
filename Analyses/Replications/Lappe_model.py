import jax.numpy as jnp
from scipy.integrate import odeint

def mu(t):
    if t < 30*365:
        return 12.21/(1000*365)
    else:
        mus = jnp.array([12.21,11.84,11.59,11.42,10.97,11.04,11.00])/(1000*365)
        time = (t-30*365) / 365
        return mus[int(time)]

def sir_model(y, t, q, sigma, chirho, A, theta):
    S, I, R = y
    N = S + I + R

    # Seasonal force of infection
    beta = q * (1 + A * jnp.cos(theta + 2 * jnp.pi * t / 365))
    mu_t = mu(t)

    dSdt = mu_t * N - beta * S * I / N - mu_t * S - chirho * S
    dIdt = beta * S * I / N - sigma * I - mu_t * I
    dRdt = sigma * I - mu_t * R + chirho * S

    return [dSdt, dIdt, dRdt]

# Parameters
N_start = 23970000  # Total population
c = 14 # Contacts per day
q = 0.397 * c  # Transmission rate
sigma = 0.2  # Recovery rate
chirho = 0.9*0.77/1750 # Vaccination rate
A = 0.08  # Seasonal variation amplitude
theta = 0.02  # Seasonal variation phase
reporting = 0.029  # Reporting fraction

# Initial conditions
S0 = (N_start*0.05 - 1)
I0 = 1
R0 = N_start*0.95
y0 = [S0, I0, R0]

# Time vector
period = 36
times = jnp.linspace(0, 365*period, 365*period)

# Solve the ODE system
sol = odeint(sir_model, y0, times, args=(q, sigma, chirho, A, theta))

# Plotting
import matplotlib.pyplot as plt

start_year = 30
plt.plot(times[365*start_year:], sol[365*start_year:, 1]*reporting, label='I')
plt.xticks(jnp.arange(365*start_year, 365*period+1, 365), jnp.arange(2024-period+start_year-1, 2024))


plt.xlabel('Time (Years)')
plt.ylabel('Population')
plt.legend()
plt.show()