## Pitzer model
## Replication of the model from Pitzer et al. 2009
## BJS August 2024

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

T = 12*30

# Initial conditions
St0 = np.array([2.9e7/960, 2.9e7*959/960, 1, 0, 0, 0, 0, 0, 0, 0])

# Parameters
B = np.log(1+0.02)/12
w0 = 0.333
w1 = 0.111
w2 = 0.083
g1 = 4.3
g2 = 8.6
s1 = 0.62
s2 = 0.35
p2 = 0.5
pA = 0.1
d1 = 0.11
d2 = 0.029
h = 0.047

seasonality = 0.055

beta0 = 0.99975*100

def dStdt(St,t):
    M, S0, I1, R1, S1, I2, R2, S2, IA, RA = St
    N = sum(St)
    beta = beta0*(1+seasonality*np.cos(2*np.pi*(t/12-0.636)))
    foi = beta*(I1+p2*I2+pA*IA)/N
    dM = B*N - w0*M - B*M
    dS0 = -foi*S0 + w0*M - B*S0
    dI1 = foi*S0 - g1*I1 - B*I1
    dR1 = g1*I1 - w1*R1 - B*R1
    dS1 = -s1*foi*S1 + w1*R1 - B*S1
    dI2 = s1*foi*S1 - g2*I2 - B*I2
    dR2 = g2*I2 - w1*R2 - B*R2
    dS2 = -s2*foi*S2 + w1*R2 + w2*RA - B*S2
    dIA = s2*foi*S2 - g2*IA - B*IA
    dRA = g2*IA - w2*RA - B*RA
    return(np.array([dM, dS0, dI1, dR1, dS1, dI2, dR2, dS2, dIA, dRA]))

result = sp.integrate.odeint(dStdt, St0, np.arange(0,T))

foi = beta0*(1+seasonality*np.cos(2*np.pi*(0.68+np.arange(0,T)/12)))*(result[:,2]+p2*result[:,5]+pA*result[:,8])/(result[:,0]+result[:,1]+result[:,2]+result[:,3]+result[:,4]+result[:,5]+result[:,6]+result[:,7]+result[:,8]+result[:,9])
H = h*d1*foi*result[:,1]+h*d2*s1*foi*result[:,4]

# Plot the results
plt.plot(H, label='Hospitalizations')
plt.ylim(0, 1500)
plt.xticks(np.arange(0, T+1, 12), np.arange(2005-30, 2005+1))
plt.xlim(12*24, 12*30)

# Plot vertical lines
for i in range(0, T//12+1):
    plt.axvline(x=12*i, color='lightgray', linewidth=0.5)
    if i%2 == 0:
        plt.axvline(x=12*i, color='lightgray', linewidth=1)
    if i%5 == 0:
        plt.axvline(x=12*i, color='gray', linewidth=1)

plt.legend()
plt.show()