## Reproducing Pitzer model with age groups
## Full reproduction of the model from Pitzer et al. 2009, including age groups
## BJS August 2024

import jax.numpy as jnp
import scipy as sp
import matplotlib.pyplot as plt

## Pitzer SIRSIRSIRS model
T = 12*50

agep = jnp.concatenate(((1/960)*jnp.ones(12), (1/80)*jnp.ones(4), jnp.array([1/16, 1/8, 1/4, 1/4, 1/4])))
al = len(agep)
u = jnp.concatenate((1*jnp.ones(12), (1/12)*jnp.ones(4), jnp.array([1/(12*5), 1/120, 1/240, 1/240, 1/126])))
c = jnp.concatenate((1.555*jnp.ones(12), jnp.array([2.312,1.738]), jnp.ones(7)))

print(jnp.sum(c*agep))
print(jnp.mean(c))

N = 2.9e7

St0 = jnp.concatenate((N/960*jnp.ones(1), jnp.zeros(al-1), N*agep - jnp.ones(al), jnp.ones(al), jnp.zeros(7*al)))

# Parameters
w0 = (1/3)
w1 = (1/9)
w2 = (1/12)
g1 = 4.3
g2 = 8.6
s1 = 0.62
s2 = 0.35
p2 = 0.5
pA = 0.1
d1 = 0.11
d2 = 0.029
h = 0.041

Byr = 0.017  # Birth rate per year
Bca = jnp.array([0.0206, 0.0200, 0.0194, 0.0187, 0.0180, 0.0174, 0.0168, 0.0162, 0.0158, 0.0155, 0.0157, 0.0153, 0.0151, 0.0152, 0.0152, 0.0152, 0.0154])
Bus = jnp.array([0.0167, 0.0162, 0.0158, 0.0154, 0.0150, 0.0146, 0.0144, 0.0142, 0.0143, 0.0142, 0.0144, 0.0141, 0.0139, 0.0141, 0.0140, 0.0140, 0.0142])
B = jnp.zeros((T,al))
B[0:240,0] = Byr
B[240:300,0] = Bus[0]
B[300:(300+15*12),0] = jnp.repeat(Bca[1:16],12)
B[(300+15*12):(300+15*12+120),0] = Bca[16]

v = jnp.zeros((T,al))
v[:,0] = jnp.concatenate((jnp.zeros(480), jnp.ones(120)))*0.96*(8/10)

seasonality = 0.055

ptrans = 23.25
b1 = .0466
phi = .636

b = ptrans*g1
beta = b/N

# print(beta*c)

def dStdt(St,t):
    dSt = jnp.zeros(len(St))
    foi = (1+b1*jnp.cos(2*jnp.pi*(t/12-phi)))*beta*c*jnp.sum((St[2*al:3*al]+p2*St[5*al:6*al]+pA*St[8*al:9*al]))
    dSt[0:al] = (1-v[int(t),:])*jnp.log(1+B[int(t)-1,:])*jnp.sum(St)/12 - w0*St[0:al]  - u*St[0:al] + jnp.concatenate((jnp.zeros(1), u[:-1]*St[0:al-1]))
    dSt[al:2*al] = -foi*St[al:2*al] + w0*St[0:al] - u*St[al:2*al] + jnp.concatenate((jnp.zeros(1), u[:-1]*St[al:2*al-1]))
    dSt[2*al:3*al] = foi*St[al:2*al] - g1*St[2*al:3*al] - u*St[2*al:3*al] + jnp.concatenate((jnp.zeros(1), u[:-1]*St[2*al:3*al-1]))
    dSt[3*al:4*al] = g1*St[2*al:3*al] - w1*St[3*al:4*al] - u*St[3*al:4*al] + jnp.concatenate((jnp.zeros(1), u[:-1]*St[3*al:4*al-1]))
    dSt[4*al:5*al] = v[int(t),:]*jnp.log(1+B[int(t)-1,:])*jnp.sum(St)/12 - s1*foi*St[4*al:5*al] + w1*St[3*al:4*al] - u*St[4*al:5*al] + jnp.concatenate((jnp.zeros(1), u[:-1]*St[4*al:5*al-1]))
    dSt[5*al:6*al] = s1*foi*St[4*al:5*al] - g2*St[5*al:6*al] - u*St[5*al:6*al] + jnp.concatenate((jnp.zeros(1), u[:-1]*St[5*al:6*al-1]))
    dSt[6*al:7*al] = g2*St[5*al:6*al] - w1*St[6*al:7*al] - u*St[6*al:7*al] + jnp.concatenate((jnp.zeros(1), u[:-1]*St[6*al:7*al-1]))
    dSt[7*al:8*al] = -s2*foi*St[7*al:8*al] + w1*St[6*al:7*al] + w2*St[9*al:10*al] - u*St[7*al:8*al] + jnp.concatenate((jnp.zeros(1), u[:-1]*St[7*al:8*al-1]))
    dSt[8*al:9*al] = s2*foi*St[7*al:8*al] - g2*St[8*al:9*al] - u*St[8*al:9*al] + jnp.concatenate((jnp.zeros(1), u[:-1]*St[8*al:9*al-1]))
    dSt[9*al:10*al] = g2*St[8*al:9*al] - w2*St[9*al:10*al] - u*St[9*al:10*al] + jnp.concatenate((jnp.zeros(1), u[:-1]*St[9*al:10*al-1]))
    return(dSt)

result = sp.integrate.odeint(dStdt, St0, jnp.arange(0,T))

foi = beta*(1+seasonality*jnp.cos(2*jnp.pi*(jnp.arange(0,T)/12-phi)))*jnp.sum(c*(result[:,2*al:3*al]+p2*result[:,5*al:6*al]+pA*result[:,8*al:9*al]), axis=1)
H = h*d1*foi*jnp.sum(result[:,al:2*al], axis=1)+h*d2*s1*foi*jnp.sum(result[:,4*al:5*al], axis=1)

# Plot the results
plt.plot(H, label='Hospitalizations')
# plt.plot(jnp.sum(result[:,2*3:3*3],axis=1), label='First infections')
# plt.plot(jnp.sum(result[:,5*3:6*3],axis=1), label='Second infections')
# plt.plot(jnp.sum(result[:,8*3:9*3],axis=1), label='Asymptomatic infections')

viridis = plt.get_cmap('viridis')

# # Plot size of each age group
# for i in range(al):
#     plt.plot(jnp.sum([result[:,j*al+i] for j in range(10)],axis=0), label=str(i),color=viridis(i/al))
# # log y axis
# # plt.xscale('log')    
# plt.yscale('log')

# # horizontal lines at N*agep colored acording to age
# for i in range(al):
#     plt.axhline(y=N*agep[i], color=viridis(i/al), linewidth=0.5)

# Plot vertical lines
for i in range(T//12+1):
    plt.axvline(x=12*i, color='lightgray', linewidth=0.5)
    if i%2 == 0:
        plt.axvline(x=12*i, color='lightgray', linewidth=1)
    if i%5 == 2:
        plt.axvline(x=12*i, color='gray', linewidth=1)

plt.xticks(jnp.arange(0, T+1, 12), jnp.arange(2018-50, 2018+1))
# plt.xlim(300-24, T-24)
# plt.ylim(0, 1500)

plt.legend()
plt.show()
