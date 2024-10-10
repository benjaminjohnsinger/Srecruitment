## SIR model with n susceptibility classes, for a single pathogen
## BJS September 2024

import numpy as np
import scipy as sp
import time
import itertools as it
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle

from vaccination import birth_vax, all_vax
import contact_model as cm
from SIRn_ODEs import single_pathogen_deltas as deltas
from Parameters.test_population import *
from Parameters.adult_disease import *

from plotting import *

## Period of simulation in months
T_FACTOR = 1
PERIOD = int(T_FACTOR*12*50)

# Correct parameters with T_FACTOR
WANE_UP /= T_FACTOR
WANE_SAME /= T_FACTOR
REC /= T_FACTOR
AGING_RATE /= T_FACTOR
BIRTH_RATE /= T_FACTOR
ACOV_SCALED = lambda t,T_VAX : ACOV(t,T_VAX)/T_FACTOR

## Contacts and force of infection
IMPORT = 0.01
# Contact matrix for all contact types
CONTACT = np.genfromtxt('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',')
CONTACT /= 12.99 # transform to contact proportions
# Lockdown and other mobility changes
T_LOCKDOWN = 37*12*T_FACTOR
LOCKDOWN_DURATION = 12*T_FACTOR
LOCKDOWN_REDUCTION = 0.4
shape_step = lambda t : cm.STEP(t,T_LOCKDOWN,LOCKDOWN_DURATION,LOCKDOWN_REDUCTION)
shape_static = lambda t : 1
def contact(t,shape,seasonality=SEASONALITY,offset=OFFSET,t_factor=T_FACTOR,c_rate=CONTACT):
    return shape(t)*(1+seasonality*np.cos(2*np.pi*(t/(12*t_factor)-offset)))*c_rate

## Initial conditions
STATE0 = np.zeros((1+N_S*3)*(NAG))
STATE0[NAG:2*NAG] = KP_AGE_POP-1 # Everyone is susceptible except
STATE0[2*NAG:3*NAG] = 1 # one individual in each age group that is infected.

## Integrate the system
# POINTS = np.concat((np.zeros(1),np.arange(T_LOCKDOWN-5*12,T_LOCKDOWN+LOCKDOWN_DURATION+5*12,0.1),np.ones(1)*PERIOD))
POINTS = np.arange(0,PERIOD+1,1)
T_VAX = PERIOD

# S_REL = S_REL**2

# fig = plt.figure(figsize=(6.5,6.5),constrained_layout=True)
# gs = GridSpec(3,3,figure=fig)
# ax1 = fig.add_subplot(gs[0,0])
# ax2 = fig.add_subplot(gs[0,1:])
# ax3 = fig.add_subplot(gs[1,0])
# ax4 = fig.add_subplot(gs[2,0])
# ax5 = fig.add_subplot(gs[1:,1:])

# BETA = 0.5*BETA
# REC = 0.5*REC
S_REL = np.ones(N_S)
params = {'NAG':NAG, 'N_S':N_S, 'AGING_RATE':AGING_RATE, 'BIRTH_RATE':BIRTH_RATE, 'WANE_UP':WANE_UP, 'WANE_SAME':WANE_SAME, 'REC':REC, 'S_REL':S_REL, 'S_AGE':S_AGE, 'I_REL':I_REL, 'P_OBS':P_OBS, 'birth_vax':birth_vax, 'all_vax':all_vax, 'S_VAX':S_VAX, 'ACOV_SCALED':ACOV_SCALED, 'BCOV':BCOV, 'T_VAX':T_VAX,
    'IMPORT':IMPORT, 'BETA':BETA, 'contact':lambda t : contact(t, shape_step,SEASONALITY,OFFSET,T_FACTOR,CONTACT)}
result = sp.integrate.solve_ivp(deltas,(0,PERIOD),STATE0,method='RK45',t_eval=POINTS,args=(params,))

# # plot each compartment
# fig, axes = plt.subplots(3,1,figsize=(6.5,6.5),sharex=True)
# for i in range(N_S):
#     axes[0].plot(result.t,np.sum(result.y[(3*i+1)*NAG:(3*i+2)*NAG,:],axis=0), label='S'+str(i+1))
#     axes[1].plot(result.t,np.sum(result.y[(3*i+2)*NAG:(3*i+3)*NAG,:],axis=0), label='I'+str(i+1))
#     axes[2].plot(result.t,np.sum(result.y[(3*i+3)*NAG:(3*i+4)*NAG,:],axis=0), label='R'+str(i+1))
# axes[0].set_xlim(25*12,43*12)
# axes[1].set_ylim(0,5e4)
# axes[2].set_ylim(0,7e5)
# axes[0].legend()
# axes[1].legend()
# axes[2].legend()
# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(figsize=(6.5,6.5))

# start = time.time()
# results = sim_grid(STATE0,params,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,N=10,
# grid_mode=("scale","fade_vec"))
# with open('Data/Processed/results_growth_wane_vec.pkl','wb') as f:
#     pickle.dump(results,f)
# end = time.time()
# print('Time to get results:',end-start)
# # # load results
# with open('Data/Processed/results_by_acqimm.pkl','rb') as f:
#     results_by_acqimm = pickle.load(f)
# fig, axes = plt.subplots(3,1,figsize=(6.5,6.5),sharex=True)
# param_line_plot(axes[0],results_by_acqimm,params,T_LOCKDOWN,LOCKDOWN_DURATION,
# grid_params=("I_REL",),grid_mode="fade_vec",factor=1,label_mode="mean",
# y_values=("child infections",),y_labels=("Child",),x_label="",
# colors=("#648FFF",))
# param_line_plot(axes[1],results_by_acqimm,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE=OBS_AGE,
# grid_params=("I_REL",),grid_mode="fade_vec",factor=1,label_mode="mean",
# y_values=("time to rebound",),y_labels=("Time to rebound",),x_label="",
# colors=("#FFB000",))
# param_line_plot(axes[2],results_by_acqimm,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE=OBS_AGE,
# grid_params=("I_REL",),grid_mode="fade_vec",factor=1,label_mode="mean",
# y_values=("rebound peak incidence",),y_labels=("Rebound peak",),x_label="Acquired immunity",
# colors=("#DC267F",))
# axes[0].set_ylabel('Child-caused infections')
# axes[1].set_ylabel('Time to rebound')
# axes[2].set_ylabel('Rebound size')
# plt.tight_layout()
# plt.show()

# measure time to get results
start = time.time()
results = sim_grid(STATE0,params,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,N=10)
# save results
with open('Data/Processed/test_adult_no_SREL.pkl','wb') as f:
    pickle.dump(results,f)
end = time.time()
print('Time to get results:',end-start)
# # load results
# with open('Data/Processed/results_by_wane_low.pkl','rb') as f:
#     results = pickle.load(f)
fig, axes = plt.subplots(4,2,figsize=(6.5,8.5))
z_values = [["child infections","under-five infections"],
["peak incidence","rebound peak incidence"],
["periodicity","time to rebound"],
["pre-lockdown susceptibility","post-lockdown susceptibility"]]
titles = [["Child infections","Under-five infections"],
["Peak incidence","Rebound peak"],
["Periodicity","Time to rebound"],
["Pre-lockdown\nsusceptibility","Post-lockdown\nsusceptibility"]]
cbar_labels = [["","Infections"],
["","Observed infections"],
["","Years"],
["","Susceptibility"]]
for i in range(4):
    for j in range(2):
        grid_plot(axes[i,j],results,params,T_LOCKDOWN,LOCKDOWN_DURATION,OBS_AGE=OBS_AGE,
        grid_params=(("BETA","REC"),("WANE_UP","WANE_SAME")),grid_mode=("scale","scale"),factors=(1,1),
        z_value=z_values[i][j],z_label=cbar_labels[i][j])
        axes[i,j].set_title(titles[i][j])
        if i == 3:
            axes[i,j].set_xlabel("Waning")
        if j == 0:
            axes[i,j].set_ylabel("Growth rate")
plt.tight_layout()
plt.show()


# w36_up = 0.36*np.array([1,1,0])
# w36_same = 0.36*np.array([0,0,1])
# g64_rec = 14.2*np.ones(3)
# g64_beta = 78.2
# params_g64_w36 = ((NAG, N_S, AGING_RATE, BIRTH_RATE, w36_up, w36_same, g64_rec, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV_SCALED, BCOV, T_VAX,
#     IMPORT, g64_beta,lambda t : contact(t, shape_step,SEASONALITY,OFFSET,T_FACTOR,CONTACT)),)
# result_g64_w36 = sp.integrate.solve_ivp(deltas,(0,PERIOD),STATE0,method='RK45',t_eval=POINTS,args=params_g64_w36)
# w24_up = 0.24*np.array([1,1,0])
# w24_same = 0.24*np.array([0,0,1])
# params_g64_w24 = ((NAG, N_S, AGING_RATE, BIRTH_RATE, w24_up, w24_same, g64_rec, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV_SCALED, BCOV, T_VAX,
#     IMPORT, g64_beta,lambda t : contact(t, shape_step,SEASONALITY,OFFSET,T_FACTOR,CONTACT)),)
# result_g64_w24 = sp.integrate.solve_ivp(deltas,(0,PERIOD),STATE0,method='RK45',t_eval=POINTS,args=params_g64_w24)
# g48_rec = 10.6*np.ones(3)
# g48_beta = 58.7
# params_g48_w36 = ((NAG, N_S, AGING_RATE, BIRTH_RATE, w36_up, w36_same, g48_rec, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV_SCALED, BCOV, T_VAX,
#     IMPORT, g48_beta,lambda t : contact(t, shape_step,SEASONALITY,OFFSET,T_FACTOR,CONTACT)),)
# result_g48_w36 = sp.integrate.solve_ivp(deltas,(0,PERIOD),STATE0,method='RK45',t_eval=POINTS,args=params_g48_w36)
# g40_rec = 8.9*np.ones(3)
# g40_beta = 48.9
# params_g40_w36 = ((NAG, N_S, AGING_RATE, BIRTH_RATE, w36_up, w36_same, g40_rec, S_REL, S_AGE, I_REL, P_OBS, birth_vax, all_vax, S_VAX, ACOV_SCALED, BCOV, T_VAX,
#     IMPORT, g40_beta,lambda t : contact(t, shape_step,SEASONALITY,OFFSET,T_FACTOR,CONTACT)),)
# result_g40_w36 = sp.integrate.solve_ivp(deltas,(0,PERIOD),STATE0,method='RK45',t_eval=POINTS,args=params_g40_w36)

# fig, axes = plt.subplots(2,1,figsize=(6.5,6))
# mx_g64_w24 = lockdown_incidence_plot(axes[0],STATE0,params_g64_w24,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result_g64_w24,label='g64_w24',color='#DC267F')
# mx_g64_w36 = lockdown_incidence_plot(axes[0],STATE0,params_g64_w36,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result_g64_w36,label='g64_w36',color='#648FFF')
# mx_g48_w36 = lockdown_incidence_plot(axes[0],STATE0,params_g48_w36,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result_g48_w36,label='g48_w36',color='#FFB000')
# mx_g40_w36 = lockdown_incidence_plot(axes[0],STATE0,params_g40_w36,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result_g40_w36,label='g40_w36',color='#785EF0')
# lockdown_incidence_format(axes[0],POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,mx=max(mx_g64_w36,mx_g64_w24,mx_g48_w36,mx_g40_w36))
# lockdown_susceptibility_plot(axes[1],STATE0,params_g64_w24,PERIOD,POINTS,T_LOCKDOWN,result_g64_w24,color='#DC267F',relative=False)
# lockdown_susceptibility_plot(axes[1],STATE0,params_g64_w36,PERIOD,POINTS,T_LOCKDOWN,result_g64_w36,color='#648FFF',relative=False)
# lockdown_susceptibility_plot(axes[1],STATE0,params_g48_w36,PERIOD,POINTS,T_LOCKDOWN,result_g48_w36,color='#FFB000',relative=False)
# lockdown_susceptibility_plot(axes[1],STATE0,params_g40_w36,PERIOD,POINTS,T_LOCKDOWN,result_g40_w36,color='#785EF0',relative=False)
# lockdown_susceptibility_format(axes[1],POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,ymin=2.5e7,ymax=2.62e7)
# axes[0].legend()
# plt.tight_layout()
# plt.savefig('Figures/lockdown_incidence_g64_w36_w24_g48_w36_g40_w36.png',dpi=300)



# fig, axes = plt.subplots(1,2,figsize=(6.5,3))
# age_infect_grid_plot(axes,STATE0,params,AGE_GROUP_NAMES,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,N=25,N_span=20)
# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(3,1,figsize=(6.5,8.5))
# mx = lockdown_incidence_plot(axes[0],STATE0,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result,by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES)
# lockdown_incidence_format(axes[0],POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,mx)
# lockdown_susceptibility_plot(axes[1],STATE0,params,PERIOD,POINTS,T_LOCKDOWN,result,by_age=True,AGE_GROUP_NAMES=AGE_GROUP_NAMES)
# lockdown_susceptibility_format(axes[1],POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,ymin=0,ymax=1)
# age_infect_plot(axes[2],STATE0,params,AGE_GROUP_NAMES,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result)
# plt.tight_layout()
# axes[0].legend()
# plt.savefig('Figuchild_age_group_susc_and_inf.png',dpi=300)
# plt.show()



# viridis = plt.cm.get_cmap('viridis', NAG)

# # plt.plot(result_static.t,np.sum(obs_static,axis=1)/pop_size, label='Observed cases',color='blue')
# # plt.plot(result.t,np.sum(obs,axis=1)/pop_size, label='Observed cases')
# plt.xlim(25*12,43*12)
# # plt.ylim(0,7e-4)
# mx = 1.1*np.max((np.sum(obs,axis=1)/pop_size)[int(2*POINTS/3):POINTS])
# # plt.ylim(mn,mx)
# plt.ylabel('Observed incidence')
# plt.fill_between([T_LOCKDOWN,T_LOCKDOWN+LOCKDOWN_DURATION],0,mx,color='gray',alpha=0.2)
# plt.xticks(np.arange(25*12,44*12,12),[str(int(x)-13) for x in np.arange(0,19,1)])
# plt.xlabel('Time (years)')
# plt.title('Incidence of non-pediatric disease with 1-year lockdown')
# plt.tight_layout()
# # plt.savefig('Figures/rec_wane_timing240930/child_growth174_wane0p07.png',dpi=300)
# plt.show()

# fig, axes = plt.subplots(2,2,figsize=(6.5,6.5))
# axes[0,0].plot(result.t,np.sum(obs,axis=1)/pop_size, label='Observed cases')
# axes[0,0].set_title('Simulation (including burn-in)')
# axes[0,0].set_ylabel('Observed incidence')
# axes[0,1].plot(result.t,np.sum(obs,axis=1)/pop_size, label='Observed cases')
# axes[0,1].set_xlim(2*PERIOD/3,PERIOD)
# axes[0,1].set_ylim(0,1e-4)
# # axes[0,1].set_ylim(0,4e-4)
# axes[0,1].set_title('Detail after burn-in')
# axes[0,1].set_ylabel('Observed incidence')
# # axes[1,0].plot(result.t,pop_size, label='Population size')
# # axes[1,0].set_title('Total population')
# # axes[1,0].set_ylabel('Persons')

# # Infected persons in each age group
# for i in range(NAG):
#     axes[1,0].plot(result.t,obs[:,i]/np.sum(result.y[range(i,10*NAG,NAG),:],axis=0), label=AGE_GROUP_NAMES[i], color=viridis(i), alpha=0.5)
# axes[1,0].set_xlim(2*PERIOD/3,PERIOD)
# axes[1,0].set_ylim(0,5e-4)
# # axes[1,0].set_ylim(0,1e-3)
# axes[1,0].set_title('Incidence by age group')
# axes[1,0].set_ylabel('Observed incidence')

# # Population in each age group
# for i in range(NAG):
#     axes[1,1].plot(result.t,np.sum(result.y[range(i,10*NAG,NAG),:],axis=0), label=AGE_GROUP_NAMES[i], color=viridis(i))
# axes[1,1].set_title('Age groups')
# axes[1,1].set_ylabel('Age group population')


# plt.tight_layout()
# # plt.savefig('Figures/SIR3_rotalike_demo_yearly.png')
# plt.show()