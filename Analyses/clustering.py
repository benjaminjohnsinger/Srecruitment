## Clustering simulation results
## BJS Oct 2024

import numpy as np
import pickle
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
import time
from plotting import lockdown_incidence_plot, lockdown_incidence_format, param_line_plot

with open('Data/Processed/SIS_3D.pickle','rb') as f:
    results = pickle.load(f)

with open('Data/Processed/SIS_3D_obs.pickle','rb') as f:
    obses = pickle.load(f)

T_LOCKDOWN = 37*12
obs_cut_and_scaled = {}
for o_key,obs in obses.items():
    obs_temp = obs[(results[o_key].t>=T_LOCKDOWN-5*12) & (results[o_key].t<T_LOCKDOWN)]
    # obs_temp = obs_temp/np.max(obs_temp)
    obs_cut_and_scaled[o_key] = obs_temp

n_clusters = 6
model = TimeSeriesKMeans(n_clusters=n_clusters,metric='euclidean',max_iter=10,random_state=241015)
start = time.time()
model.fit(np.array([obs_cut_and_scaled[key] for key in obs_cut_and_scaled.keys()]))
print(time.time()-start)
print(model.cluster_centers_.shape)
print(model.labels_)
print(model.inertia_)
print(model.n_clusters)

n_clusters1 = 1
model1 = TimeSeriesKMeans(n_clusters=n_clusters1,metric='euclidean',max_iter=10,random_state=241015)
start = time.time()
model1.fit(np.array([obs_cut_and_scaled[key] for key in obs_cut_and_scaled.keys()]))
print(time.time()-start)

with open('Data/Processed/SIS_3D_'+str(n_clusters)+'preclusters_nonscaled_euclidean.pickle','wb') as f:
    pickle.dump(model,f)

# with open('Data/Processed/SIS_3D_preclusters_euclidean.pickle','rb') as f:
#     model = pickle.load(f)


T_LOCKDOWN = 37*12
LOCKDOWN_DURATION = 12

times = np.arange(32*12,37*12,1)
def incidence_and_box_plot():
    fig, ax = plt.subplots(n_clusters,2,figsize=(6.5,1.7*n_clusters),sharex='col')
    for i in range(n_clusters):
        idx = np.where(model.labels_==i)[0]
        mx = 0
        param_values = np.zeros((len(idx),3))
        for n_j,j in enumerate(idx):
            result = results[list(results.keys())[j]]
            obs = obses[list(obs_cut_and_scaled.keys())[j]]
            p_n = list(results.keys())[j]
            param_values[n_j,0] = (1+(p_n[0]/25-1/2))
            param_values[n_j,1] = (1+(p_n[1]/25-1/2))
            param_values[n_j,2] = p_n[2]/50
            # ax[i,0].plot(times,obs,color='black',alpha=0.1)
            mxs = lockdown_incidence_plot(ax[i,0],None,None,None,None,None,T_LOCKDOWN,LOCKDOWN_DURATION,color='black',result=result,obs=obs,alpha=0.01,relative=False)
            mx = max(mx,mxs)
        ax[i,0].plot(times,model.cluster_centers_[i],color='red',label='Cluster center')
        # lockdown_incidence_plot(ax[i],None,None,None,None,None,T_LOCKDOWN,LOCKDOWN_DURATION,color='red',result=result,times=times,obs=model.cluster_centers_[i])
        lockdown_incidence_format(ax[i,0],T_LOCKDOWN,LOCKDOWN_DURATION,mx,title='')
        ax[i,0].set_ylabel(f"Cluster {i+1}")
        # ax[i,0].set_ylim(0,2)
        ax[i,1].boxplot(param_values)
        ax[i,1].axhline(1/2,color='black',linestyle='--',alpha=0.3)
        # ax[i,1].axhline(3/2,color='black',linestyle='--',alpha=0.3)
        ax[i,1].set_xticks([1,2,3])
        ax[i,1].set_xticklabels(["Infectiousness","Waning","Acquired\nimmunity"])
        ax[i,1].set_yticks([0,0.5,1,1.5])
        ax[i,1].yaxis.set_label_position("right")
        ax[i,1].yaxis.tick_right()
        ax[i,1].set_ylabel("Relative value")
    plt.tight_layout()

# incidence_and_box_plot()
# plt.savefig('Figures/SIS_3D_'+str(n_clusters)+'_preclusters_nonscaled_euclidean.png',dpi=300)

from SISn_single_pathogen import params, OBS_AGE
def cluster_param_lines():
    fig, ax = plt.subplots(n_clusters,3,figsize=(6.5,1.7*n_clusters),sharey='row',sharex='col')
    for i in range(n_clusters):
        idx = np.where(model.labels_==i)[0]
        param_values = np.zeros((len(idx),3))
        values = np.zeros(len(idx))
        mx = 0
        for n_j,j in enumerate(idx):
            p_n = list(results.keys())[j]
            param_values[n_j,0] = 30*(1+(p_n[0]/25-1/2))
            param_values[n_j,1] = (1/12)*(1+(p_n[1]/25-1/2))
            param_values[n_j,2] = p_n[2]/50
            obs = obses[list(obs_cut_and_scaled.keys())[j]]
            result = results[list(results.keys())[j]]
            values[n_j] = np.max(obs[result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION)])
            # post_peak_arg = np.argmax(obs[result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION)] > np.max(obs[(result.t>T_LOCKDOWN-12*12) & (result.t<T_LOCKDOWN)])/2)
            # val = result.t[np.argmax(result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION))+post_peak_arg]-(T_LOCKDOWN+LOCKDOWN_DURATION)
            # values[n_j] = min(val/12,5)
            # norm_param_values = param_values/np.max(param_values,axis=0)
        for n_fp,focal_param in enumerate(['BETA','WANE','S_REL']):
            param = param_values[:,n_fp]
            # cs = norm_param_values*0.9
            # ax[i,n_fp].scatter(param,values,c=cs,alpha=0.3,s=10)
            median_values = np.zeros(len(param))
            lower_Qs = np.zeros(len(param))
            upper_Qs = np.zeros(len(param))
            # for each value of param, take median over all other params
            for n_p in range(len(param)):
                median_values[n_p] = np.median(values[param==param[n_p]])
                lower_Qs[n_p] = np.percentile(values[param==param[n_p]],25)
                upper_Qs[n_p] = np.percentile(values[param==param[n_p]],75)
            # sort by parameter value
            sort_args = np.argsort(param)
            param = param[sort_args]
            median_values = median_values[sort_args]
            lower_Qs = lower_Qs[sort_args]
            upper_Qs = upper_Qs[sort_args]
            ax[i,n_fp].plot(param,median_values,color="black")
            ax[i,n_fp].fill_between(param,lower_Qs,upper_Qs,alpha=0.3,color="black")
            ax[i,n_fp].set_xlabel(f"{focal_param}")
        # if mx > 6:
        #     ax[i,0].set_yscale('log')


def cluster_plot(axes,results,obses,model,relative=False,color=False,line=True,clusters=None,
parameters=["BETA","WANE","S_REL"],param_labels=["Infectiousness","Waning","Acquired immunity"],
grid_mode=["scale","scale","fade_vec"],base_values=[30,1/12,1/2],factors=[1,1,1],N=25,
y_value=("time to rebound"),y_label="Time to rebound",
T_LOCKDOWN=37*12,LOCKDOWN_DURATION=12):
    n_clusters = model.n_clusters
    if clusters is None:
        clusters = np.arange(n_clusters)
    times = np.arange(T_LOCKDOWN-5*12,T_LOCKDOWN,1)
    for i,cluster in enumerate(clusters):
        idx = np.where(model.labels_==cluster)[0]
        param_values = np.zeros((len(idx),len(parameters)))
        values = np.zeros(len(idx))
        mx = 0
        for n_j,j in enumerate(idx):
            result = results[list(results.keys())[j]]
            obs = 100*obses[list(obses.keys())[j]]
            p_n = list(results.keys())[j]
            if y_value == "rebound peak incidence":
                values[n_j] = np.max(obs[result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION)])
            elif y_value == "time to rebound":
                post_peak_arg = np.argmax(obs[result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION)] > np.max(obs[(result.t>T_LOCKDOWN-5*12) & (result.t<T_LOCKDOWN)])/2)
                val = result.t[np.argmax(result.t>=(T_LOCKDOWN+LOCKDOWN_DURATION))+post_peak_arg]-(T_LOCKDOWN+LOCKDOWN_DURATION)
                values[n_j] = min(val/12,5)
            for n_p,param in enumerate(parameters):
                if grid_mode[n_p] == "scale":
                    param_values[n_j,n_p] = base_values[n_p]*(1+(p_n[n_p]/N-1/2))**factors[n_p]
                elif grid_mode[n_p] == "fade_vec":
                    param_values[n_j,n_p] = p_n[n_p]/(2*N)
            if color:
                norm_param_values = param_values/np.max(param_values,axis=0)
                mxs = lockdown_incidence_plot(axes[i,0],None,None,None,None,None,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,obs=obs,relative=relative,
                color=norm_param_values[n_j]*0.95,alpha=0.005)
            else:
                mxs = lockdown_incidence_plot(axes[i,0],None,None,None,None,None,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,obs=obs,relative=relative,
                color='black',alpha=0.01)
            mx = max(mx,mxs)
        if color:
            axes[i,0].plot(times,100*model.cluster_centers_[cluster],color='black',label='Cluster center')
        else:
            axes[i,0].plot(times,100*model.cluster_centers_[cluster],color='red',label='Cluster center')
        lockdown_incidence_format(axes[i,0],T_LOCKDOWN,LOCKDOWN_DURATION,mx,title='')
        axes[i,0].set_xlabel("")
        for n_p,param in enumerate(parameters):
            if color:
                axes[i,n_p+1].scatter(param_values[:,n_p],values,c=norm_param_values*0.95,alpha=0.3,s=10)
            elif not line:
                axes[i,n_p+1].scatter(param_values[:,n_p],values,color="black",alpha=0.3,s=10)
            else:
                pf = param_values[:,n_p]
                Qs = np.zeros((len(pf),3))
                for pidx in range(len(pf)):
                    Qs[pidx,:] = np.percentile(values[pf==pf[pidx]],[25,50,75])
                sort_args = np.argsort(pf)
                param_sorted = pf[sort_args]
                Qs_sorted = Qs[sort_args,:]
                axes[i,n_p+1].plot(param_sorted,Qs_sorted[:,1],color="black")
                axes[i,n_p+1].fill_between(param_sorted,Qs_sorted[:,0],Qs_sorted[:,2],alpha=0.3,color="black")
            if n_p > 0:
                axes[i,n_p+1].set_yticklabels([])
        if n_clusters > 1:
            axes[i,0].set_ylabel(f"Cluster {cluster+1}\n\nIncidence")
        else:
            axes[i,0].set_ylabel(f"All simulations\n\nIncidence")
        axes[i,1].set_ylabel("\n"+y_label)
    if n_clusters > 1:
        axes[len(clusters)-1,0].set_xlabel("Time (years)")
        for n_p,label in enumerate(param_labels):
            axes[len(clusters)-1,n_p+1].set_xlabel(f"{label}")

fig, axes = plt.subplots(4,4,figsize=(6.5,1.7*4),sharex='col',layout='constrained',squeeze=False)

for row in range(4):
    axes[row,1].sharey(axes[row,2])
    axes[row,2].sharey(axes[row,3])

first_row = axes[0,:]
first_row.shape = (1,4)

cluster_plot(first_row,results,obses,model1,color=False)
cluster_plot(axes[1:,:],results,obses,model,color=False,clusters=[0,2,4])
fig.align_ylabels()
plt.savefig('Figures/SIS_3D_'+str(n_clusters)+'_clusters_test_1plus3of6.png',dpi=900)