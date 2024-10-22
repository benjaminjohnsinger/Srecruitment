## Clustering simulation results
## BJS Oct 2024

import numpy as np
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
from plotting import lockdown_incidence_plot, lockdown_incidence_format, param_line_plot

def cluster_sims(results,obses,T_LOCKDOWN,n_clusters,pre=True,width=5,scaled=False,metric='euclidean'):
    obs_cut_and_scaled = {}
    for o_key,obs in obses.items():
        obs_temp = obs[(results[o_key].t>=T_LOCKDOWN-width*12) & (results[o_key].t<T_LOCKDOWN+(1-pre)*(width+1)*12)]
        if scaled:
            obs_temp = obs_temp/np.max(obs_temp)
        obs_cut_and_scaled[o_key] = obs_temp
    model = TimeSeriesKMeans(n_clusters=n_clusters,metric=metric,max_iter=10,random_state=241015)
    model.fit(np.array([obs_cut_and_scaled[key] for key in obs_cut_and_scaled.keys()]))
    return model

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
        lockdown_incidence_format(axes[i,0],T_LOCKDOWN,LOCKDOWN_DURATION,mx,title='',year_skip=2)
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