## Clustering simulation results
## BJS Oct 2024

import jax.numpy as np
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
from plotting import age_of_first_infection

def cluster_sims(results,obses,T_LOCKDOWN,n_clusters,pre=True,width=5,scaled=False,metric='euclidean'):
    obs_cut_and_scaled = {}
    for o_key,obs in obses.items():
        obs_temp = obs[(results[o_key].t>=T_LOCKDOWN-width*365) & (results[o_key].t<T_LOCKDOWN+(1-pre)*(width+1)*365)]
        if scaled:
            obs_temp = obs_temp/np.max(obs_temp)
        obs_cut_and_scaled[o_key] = obs_temp
    model = TimeSeriesKMeans(n_clusters=n_clusters,metric=metric,max_iter=10,random_state=241015)
    model.fit(np.array([obs_cut_and_scaled[key] for key in obs_cut_and_scaled.keys()]))
    return model

def cluster_first_infect(results,params,model,T_LOCKDOWN,MEDIAN_AGE):
    n_clusters = model.n_clusters
    result_cluster_labels = model.labels_
    cluster_mean_first_infect = {}
    for i in range(n_clusters):
        first_infect = np.zeros(sum(result_cluster_labels==i))
        aidx = 0
        for j,key in enumerate(results.keys()):
            if result_cluster_labels[j]==i:
                first_infect[aidx] = np.mean(age_of_first_infection(results[key],MEDIAN_AGE)[np.argmax(results[key].t>=T_LOCKDOWN-5*12):np.argmax(results[key].t>T_LOCKDOWN)])
                aidx += 1
        cluster_mean_first_infect[i] = np.mean(first_infect)/12
    return cluster_mean_first_infect