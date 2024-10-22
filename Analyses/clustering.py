## Clustering simulation results
## BJS Oct 2024

import numpy as np
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
from plotting import infections_by_age

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

def cluster_age_infect(results,params,model,T_LOCKDOWN,N=25):
    n_clusters = model.n_clusters
    result_cluster_labels = model.labels_
    cluster_mean_age_infect = {}
    for i in range(n_clusters):
        age_infect = np.zeros((sum(result_cluster_labels==i),params['NAG']))
        aidx = 0
        for j,key in enumerate(results.keys()):
            if result_cluster_labels[j]==i:
                params_n = params.copy()
                params_n['BETA'] = params['BETA']*(1+(key[0]/N-1/2))
                params_n['WANE'] = params['WANE']*(1+(key[1]/N-1/2))
                vec_len = len(params['S_REL'])
                vec = np.array([(1-j*(key[2]/(N*(vec_len-1)))) for j in range(vec_len)])
                vec = vec.reshape(params['S_REL'].shape)
                params_n['S_REL'] = vec
                result = results[key]
                i_by_age = infections_by_age(result,params_n)
                age_infect[aidx] = np.mean(i_by_age[(result.t>=T_LOCKDOWN-5*12) & (result.t<T_LOCKDOWN)],axis=0)
                aidx += 1
        cluster_mean_age_infect[i] = np.mean(age_infect,axis=0)
    return cluster_mean_age_infect