## Clustering simulation results
## BJS Oct 2024

import numpy as np
import pickle
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
import time
from plotting import lockdown_incidence_plot, lockdown_incidence_format

with open('Data/Processed/SIS_6D.pickle','rb') as f:
    results = pickle.load(f)

with open('Data/Processed/SIS_6D_obs.pickle','rb') as f:
    obses = pickle.load(f)

n_clusters = 10
# model = TimeSeriesKMeans(n_clusters=n_clusters,metric='euclidean',max_iter=10,random_state=241015)
# # np.set_printoptions(threshold=np.inf)
# # print(results[(2,2,2,2,2,2)].t%1)
# start = time.time()
# model.fit([obses[key][(results[key].t>32*12) & (results[key].t<=43*12) & (results[key].t%1 < 1e-8)] for key in results.keys()])
# print(time.time()-start)
# print(model.cluster_centers_.shape)
# print(model.labels_)
# print(model.inertia_)

# with open('Data/Processed/SIS_6D_clusters_euclidean.pickle','wb') as f:
#     pickle.dump(model,f)

with open('Data/Processed/SIS_6D_clusters_euclidean.pickle','rb') as f:
    model = pickle.load(f)

T_LOCKDOWN = 37*12
LOCKDOWN_DURATION = 12

# times = np.arange(32*12+2,43*12+1,1)
# incidence plot for each cluster
fig, ax = plt.subplots(n_clusters,1,figsize=(6.5,1.7*n_clusters),sharex=True)
for i in range(n_clusters):
    idx = np.where(model.labels_==i)[0]
    mx = 0
    for j in idx:
        if i == 3:
            print(list(results.keys())[j])
        result = results[list(results.keys())[j]]
        obs = obses[list(obses.keys())[j]]
        mx_temp = lockdown_incidence_plot(ax[i],None,None,None,None,None,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,color='gray',alpha=0.01,obs=obs)
        mx = max(mx,mx_temp)
    lockdown_incidence_plot(ax[i],None,None,None,None,None,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,color='red',obs=obses[list(obses.keys())[idx[len(idx)//2]]])
    # lockdown_incidence_plot(ax[i],None,None,None,None,None,T_LOCKDOWN,LOCKDOWN_DURATION,color='red',result=result,times=times,obs=model.cluster_centers_[i])
    lockdown_incidence_format(ax[i],T_LOCKDOWN,LOCKDOWN_DURATION,mx,title='')
    ax[i].set_ylabel(f"Cluster {i+1}")
plt.tight_layout()
plt.savefig('Figures/SIS_6D_'+str(n_clusters)+'clusters_euclidean.png',dpi=300)

