## Clustering simulation results
## BJS Oct 2024

import numpy as np
import pickle
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt

with open('Data/Processed/SIS_6D.pickle','rb') as f:
    results = pickle.load(f)

result = results[(2,2,2,2,2,2)]
print(result.y[:,(result.t>34*12) & (result.t<=42*12)].shape)
plt.plot(result.y[:,(result.t>34*12) & (result.t<=42*12)].T)
plt.show()

# model = TimeSeriesKMeans(n_clusters=5,metric='dtw',max_iter=10)
# model.fit([result.y[(result.t>38*12) & (result.t<=42*12)] for result in results.values()])
# print(model.cluster_centers_.shape)
# print(model.labels_)
# print(model.inertia_)
# print(model.score(results.values()))
# print(model.predict(results.values()))

# # incidence plot for each cluster
# fig, ax = plt.subplots(5,1,figsize=(6.5,8.5))
# for i in range(5):
#     idx = np.where(model.labels_==i)[0]
#     for j in idx:
#         result = results[j]
#         incidence_plot(ax[i],result,params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,result=result,color='gray',alpha=0.1)
#     incidence_plot(ax[i],model.cluster_centers_[i],params,OBS_AGE,PERIOD,POINTS,T_LOCKDOWN,LOCKDOWN_DURATION,color='red')
#     incidence_format(ax[i],POINTS,T_LOCKDOWN,LOCKDOWN_DURATION)
#     ax[i].set_ylabel(f"Cluster {i+1}")
# plt.tight_layout()
# plt.savefig('Figures/SIS_6D_clusters_post.png',dpi=300)

