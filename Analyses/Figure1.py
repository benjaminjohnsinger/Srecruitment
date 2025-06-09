import jax.numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.gridspec import GridSpec
from clustering import cluster_sims
from plotting import cluster_plot
from plotting import age_of_first_infection
from utils import *

from Parameters.census_population import MEDIAN_AGE


with open('Data/Processed/SIS_3D_power.pickle','rb') as f:
    results = pickle.load(f)
with open('Data/Processed/SIS_3D_power_obs.pickle','rb') as f:
    obses = pickle.load(f)

N = max([max(key) for key in results.keys()])+1

# print(results.values())

T_LOCKDOWN = date_to_t('2014-01-01')
LOCKDOWN_DURATION = 365

#### complex figure
# Gridspec for row of five axes across top of figure, with two 3x4 panels below
fig = plt.figure(figsize=(14,10.5),layout='constrained')
gs = GridSpec(5,2,figure=fig)
# top row of five axes
gs_top = gs[0:2,:].subgridspec(2,4)
big_axis = fig.add_subplot(gs_top[:,0])
big_axis_hidden = big_axis.twinx()
big_axis_hidden.set_visible(False)
top_row_up = np.array([big_axis_hidden] + [fig.add_subplot(gs_top[0,j]) for j in range(1,4)])
top_row_down = np.array([big_axis] + [fig.add_subplot(gs_top[1,j]) for j in range(1,4)])
top_row_up.shape = (1,4)
top_row_down.shape = (1,4)
# Two 3x4 panels below
gs_panel1 = gs[2:5,0].subgridspec(3,4)
panel1_axes = np.array([[fig.add_subplot(gs_panel1[i,j]) for j in range(4)] for i in range(3)])
gs_panel2 = gs[2:5,1].subgridspec(3,4)
panel2_axes = np.array([[fig.add_subplot(gs_panel2[i,j]) for j in range(4)] for i in range(3)])

top_row_up[0,1].sharey(top_row_up[0,2])
top_row_up[0,2].sharey(top_row_up[0,3])
top_row_down[0,1].sharey(top_row_down[0,2])
top_row_down[0,2].sharey(top_row_down[0,3])
for row in range(3):
    panel1_axes[row,1].sharey(panel1_axes[row,2])
    panel1_axes[row,2].sharey(panel1_axes[row,3])
    panel2_axes[row,1].sharey(panel2_axes[row,2])
    panel2_axes[row,2].sharey(panel2_axes[row,3])
for col in range(4):
    panel1_axes[2,col].sharex(panel1_axes[1,col])
    panel1_axes[1,col].sharex(panel1_axes[0,col])
    panel2_axes[2,col].sharex(panel2_axes[1,col])
    panel2_axes[1,col].sharex(panel2_axes[0,col])

model = cluster_sims(results,obses,T_LOCKDOWN,3)
# with open('Data/Processed/SIS_3D_based_cluster6_scaled.pickle','wb') as f:
#     pickle.dump(model,f)
# with open('Data/Processed/SIS_3D_2_cluster6_no_cap.pickle','rb') as f:
#     model = pickle.load(f)
# with open('Data/Processed/SIS_3D_based_5clusters_of_cluster1of3.pickle','rb') as f:
#     model_5of1 = pickle.load(f)
## caculate five clusters of cluster 1, first by filtering cluster 1 out of results and then doing clustering
results_5of1 = {}
obses_5of1 = {}
for i in range(len(results.keys())):
    key = list(results.keys())[i]
    if model.labels_[i] == 1:
        results_5of1[key] = results[key]
        obses_5of1[key] = obses[key]
model_5of1 = cluster_sims(results_5of1,obses_5of1,T_LOCKDOWN,5)
# relabel cluster 1 in model
model.labels_[model.labels_==0] = 5
model.labels_[model.labels_==2] = 5
model.labels_[model.labels_==1] = model_5of1.labels_
model.n_clusters = 6

model1 = cluster_sims(results,obses,T_LOCKDOWN,1)
ages = np.array([np.mean(age_of_first_infection(results[key],MEDIAN_AGE)[np.argmax(results[key].t>=T_LOCKDOWN-5*365):np.argmax(results[key].t>T_LOCKDOWN)]) for key in results.keys()])
ages_label = np.array([int(age>3) + int(age>12) + int(age>5*12) + int(age>18*12) + int(age>40*12) + int(age>65*12) for age in ages])
n_age_clusters = max(ages_label)+1
cluster_colors = np.array([["#FF832B", "#FFB000", "#DC267F", "#648FFF", "#BBBBBB", "#785EF0", "#8B0000", "#00FF00"][i] for i in model.labels_])

cluster_plot(top_row_up,results,obses,model1.n_clusters,model1.labels_,None,color=False,N=N)
cluster_plot(top_row_down,results,obses,model1.n_clusters,model1.labels_,None,color=True,line=False,color_values_all=cluster_colors,N=N)
cluster_plot(panel1_axes,results,obses,model.n_clusters,model.labels_,None,color=True,line=True,color_values_all=cluster_colors,clusters=[4,0,3],N=N)
cluster_plot(panel2_axes,results,obses,n_age_clusters,ages_label,None,color=False,clusters=[0,1,2],N=N)

top_row_down[0,0].set_xlabel("Time (years)\n")
top_row_down[0,0].set_ylabel("All simulations\n\nIncidence")
top_row_up[0,1].set_ylabel("")
top_row_down[0,1].set_ylabel("\n                            Time to rebound (years)")
top_row_down[0,1].set_xlabel("Transmissibility\n")
top_row_down[0,2].set_xlabel("Immune waning rate\n")
top_row_down[0,3].set_xlabel("Acquired immunity\n")
top_row_up[0,1].set_ylim(top_row_down[0,1].get_ylim())
# panel1_axes[1,1].set_ylim([-0.1,2.1])
# panel1_axes[2,3].set_xlim([-0.01,0.51])
# panel2_axes[2,3].set_xlim([-0.01,0.51])
for col in range(1,4):
    top_row_up[0,col].set_xticklabels([])
for axes in [panel1_axes,panel2_axes]:
    for col in range(4):
        axes[0,col].set_xticklabels([])
    axes[2,0].set_xlabel("Time")
    axes[2,1].set_xlabel("Transm.")
    axes[2,3].set_xlabel("Immunity")
age_names = ["Newborn","Infant","Childhood"]
for row in range(3):
    panel1_axes[row,0].set_ylabel("Cluster "+str(row+1))
    panel1_axes[row,1].set_ylabel("")
    panel2_axes[row,0].set_ylabel("\n"+age_names[row]+"\ndiseases")
    panel2_axes[row,1].set_ylabel("")
panel1_axes[1,1].set_ylabel("Time to rebound (years)")
panel2_axes[1,1].set_ylabel("Time to rebound (years)")

plt.savefig('Figures/SIS_3D_power_complex_clusters_split1of3in5_poster_test.png',dpi=500)