# import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
from matplotlib.gridspec import GridSpec
from clustering import cluster_sims
from plotting import cluster_plot
from plotting import age_of_first_infection

from Parameters.census_population import MEDIAN_AGE


with open('Data/Processed/SIS_3D_based.pickle','rb') as f:
    results = pickle.load(f)
with open('Data/Processed/SIS_3D_based_obs_full.pickle','rb') as f:
    obses = pickle.load(f)

N = max([max(key) for key in results.keys()])+1

T_LOCKDOWN = 37*12
LOCKDOWN_DURATION = 12

model = cluster_sims(results,obses,T_LOCKDOWN,6)
# with open('Data/Processed/SIS_3D_based_cluster6_scaled.pickle','wb') as f:
#     pickle.dump(model,f)
# with open('Data/Processed/SIS_3D_2_cluster6_no_cap.pickle','rb') as f:
#     model = pickle.load(f)
# with open('Data/Processed/SIS_3D_based_5clusters_of_cluster1of3.pickle','rb') as f:
#     model_5of1 = pickle.load(f)
# # relabel cluster 1 in model
# model.labels_[model.labels_==2] = 5
# model.labels_[model.labels_==1] = 5
# model.labels_[model.labels_==0] = model_5of1.labels_
# model.n_clusters = 6

model1 = cluster_sims(results,obses,T_LOCKDOWN,1)
ages = jnp.array([jnp.mean(age_of_first_infection(results[key],MEDIAN_AGE)[jnp.argmax(results[key].t>=T_LOCKDOWN-5*12):jnp.argmax(results[key].t>T_LOCKDOWN)]) for key in results.keys()])
ages_label = jnp.array([int(age>3) + int(age>12) + int(age>5*12) + int(age>18*12) + int(age>40*12) + int(age>65*12) for age in ages])
n_age_clusters = max(ages_label)+1
cluster_colors = jnp.array([["#FF832B", "#FFB000", "#DC267F", "#648FFF", "#BBBBBB", "#785EF0", "#8B0000", "#00FF00"][i] for i in model.labels_])

incidences = {}
for obs_key in obses.keys():
    incidences[obs_key] = jnp.sum(obses[obs_key],axis=1)/jnp.sum(results[obs_key].y,axis=0)

#### top row of complex figure
# Top row plot
fig = plt.figure(figsize=(10,5.6),layout='constrained')
gs = GridSpec(2,4,figure=fig)
# top row of five axes
big_axis = fig.add_subplot(gs[:,0])
big_axis_hidden = big_axis.twinx()
big_axis_hidden.set_visible(False)
top_row_up = jnp.array([big_axis_hidden] + [fig.add_subplot(gs[0,j]) for j in range(1,4)])
top_row_down = jnp.array([big_axis] + [fig.add_subplot(gs[1,j]) for j in range(1,4)])
top_row_up.shape = (1,4)
top_row_down.shape = (1,4)
top_row_up[0,1].sharey(top_row_up[0,2])
top_row_up[0,2].sharey(top_row_up[0,3])
top_row_down[0,1].sharey(top_row_down[0,2])
top_row_down[0,2].sharey(top_row_down[0,3])
for col in range(1,4):
    top_row_up[0,col].set_xticklabels([])

cluster_plot(top_row_up,results,incidences,model1.n_clusters,model1.labels_,None,color=False,N=N)
cluster_plot(top_row_down,results,incidences,model1.n_clusters,model1.labels_,None,color=True,line=False,color_values_all=cluster_colors,N=N)

top_row_down[0,0].set_xlabel("Time (years)\n")
top_row_down[0,0].set_ylabel("All simulations\n\nIncidence")
top_row_up[0,1].set_ylabel("")
top_row_down[0,1].set_ylabel("\n                                                        Time to rebound (years)")
top_row_down[0,1].set_xlabel("Transmissibility\n")
top_row_down[0,2].set_xlabel("Immune waning rate\n")
top_row_down[0,3].set_xlabel("Acquired immunity\n")
top_row_up[0,1].set_ylim(top_row_down[0,1].get_ylim())

plt.savefig('Figures/SIS_3D_based_complex_full_clusters_top_row_slide.png',dpi=500)

# 3x4 panels
fig, panel1_axes = plt.subplots(3,4,figsize=(10,5.6))

for row in range(3):
    panel1_axes[row,1].sharey(panel1_axes[row,2])
    panel1_axes[row,2].sharey(panel1_axes[row,3])
for col in range(4):
    panel1_axes[2,col].sharex(panel1_axes[1,col])
    panel1_axes[1,col].sharex(panel1_axes[0,col])

cluster_plot(panel1_axes,results,incidences,model.n_clusters,model.labels_,None,color=True,line=True,color_values_all=cluster_colors,clusters=[4,0,3],N=N)

for col in range(4):
    panel1_axes[0,col].set_xticklabels([])
panel1_axes[2,0].set_xlabel("Time")
panel1_axes[2,1].set_xlabel("Transm.")
panel1_axes[2,3].set_xlabel("Immunity")
age_names = ["Newborn","Infant","Childhood"]
for row in range(3):
    panel1_axes[row,0].set_ylabel("Cluster "+str(row+1))
    panel1_axes[row,1].set_ylabel("")
panel1_axes[1,1].set_ylabel("Time to rebound (years)")
plt.tight_layout()
plt.savefig('Figures/SIS_3D_based_complex_full_clusters_panel1_slide.png',dpi=500)

# 3x4 panels
fig, panel2_axes = plt.subplots(3,4,figsize=(10,5.6))

for row in range(3):
    panel2_axes[row,1].sharey(panel2_axes[row,2])
    panel2_axes[row,2].sharey(panel2_axes[row,3])
for col in range(4):
    panel2_axes[2,col].sharex(panel2_axes[1,col])
    panel2_axes[1,col].sharex(panel2_axes[0,col])

cluster_plot(panel2_axes,results,incidences,n_age_clusters,ages_label,None,color=False,clusters=[0,1,2],N=N)


for col in range(4):
    panel2_axes[0,col].set_xticklabels([])
panel2_axes[2,0].set_xlabel("Time")
panel2_axes[2,1].set_xlabel("Transm.")
panel2_axes[2,3].set_xlabel("Immunity")
age_names = ["Newborn","Infant","Childhood"]
for row in range(3):
    panel2_axes[row,0].set_ylabel("\n"+age_names[row]+"\ndiseases")
    panel2_axes[row,1].set_ylabel("")
panel2_axes[1,1].set_ylabel("Time to rebound (years)")
plt.tight_layout()
plt.savefig('Figures/SIS_3D_based_complex_full_clusters_panel2_slide.svg',transparent=True)