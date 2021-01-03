import pylab as plt

import sys
sys.path.append("..")

from model.unet import UndirectedNetwork
from model.visualization import draw_communities_graph, draw_communities_barplot


#name = 'deezer'
#name = 'facebook'
name = 'General_Relativity'

file = '../data/' + name + '.txt'

network = UndirectedNetwork.fromfile(file)

part = network.clustering()
comms = network.number_of_communities
mod = network.modularity(part)

print(f'number of communities = {comms}')
print(f'modularity = {mod}')


#%%
plt.style.use('seaborn-paper')


fig, ax = plt.subplots(figsize=(8,8))
draw_communities_graph(unet=network, partition=part, ax=ax, min_size=15, scale_size=1e6, scale='root')

plt.show()

#%%
fig.savefig('../images/communities_graph_' + name + '.pdf', bbox_inches='tight', dpi=1200)


#%%
plt.style.use('seaborn-paper')


fig, ax = plt.subplots(figsize=(8,6))
draw_communities_barplot(unet=network, partition=part, ax=ax, min_size=15, scale='log')

plt.show()

#%%
fig.savefig('../images/communities_barplot_' + name + '.pdf', bbox_inches='tight', dpi=1200)
