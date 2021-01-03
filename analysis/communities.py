import pylab as plt

import sys
sys.path.append("..")

from model.unet import UndirectedNetwork
from model.visualization import draw_communities_graph, draw_communities_barplot


#file = '../data/deezer.txt'
#file = '../data/facebook.txt'
file = '../data/General_Relativity.txt'

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
plt.style.use('seaborn-paper')


fig, ax = plt.subplots(figsize=(8,8))
draw_communities_barplot(unet=network, partition=part, ax=ax, min_size=15, scale='log')

plt.show()
