import sys
sys.path.append("..")

from model.unet import UndirectedNetwork
from model.randomnet import RandomNetwork


file = '../data/HEP_Collaboration.txt'

real_network = UndirectedNetwork.fromfile(file)

part = real_network.clustering()
mod_real = real_network.modularity(part)
print(f'\nmod_real = {mod_real}\n', flush=True)

n = real_network.number_of_nodes
m = real_network.number_of_edges

random_network = RandomNetwork(n, m)

part = random_network.clustering()
mod_rand = random_network.modularity(part)
print(f'\nmod_random = {mod_rand}\n', flush=True)


#%%
# using networkx

from networkx.algorithms.bipartite.edgelist import read_edgelist
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity

netx = read_edgelist(file)

partx = greedy_modularity_communities(netx)
modx_real = modularity(G=netx, communities=partx)
print(f'\nmod_real_nx = {modx_real}')

p = (2. * m) / n**2
netx_er = erdos_renyi_graph(n=n, p=p)

partx = greedy_modularity_communities(netx_er)
modx_rand = modularity(G=netx_er, communities=partx)
print(f'\nmod_random_nx = {modx_rand}')


#%%
# data plotting

import pandas as pd
import seaborn as sns
import pylab as plt

data = {'Networks': ['Real network', 'Random', 'Real Network', 'Random'],
        'Modularity': [mod_real, mod_rand, modx_real, modx_rand],
        'Class': ['Our Algorithm', 'Our Algorithm', 'NetworkX', 'NetworkX']
        }

df = pd.DataFrame(data=data)

g = sns.catplot(x='Networks', y='Modularity', kind='bar', hue='Class', dodge=False, data=df, height=5)
plt.show()
