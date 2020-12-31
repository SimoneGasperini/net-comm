import sys
sys.path.append("..")

from model.unet import UndirectedNetwork
from model.erdosrenyi import ErdosRenyi

file = "../data/HEP_Collaboration.txt"

real_network = UndirectedNetwork.fromfile(file)

part = real_network.clustering()
mod = real_network.modularity(part)
print(f"\nmod = {mod}", flush=True)

n = real_network.number_of_nodes
m = real_network.number_of_edges
p = (2. * m) / n**2

random_network = ErdosRenyi(n, p)

part_er = random_network.clustering()
mod_er = random_network.modularity(part_er)
print(f"\nmod_random = {mod_er}", flush=True)

#%%
# using networkx
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity

netx = nx.read_edgelist(file)

partx = greedy_modularity_communities(netx)
modx = modularity(G=netx, communities=partx)
print(f"\nmod_nx = {modx}")

random_networkx = erdos_renyi_graph(n=n, p=p)

partx_er = greedy_modularity_communities(random_networkx)
modx_er = modularity(G=random_networkx, communities=partx_er)
print(f"\nmod_nx_random = {modx_er}")

#%%

# data plotting

import pandas as pd
import seaborn as sns
import pylab as plt

data = {'Networks': ['Real network', 'Random', 'Real Network  ', 'Random '],
        'Modularity': [mod, mod_er, modx, modx_er],
        'Class': ['Our Algorithm', 'Our Algorithm', 'NetworkX', 'NetworkX']
        }

df = pd.DataFrame(data=data)

g = sns.catplot(x='Networks', y='Modularity', kind='bar', hue="Class", dodge=False, data=df, height=5)
plt.show()
