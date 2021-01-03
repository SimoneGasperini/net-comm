import numpy as np
import pylab as plt

import sys
sys.path.append("..")

from model.erdosrenyi import ErdosRenyi
from model.composite_network import CompositeNetwork


def natural_partition(blocks_sizes):
    n = np.append(np.array([0], dtype=int), np.cumsum(blocks_sizes))
    return [set(range(n[i], n[i+1])) for i in range(len(n)-1)]


'''
##############################################################################

We consider a series of networks composed by a fixed number of blocks with
random edges between them. Each block is a complete graph with a fixed mean
number of nodes (and links).
The blocks sizes heterogeneity is the only variable parameter: in particular,
we sample the nodes number of each block from a wider and wider uniform
distribution, progressively increasing the sizes heterogeneity of the blocks.

We run the clustering algorithm on this model to verify that the success of
the communities detection does not depend on the blocks sizes heterogeneity.
Moreover, we observe that the modularity computed both on the natural and
detected partition is slowly decreasing as the heterogeneity increases.

##############################################################################
'''

simulations = 30
blocks = 10
nodes = 1000
mu_n = nodes / blocks

edge_matrix = np.empty(shape=(blocks, blocks), dtype=int)
for i in range(blocks):
    for j in range(i, blocks):
        edge_matrix[i][j] = edge_matrix[j][i] = 0 if i == j else np.random.randint(1,7)

nodes_distributions = []

mod1 = np.empty(simulations)
mod2 = np.empty(simulations)

N = np.empty(simulations)

for i in range(1, simulations+1):

    diff = mu_n * 0.033 * i
    a = int(mu_n - diff)
    b = int(mu_n + diff)
    blocks_sizes = np.random.randint(a, b+1, size=blocks)
    nodes_distributions.append(blocks_sizes)

    network = CompositeNetwork([ErdosRenyi(n=n, p=1) for n in blocks_sizes], edge_matrix)

    part = natural_partition(blocks_sizes)
    mod1[i-1] = network.modularity(part)

    part = network.clustering(check_result=True)
    mod2[i-1] = network.modularity(part)

    N[i-1] = network.number_of_nodes


#%%
plt.style.use('seaborn-paper')


fig, ax = plt.subplots(figsize=(8,6))

ax.axhline(y=mu_n, color='black', linestyle='--', label='mean')
ax.violinplot(nodes_distributions, positions=np.arange(simulations))

ax.set_xlabel('simulation', fontsize=14)
ax.set_ylabel('sizes of blocks', fontsize=14)
ax.legend(fontsize=12, loc='lower left')

for tx in ax.xaxis.get_major_ticks():
    tx.label.set_fontsize(12)

for ty in ax.yaxis.get_major_ticks():
    ty.label.set_fontsize(12)

plt.show()


#%%
fig.savefig('../images/sizes_distributions.pdf', bbox_inches='tight', dpi=1200)


#%%
plt.style.use('seaborn-paper')


fig, ax = plt.subplots(figsize=(8,6))

mean_M = (mu_n * (mu_n - 1) / 2) * blocks + np.sum(edge_matrix) / 2
f = lambda x, y : 1 - x / y - 1 / x
ax.axhline(y=f(blocks, mean_M), color='black', linestyle='--', label='homogeneous')
ax.scatter(np.arange(simulations), mod1, marker='+', s=50, color='green', label='natural partition')
ax.scatter(np.arange(simulations), mod2, marker='o', s=20, color='red', label='detected partition')

ax.set_ylim(bottom=0., top=1.)
ax.set_xlabel('simulation', fontsize=14)
ax.set_ylabel('modularity', fontsize=14)
ax.legend(fontsize=12, loc='lower left')

for tx in ax.xaxis.get_major_ticks():
    tx.label.set_fontsize(12)

for ty in ax.yaxis.get_major_ticks():
    ty.label.set_fontsize(12)

plt.grid()
plt.show()


#%%
fig.savefig('../images/blocks_heterogeneity.pdf', bbox_inches='tight', dpi=1200)
