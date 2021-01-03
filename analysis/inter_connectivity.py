import numpy as np
import pylab as plt

import sys
sys.path.append("..")

from model.erdosrenyi import ErdosRenyi
from model.composite_network import CompositeNetwork


def natural_partition(comms, nodes):
    n = np.cumsum([nodes for _ in range(comms)])
    return [set(range(n1, n2)) for n1, n2 in zip(n-nodes, n)]


def run_simulation(inter_edges, points, blocks, n, p):

    mod1 = np.empty(points)
    mod2 = np.empty(points)

    for k in range(points):

        edge_matrix = np.empty(shape=(blocks, blocks), dtype=int)
        for i in range(blocks):
            for j in range(i, blocks):
                edge_matrix[i][j] = edge_matrix[j][i] = 0 if i == j else inter_edges[k]

        network = CompositeNetwork([ErdosRenyi(n, p) for _ in range(blocks)], edge_matrix)

        part1 = natural_partition(comms=blocks, nodes=n)
        mod1[k] = network.modularity(part1)

        part2 = network.clustering(check_result=True)
        mod2[k] = network.modularity(part2)

    return mod1, mod2


points = 100
blocks = 5
n = 100
probabilities = [0.1, 0.15, 0.2]

inter_edges1 = np.linspace(1, 1200, points).astype(int)

mod_list = []

for p in probabilities:

    print(f'\nSTART SIMULATION --> intra-blocks edge probability = {p}', flush=True)

    natural_mod, detected_mod = run_simulation(inter_edges1, points, blocks, n, p)
    mod_list.append(detected_mod)


inter_edges2 = np.linspace(1, 300, points).astype(int)

mod1_list = []
mod2_list = []

for p in probabilities:

    print(f'\nSTART SIMULATION --> intra-blocks edge probability = {p}', flush=True)

    mod1, mod2 = run_simulation(inter_edges2, points, blocks, n, p)
    mod1_list.append(mod1)
    mod2_list.append(mod2)


#%%
plt.style.use('seaborn-paper')


fig, ax = plt.subplots(figsize=(8,6))

cols1 = ['red', 'blue', 'green']
cols2 = ['#ffb33b', '#2b8eff', '#44e000']
labs = [f'p = {p}' for p in probabilities]

for i in range(len(probabilities)):

    ax.scatter(inter_edges1, mod_list[i], s=20, color=cols2[i], label=labs[i])

ax.set_ylim(bottom=0.0, top=1.0)
ax.set_xlabel('inter-blocks edges', fontsize=14)
ax.set_ylabel('modularity', fontsize=14)
ax.legend(fontsize=12, loc='upper right')

for tx in ax.xaxis.get_major_ticks():
    tx.label.set_fontsize(12)

for ty in ax.yaxis.get_major_ticks():
    ty.label.set_fontsize(12)

plt.grid()
plt.show()

#%%
fig.savefig('../images/inter_connectivity1.pdf', bbox_inches='tight', dpi=1200)


#%%
plt.style.use('seaborn-paper')


fig, ax = plt.subplots(figsize=(8,6))

for j in range(len(probabilities)):

    ax.plot(inter_edges2, mod1_list[j], color=cols1[j], label=labs[j])
    ax.scatter(inter_edges2, mod2_list[j], s=20, color=cols2[j], label=labs[j])

ax.set_ylim(bottom=0.0, top=1.0)
ax.set_xlabel('inter-blocks edges', fontsize=14)
ax.set_ylabel('modularity', fontsize=14)

ax.legend(fontsize=12, loc='upper right',  ncol=2,
          title='natural partition - detected partition', title_fontsize=12)

for tx in ax.xaxis.get_major_ticks():
    tx.label.set_fontsize(12)

for ty in ax.yaxis.get_major_ticks():
    ty.label.set_fontsize(12)

ax.grid()
plt.show()

#%%
fig.savefig('../images/inter_connectivity2.pdf', bbox_inches='tight', dpi=1200)
