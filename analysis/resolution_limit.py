import numpy as np
import pylab as plt

import sys
sys.path.append("..")

from model.randomnet import RandomNetwork
from model.compositenet import CompositeNetwork


def natural_partition(comms, nodes):
    n = np.cumsum([nodes for _ in range(comms)])
    return [set(range(n1, n2)) for n1, n2 in zip(n-nodes, n)]

def resolution_limit(m, L):
    condition = m < np.sqrt(2 * L)
    return m[condition][-1]



# MODEL: ring of identical complete networks
comms = np.arange(2,60)
nodes = 5
edges = int(nodes * (nodes - 1) * 0.5)

mod1 = np.empty(comms.shape)
mod2 = np.empty(comms.shape)
tot_edges = np.empty(comms.shape)


for c in comms:

    unetworks = [RandomNetwork(n=nodes, m=edges, force_connected=True)
                 for _ in range(c)]

    edge_matrix = np.zeros(shape=(c, c), dtype=int)
    up_diag = np.array([(i, i+1) for i in range(c-1)])
    edge_matrix[up_diag[:,0], up_diag[:,1]] = 1
    edge_matrix[-1][0] = 1
    edge_matrix += edge_matrix.transpose()

    composite = CompositeNetwork(unetworks, edge_matrix)
    tot_edges[c-comms[0]] = composite.number_of_edges

    part = natural_partition(c, nodes)
    mod1[c-comms[0]] = composite.modularity(part)

    part = composite.clustering(check_result=True)
    mod2[c-comms[0]] = composite.modularity(part)



plt.style.use('seaborn-paper')

fig, ax = plt.subplots(figsize=(8,8))

# view resolution limit
rl  = resolution_limit(comms, tot_edges)
ax.axvline(x=rl, color='black', linestyle='--')
plt.text(rl+1, 0.51, 'RL', fontsize=16)

# plot modularity relative to natural partition
ax.scatter(comms, mod1, s=30, marker='^', color='red', label='natural partition')

# plot modularity relative to algorithm partition
ax.scatter(comms, mod2, s=30, marker='o', color='green', label='algorithm partition')

ax.set_xlabel('communities', fontsize=16)
ax.set_ylabel('modularity', fontsize=16)
plt.legend(fontsize=12, loc='lower right')

for tx in ax.xaxis.get_major_ticks():
    tx.label.set_fontsize(12)
    
for ty in ax.yaxis.get_major_ticks():
    ty.label.set_fontsize(12)

plt.grid()
plt.show()
