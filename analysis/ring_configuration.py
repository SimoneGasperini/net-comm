import numpy as np
import pylab as plt

import sys
sys.path.append("..")

from model.randomnet import RandomNetwork
from model.composite_network import CompositeNetwork


def integer_approximation(tot, div):
    low = int(tot / div - 1)
    high = int(tot / div)
    t1 = (high * div) - (tot - div)
    t2 = div - t1
    return np.array([low]*t1 + [high]*t2)

def natural_partition(comms, nodes):
    n = np.cumsum([nodes for _ in range(comms)])
    return [set(range(n1, n2)) for n1, n2 in zip(n-nodes, n)]

def resolution_limit(m, L):
    condition = m < np.sqrt(2 * L)
    return m[condition][-1]


'''
##############################################################################

We consider a series of networks with a fixed total number of edges 'M' and an
increasing number of blocks 'c' in ring configuration.                                      
Each block is a random connected network with a fixed number of nodes 'n' and
the same number of internal edges 'm' = M/c - 1 (approximation to the
nearest integer).

We run the clustering algorithm on this model to check the correctness of the
communities detection and plot the modularity trend. The expected theoretical
function for modularity Q is:

    Q(c,M) = 1 - c/M - 1/c

In particular, we empirically verify the following theoretical results:

    (1) the modularity maximum value is observed at c_MAX = sqrt(M);

    (2) above the resolution limit at c_RL = sqrt(2*M), the communities
        detection algorithm (based on modularity optimization) fails because
        the natural partition does not correspond to the one with maximum
        modularity.

##############################################################################
'''

comms = np.arange(16,101) # number of blocks/communities
n = 20                    # number of nodes for each block
M = 3000                  # total number of edges

mod1 = np.empty(comms.shape)
mod2 = np.empty(comms.shape)

for c in comms:

    edges = integer_approximation(M, c)
    unetworks = [RandomNetwork(n=n, m=m, force_connected=True) for m in edges]
    
    edge_matrix = np.zeros(shape=(c, c), dtype=int)
    up_diag = np.array([(i, i+1) for i in range(c-1)])
    edge_matrix[up_diag[:,0], up_diag[:,1]] = 1
    edge_matrix[0,-1] = 1
    edge_matrix += edge_matrix.transpose()

    composite = CompositeNetwork(unetworks, edge_matrix)

    part = natural_partition(c, n)
    mod1[c-comms[0]] = composite.modularity(part)

    part = composite.clustering(check_result=True)
    mod2[c-comms[0]] = composite.modularity(part)


#%%
plt.style.use('seaborn-paper')


fig, ax = plt.subplots(figsize=(8,8))

maxq = np.sqrt(M)
ax.axvline(x=maxq, color='black', linestyle='--')
plt.text(maxq+1, 0.951, '$c_{MAX}=\sqrt{M}$', fontsize=12)

rl  = resolution_limit(comms, M)
ax.axvline(x=rl, color='black', linestyle='--')
plt.text(rl+1, 0.951, '$c_{RL}=\sqrt{2M}$', fontsize=12)

f = lambda x, y : 1 - x / y - 1 / x
x = np.linspace(comms[0], comms[-1], num=1000)
ax.plot(x, f(x, M), label='theory')

ax.scatter(comms, mod1, s=20, marker='^', color='green', label='natural partition')

ax.scatter(comms, mod2, s=20, marker='o', color='red', label='detected partition')

ax.set_xlabel('number of blocks', fontsize=16)
ax.set_ylabel('modularity', fontsize=16)
plt.legend(fontsize=12, loc='lower right')

for tx in ax.xaxis.get_major_ticks():
    tx.label.set_fontsize(12)

for ty in ax.yaxis.get_major_ticks():
    ty.label.set_fontsize(12)

plt.grid()
plt.show()
