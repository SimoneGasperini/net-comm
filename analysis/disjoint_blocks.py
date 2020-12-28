import numpy as np
import pylab as plt

import sys
sys.path.append("..")

from model.randomnet import RandomNetwork
from model.composite_network import CompositeNetwork


'''
##############################################################################

We consider a series of networks composed by an increasing number of disjoint
blocks 'c'. Each block is a random connected network with a variable number of
nodes 'n' and a fixed number of edges 'm'.

We run the clustering algorithm on this model to check the correctness of the
communities detection (in this trivial case) and plot the modularity trend.
The expected theoretical function for modularity Q is:

    Q(c) = 1 - 1/c

##############################################################################
'''

comms = np.arange(2,41) # number of blocks/communities
n_min = 30              # minimum number of nodes for each block
n_max = 60              # maximum number of nodes for each block
m = 200                 # number of edges for each block

modularity = np.empty(comms.shape)


for c in comms:

    nodes = np.random.randint(low=n_min, high=n_max+1, size=c)
    unetworks = [RandomNetwork(n=n, m=m, force_connected=True) for n in nodes]

    edge_matrix = np.zeros(shape=(c, c), dtype=int)

    composite = CompositeNetwork(unetworks, edge_matrix)

    part = composite.clustering(check_result=True)
    modularity[c-comms[0]] = composite.modularity(part)


#%%
plt.style.use('seaborn-paper')


fig, ax = plt.subplots(figsize=(8,8))

f = lambda x : 1 - 1 / x
x = np.linspace(comms[0], comms[-1], num=1000)
ax.plot(x, f(x), label='theory')

ax.scatter(comms, modularity, s=30, color='red', label='simulation')

ax.set_xlabel('number of blocks', fontsize=16)
ax.set_ylabel('modularity', fontsize=16)
plt.legend(fontsize=12, loc='lower right')

for tx in ax.xaxis.get_major_ticks():
    tx.label.set_fontsize(12)

for ty in ax.yaxis.get_major_ticks():
    ty.label.set_fontsize(12)

plt.grid()
plt.show()
