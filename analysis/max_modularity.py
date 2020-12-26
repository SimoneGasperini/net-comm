import numpy as np
import pylab as plt

import sys
sys.path.append("..")

from model.randomnet import RandomNetwork
from model.compositenet import CompositeNetwork



# MODEL: disjoint random networks with fixed number of edges
comms = np.arange(1,31)
edges = 800

modularity = np.empty(comms.shape)


for c in comms:

    if c == 1:
        modularity[0] = 0.
        continue

    nodes = np.random.randint(low=50, high=150, size=c)
    unetworks = [RandomNetwork(n=n, m=edges) for n in nodes]
    edge_matrix = np.zeros(shape=(c, c), dtype=int)

    composite = CompositeNetwork(unetworks, edge_matrix)
    part = composite.clustering(check_result=True)
    modularity[c-1] = composite.modularity(part)


#%%
plt.style.use('seaborn-paper')

fig, ax = plt.subplots(figsize=(8,8))

# plot theoretical function for maximum modularity
ax.plot(comms, 1 - (1 / comms), label='theory')

# plot simulations result for maximum modularity
ax.scatter(comms, modularity, s=30, color='red', label='simulation')

ax.set_xlabel('networks', fontsize=16)
ax.set_ylabel('modularity', fontsize=16)
plt.legend(fontsize=12, loc='lower right')

for tx in ax.xaxis.get_major_ticks():
    tx.label.set_fontsize(12)
    
for ty in ax.yaxis.get_major_ticks():
    ty.label.set_fontsize(12)

plt.grid()
plt.show()
