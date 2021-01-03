import numpy as np
import pylab as plt

import sys
sys.path.append("..")

from model.erdosrenyi_blocks import ErdosRenyiBlocks
from time import time


simulations = 20
blocks = 2
n_i = 500
n_f = 2000
prob_within = 0.1
prob_between = 0.005

prob_matrix = np.zeros(shape=(blocks,blocks))
for i in range(blocks):
    for j in range(i, blocks):
        prob_matrix[i][j] = prob_matrix[j][i] = prob_within if i == j else prob_between

times = np.empty(simulations)
nodes = np.empty(simulations)

delta = (n_f - n_i) / (simulations - 1)

for i in range(simulations):

    blocks_sizes = np.array([n_i + (delta * i)] * blocks).astype(int)
    network = ErdosRenyiBlocks(blocks_sizes, prob_matrix)

    ti = time()
    part = network.clustering(check_result=True)
    tf = time()

    times[i] = tf - ti
    nodes[i] = network.number_of_nodes


#%%
plt.style.use('seaborn-paper')


pars = np.polyfit(x=nodes, y=times, deg=2)

def parabola (x, pars):
    a, b, c = pars
    return (a * x**2) + (b * x) + c

fit_function = parabola(x=nodes, pars=pars)

fig, ax = plt.subplots(figsize=(8,6))

ax.scatter(nodes, times, marker='o', color='red', label='simulation')
ax.plot(nodes, fit_function, label='quadratic fit')

ax.set_xlabel('number of nodes', fontsize=16)
ax.set_ylabel('time [s]', fontsize=16)
ax.legend(fontsize=12, loc='upper left')

for tx in ax.xaxis.get_major_ticks():
    tx.label.set_fontsize(12)

for ty in ax.yaxis.get_major_ticks():
    ty.label.set_fontsize(12)

plt.grid()
plt.show()


#%%
fig.savefig('../images/clustering_time.pdf', bbox_inches='tight', dpi=1200)
