import numpy as np
import pylab as plt
from time import time

import sys
sys.path.insert(0,"..")
from randblocks import RandomBlocks



blocks = 2
n_i = 200 # initial size of each block
n_f = 2000 # final size of each block
num_points = 10
delta_n = int((n_f-n_i)/(num_points-1))

blocks_sizes = np.array([n_i, n_i])

prob_matrix = np.zeros(shape=(blocks,blocks))
for i in range(blocks-1):
    for j in range(i+1, blocks):
        prob_matrix[i][j] = 0.005
prob_matrix += prob_matrix.T
for k in range(blocks):
    prob_matrix[k][k] = 0.1


time_seq = np.empty(num_points)
nodes_seq = np.empty(num_points)
edges_seq = np.empty(num_points)

print(f"Clustering adding each time {delta_n} nodes for each one of the {blocks} blocks -->", flush=True)
for i in range(num_points):

    random_blocks = RandomBlocks(blocks_sizes, prob_matrix)

    ti = time()
    random_blocks.clustering(check_result=True)
    tf = time()
    time_seq[i] = tf-ti

    nodes_seq[i] = random_blocks.number_of_nodes
    edges_seq[i] = random_blocks.number_of_edges

    blocks_sizes += delta_n


plt.style.use('seaborn-paper')
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,8))

ax0.set_title("time trend with nodes")
ax0.plot(nodes_seq, time_seq)
ax0.set_xlabel("number of nodes")
ax0.set_ylabel("clustering time")
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.xaxis.set_ticks_position('bottom')
ax0.yaxis.set_ticks_position('left')

ax1.set_title("time trend with edges")
ax1.plot(edges_seq, time_seq, color="orange")
ax1.set_xlabel("number of edges")
ax1.set_ylabel("clustering time")
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')


plt.suptitle(f"Clustering adding each time {delta_n} nodes for each one of the {blocks} blocks")
plt.show()
