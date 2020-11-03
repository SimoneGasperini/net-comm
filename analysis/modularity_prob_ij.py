import numpy as np
import pylab as plt
from time import time

import sys
sys.path.insert(0,"..")
from randblocks import RandomBlocks


blocks = 3
n1=1000; n2=1000; n3=1000
p_ij_i = 0.0005 # initial edge probability between different blocks
p_ij_f = 0.005 # final edge probability between different blocks
num_points = 10
delta_p_ij = round((p_ij_f-p_ij_i)/(num_points-1), 5)

blocks_sizes = np.array([n1, n2, n3])

prob_matrix = np.zeros(shape=(blocks,blocks))
for i in range(blocks-1):
    for j in range(i+1, blocks):
        prob_matrix[i][j] = p_ij_i
prob_matrix += prob_matrix.T
for k in range(blocks):
    prob_matrix[k][k] = 0.02


update_matrix = np.ones(shape=(blocks,blocks)) - np.eye(blocks)

time_seq = np.empty(num_points)
mod_seq = np.empty(num_points)
p_ij_seq = np.empty(num_points)

print(f"Clustering increasing each time edge probability between blocks by {delta_p_ij} -->", flush=True)
for i in range(num_points):

    random_blocks = RandomBlocks(blocks_sizes, prob_matrix)

    ti = time()
    random_blocks.clustering(check_result=True)
    tf = time()
    time_seq[i] = tf-ti

    mod_seq[i] = random_blocks.modularity()
    p_ij_seq[i] = prob_matrix[0][1]

    prob_matrix += update_matrix * delta_p_ij

plt.style.use('seaborn-paper')
fig, ax1 = plt.subplots(figsize=(8,8))

ax1.set_title("modularity trend")
ax1.plot(p_ij_seq, mod_seq, color="orange")
ax1.set_xlabel("edge probability between blocks")
ax1.set_ylabel("modularity after clustering")
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')


plt.suptitle(f"Clustering increasing each time edge probability between blocks by {delta_p_ij}")
plt.show()
