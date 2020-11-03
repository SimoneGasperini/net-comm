import numpy as np
import pylab as plt
from time import time

import sys
sys.path.insert(0,"..")
from randblocks import RandomBlocks


blocks = 3
n1=800; n2=800; n3=800
p_kk_i = 0.02 # initial edge probability within each block
p_kk_f = 0.2 # final edge probability within each block
num_points = 10
delta_p_kk = round((p_kk_f-p_kk_i)/(num_points-1), 5)

blocks_sizes = np.array([n1, n2, n3])

prob_matrix = np.zeros(shape=(blocks,blocks))
for i in range(blocks-1):
    for j in range(i+1, blocks):
        prob_matrix[i][j] = 0.002
prob_matrix += prob_matrix.T
for k in range(blocks):
    prob_matrix[k][k] = p_kk_i


update_matrix = np.ones(shape=(blocks,blocks)) - np.eye(blocks)

time_seq = np.empty(num_points)
mod_seq = np.empty(num_points)
p_ij_seq = np.empty(num_points)

print(f"Clustering increasing each time edge probability within each block by {delta_p_kk} -->", flush=True)
for i in range(num_points):

    random_blocks = RandomBlocks(blocks_sizes, prob_matrix)

    ti = time()
    random_blocks.clustering(check_result=True)
    tf = time()
    time_seq[i] = tf-ti

    mod_seq[i] = random_blocks.modularity()
    p_ij_seq[i] = prob_matrix[0][0]

    np.fill_diagonal(prob_matrix, np.diag(prob_matrix)+delta_p_kk)

plt.style.use('seaborn-paper')
fig, ax1 = plt.subplots(figsize=(8,8))

ax1.set_title("modularity trend")
ax1.plot(p_ij_seq, mod_seq, color="orange")
ax1.set_xlabel("edge probability within each block")
ax1.set_ylabel("modularity after clustering")
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')


plt.suptitle(f"Clustering increasing each time edge probability within each block by {delta_p_kk}")
plt.show()
