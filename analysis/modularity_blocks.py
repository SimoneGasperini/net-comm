import numpy as np
import pylab as plt
from time import time

import sys
sys.path.insert(0,"..")
from randblocks import RandomBlocks


n = 300 # size of each block
blocks_i = 2 # initial number of blocks
blocks_f = 16 # final number of blocks
iters = blocks_f-blocks_i+1

blocks_sizes = np.ones(blocks_f, dtype=int) * n

prob_matrix = np.zeros(shape=(blocks_f,blocks_f))
for i in range(blocks_f-1):
    for j in range(i+1, blocks_f):
        prob_matrix[i][j] = 0.001
prob_matrix += prob_matrix.T
for k in range(blocks_f):
    prob_matrix[k][k] = 0.1


time_seq = np.empty(iters)
mod_seq = np.empty(iters)
blocks_seq = np.empty(iters)

print(f"Clustering adding each time 1 block of size {n} -->", flush=True)
for i in range(iters):

    k = i+blocks_i
    blocks_sizes_k = blocks_sizes[:k]
    prob_matrix_k = prob_matrix[:k, :k]
    random_blocks = RandomBlocks(blocks_sizes_k, prob_matrix_k)

    ti = time()
    random_blocks.clustering(check_result=True)
    tf = time()
    time_seq[i] = tf-ti

    mod_seq[i] = random_blocks.modularity()
    blocks_seq[i] = k


fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,8))

ax0.set_title("time trend")
ax0.plot(blocks_seq, time_seq)
ax0.set_xlabel("number of blocks")
ax0.set_ylabel("clustering time")

ax1.set_title("modularity trend")
ax1.plot(blocks_seq, mod_seq, color="orange")
ax1.set_xlabel("number of blocks")
ax1.set_ylabel("modularity after clustering")

plt.suptitle(f"Clustering adding each time 1 block of size {n}")
plt.show()