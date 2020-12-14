import numpy as np
import pylab as plt
from time import time
from model.randblocks import RandomBlocks



n = 300 # size of each block
blocks_i = 2 # initial number of blocks
blocks_f = 16 # final number of blocks

iters = blocks_f - blocks_i + 1

blocks_sizes = np.ones(blocks_f, dtype=int) * n

prob_matrix = np.zeros(shape=(blocks_f,blocks_f))

for i in range(blocks_f-1):
    for j in range(i+1, blocks_f):
        prob_matrix[i][j] = 0.001

prob_matrix += prob_matrix.transpose()

for k in range(blocks_f):
    prob_matrix[k][k] = 0.1


time_seq = np.empty(iters)
mod_seq = np.empty(iters)
blocks_seq = np.empty(iters)

print(f"Clustering adding each time 1 block of size {n} -->", flush=True)

for i in range(iters):

    k = i + blocks_i
    blocks_sizes_k = blocks_sizes[:k]
    prob_matrix_k = prob_matrix[:k, :k]

    random_blocks = RandomBlocks(blocks_sizes_k, prob_matrix_k)

    ti = time()
    random_blocks.clustering(check_result=True)
    tf = time()
    time_seq[i] = tf - ti

    mod_seq[i] = random_blocks.modularity()

    blocks_seq[i] = k



#%% plot modularity varying the number of blocks

plt.style.use("seaborn-paper")

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(blocks_seq, mod_seq, color="orange")

ax.set_xlabel("number of blocks", fontsize=16)
ax.set_ylabel("modularity", fontsize=16)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")

for tx in ax.xaxis.get_major_ticks():
    tx.label.set_fontsize(12)

for ty in ax.yaxis.get_major_ticks():
    ty.label.set_fontsize(12)

plt.show()
