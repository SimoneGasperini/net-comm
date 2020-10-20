import numpy as np
import matplotlib.pylab as plt
from time import time
from randblocks import RandomBlocks

modularity_list = []
time_seq = []

n = 300

block_list = []



blocks_i = 2
blocks_f = 32
blocks_sizes = np.ones(blocks_f, dtype=int)*n


prob_matrix = np.zeros(shape=(blocks_f,blocks_f))

for i in range(blocks_f-1):
    for j in range(i+1, blocks_f):
        prob_matrix[i][j] = 0.002
prob_matrix += prob_matrix.T
for k in range(blocks_f):
    prob_matrix[k][k] = 0.02

print(f"Clustering adding each time 1 block of size {n}")
for i in range(blocks_i, blocks_f+1):

    blocks_sizes_i = blocks_sizes[:i]
    prob_matrix_i = prob_matrix[:i, :i]

    random_blocks = RandomBlocks(blocks_sizes_i, prob_matrix_i)


    ti = time()
    random_blocks.clustering()
    tf = time()
    time_seq.append(tf-ti)

    mod2 = random_blocks.modularity()

    modularity_list.append(mod2)

    block_list.append(i)




fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,8))

ax0.set_title("time trend")
ax0.plot(block_list, time_seq)
ax0.set_xlabel("number of blocks")
ax0.set_ylabel("clustering time")



ax1.set_title("modularity trend")
ax1.plot(block_list, modularity_list, color="orange")
ax1.set_xlabel("number of blocks")
ax1.set_ylabel("modularity after clustering")

plt.suptitle(f"Clustering adding each time 1 block of size {n}")

plt.show()
