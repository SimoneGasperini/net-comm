import numpy as np
import matplotlib.pylab as plt
from time import time
from randblocks import RandomBlocks


time_seq = []
nodes_list = []
edges_list = []


blocks = 2

n_i = 200

n_f = 2000

numero_punti = 40

delta_n = int((n_f -n_i)/numero_punti)

blocks_sizes = np.array([n_i, n_i])

prob_matrix = np.zeros(shape=(blocks,blocks))
for i in range(blocks-1):
    for j in range(i+1, blocks):
        prob_matrix[i][j] = 0.002
prob_matrix += prob_matrix.T
for k in range(blocks):
    prob_matrix[k][k] = 0.02



print(f"Clustering adding each time {delta_n*blocks} nodes for each fixed block")

for i in range(numero_punti):

    blocks_sizes += delta_n

    random_blocks = RandomBlocks(blocks_sizes, prob_matrix)

    ti = time()
    random_blocks.clustering()
    tf = time()
    time_seq.append(tf-ti)



    n = random_blocks.number_of_nodes
    m = random_blocks.number_of_edges
    nodes_list.append(n)
    edges_list.append(m)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,8))
ax0.set_title("time trend with nodes")
ax0.plot(nodes_list, time_seq)
ax0.set_xlabel("number of nodes")
ax0.set_ylabel("clustering time")


ax1.set_title("time trend with edges")
ax1.plot(edges_list, time_seq, color="orange")
ax1.set_xlabel("number of nodes")
ax1.set_ylabel("clustering time")



plt.suptitle(f"Clustering adding each time {delta_n*blocks} nodes for each fixed block")



plt.show()
