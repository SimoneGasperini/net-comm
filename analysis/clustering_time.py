import numpy as np
import pylab as plt
from time import time

import sys
sys.path.insert(0,"..")
from randblocks import RandomBlocks



num_simulations = 10 # number of repetitions of the same clustering
sigma_conf = 5. # sigma confidence level shown in the plot

blocks = 2
n_i = 300 # initial size of each block
n_f = 1000 # final size of each block
num_points = 10
delta_n = int((n_f-n_i)/(num_points-1))

prob_matrix = np.zeros(shape=(blocks,blocks))
for i in range(blocks-1):
    for j in range(i+1, blocks):
        prob_matrix[i][j] = 0.005
prob_matrix += prob_matrix.T
for k in range(blocks):
    prob_matrix[k][k] = 0.1


times = np.empty(shape=(num_simulations,num_points))
nodes_seq = np.empty(num_points)
edges_seq = np.empty(num_points)

for i in range(num_simulations):

    print(f"\nSimulation {i+1}/{num_simulations}")
    blocks_sizes = np.array([n_i, n_i])

    for j in range(num_points):

        random_blocks = RandomBlocks(blocks_sizes, prob_matrix)

        ti = time()
        random_blocks.clustering(check_result=True)
        tf = time()
        times[i][j] = tf-ti

        if i == 0:
            nodes_seq[j] = random_blocks.number_of_nodes
            edges_seq[j] = random_blocks.number_of_edges

        blocks_sizes += delta_n


plt.style.use('seaborn-paper')

mean_time = np.mean(times, axis=0)
std_time = np.std(times, axis=0)

fig, ax = plt.subplots(figsize=(8,8))
ax.set_title("time trend with nodes")
ax.plot(nodes_seq, mean_time)
ci = sigma_conf * std_time
ax.fill_between(nodes_seq, (mean_time-ci), (mean_time+ci), alpha=.2)
ax.set_xlabel("number of nodes")
ax.set_ylabel("clustering time")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.show()

fig, ax = plt.subplots(figsize=(8,8))
ax.set_title("time trend with edges")
ax.plot(edges_seq, mean_time, color="orange")
ci = sigma_conf * std_time
ax.fill_between(edges_seq, (mean_time-ci), (mean_time+ci), color="orange", alpha=.2)
ax.set_xlabel("number of edges")
ax.set_ylabel("clustering time")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.show()
