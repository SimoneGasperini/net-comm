import numpy as np
import pylab as plt

import sys
sys.path.append("..")

from time import time
from model.randomblocks import RandomBlocks



num_simulations = 10 # number of repetitions of the same clustering
sigma_conf = 5. # sigma confidence level shown in the plot
blocks = 2 # number of blocks
n_i = 300 # initial size of each block
n_f = 1000 # final size of each block
num_points = 10

delta_n = int((n_f - n_i) / (num_points - 1))


prob_matrix = np.zeros(shape=(blocks,blocks))

for i in range(blocks-1):
    for j in range(i+1, blocks):
        prob_matrix[i][j] = 0.005
        
prob_matrix += prob_matrix.transpose()

for k in range(blocks):
    prob_matrix[k][k] = 0.1


times = np.empty(shape=(num_simulations,num_points))
nodes_seq = np.empty(num_points)


for i in range(num_simulations):

    blocks_sizes = np.array([n_i, n_i])

    for j in range(num_points):

        random_blocks = RandomBlocks(blocks_sizes, prob_matrix)

        ti = time()
        part = random_blocks.clustering(check_result=True)
        tf = time()
        times[i][j] = tf - ti

        if i == 0:
            nodes_seq[j] = random_blocks.number_of_nodes

        blocks_sizes += delta_n


#%% plot and fit clustering time, complexity O(n**2)

plt.style.use("seaborn-paper")

mean_time = np.mean(times, axis=0)
std_time = np.std(times, axis=0)

pars = np.polyfit(x=nodes_seq, y=mean_time, deg=2)

def parabola (x, pars):

    a, b, c = pars
    return (a * x**2) + (b * x) + c

fit_function = parabola(x=nodes_seq, pars=pars)

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(nodes_seq, mean_time)
ax.plot(nodes_seq, fit_function, color="black", linestyle="--", label="quadratic fit")

ci = sigma_conf * std_time
ax.fill_between(nodes_seq, (mean_time - ci), (mean_time + ci), alpha=0.2)

ax.set_xlabel("nodes", fontsize=16)
ax.set_ylabel("time [s]", fontsize=16)

for tx in ax.xaxis.get_major_ticks():
    tx.label.set_fontsize(12)
    
for ty in ax.yaxis.get_major_ticks():
    ty.label.set_fontsize(12)

plt.legend(loc="upper left", fontsize=16)

plt.grid()
plt.show()
