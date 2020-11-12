import numpy as np
import pylab as plt
from time import time

import sys
sys.path.insert(0,"..")
from randblocks import RandomBlocks

from scipy.optimize import curve_fit

def f_1(x, intercept, slope):
       return intercept + slope*x


def f_2(x, p0, p1, p2):
       return p0 + p1*x + p2*(x**2)

num_simulations = 10# number of repetitions of the same clustering
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

#fit
x = nodes_seq
y = mean_time

popt, pcov = curve_fit(f_2, x, y)

p0 = popt[0]
p1 = popt[1]
p2 = popt[2]

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(nodes_seq, mean_time, label='data')
#fit_plot
ax.plot(nodes_seq, f_2(nodes_seq, p0, p1, p2), 'r--', label='fit: p0=%5.3f, p1=%5.3f, p2=%5.3f' % tuple(popt))
ci = sigma_conf * std_time
ax.fill_between(nodes_seq, (mean_time-ci), (mean_time+ci), alpha=.2, label='c.i.')
ax.set_xlabel("number of nodes", fontsize=16)
ax.set_ylabel("time [s]", fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
for tx in ax.xaxis.get_major_ticks():
    tx.label.set_fontsize(12)
for ty in ax.yaxis.get_major_ticks():
    ty.label.set_fontsize(12)
plt.legend(loc="best")
plt.show()


#fit
x = edges_seq
y = mean_time

popt, pcov = curve_fit(f_1, x, y)
intercept = popt[0]
slope = popt[1]

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(edges_seq, mean_time, color="orange", label='data')
#fit_plot
ax.plot(edges_seq, f_1(edges_seq, intercept, slope), 'g--', label='fit: intercept=%5.4f, slope=%5.4f'  % (intercept, slope))
ci = sigma_conf * std_time
ax.fill_between(edges_seq, (mean_time-ci), (mean_time+ci), color="orange", alpha=.2, label='c.i.')
ax.set_xlabel("number of edges", fontsize=16)
ax.set_ylabel("time [s]", fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
for tx in ax.xaxis.get_major_ticks():
    tx.label.set_fontsize(12)
for ty in ax.yaxis.get_major_ticks():
    ty.label.set_fontsize(12)
plt.legend(loc="best")
plt.show()
