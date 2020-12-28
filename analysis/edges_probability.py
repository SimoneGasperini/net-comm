import numpy as np
import pylab as plt

import sys
sys.path.append("..")

from model.erdosrenyi_blocks import ErdosRenyiBlocks


''' Changing link density within'''

blocks = 5
blocks_sizes = np.array([600 for i in range(blocks)])
prob_within = 0.01
prob_between = 0.01
prob_matrix = np.empty(shape=(blocks, blocks))
for i in range(blocks):
    for j in range(i, blocks):
        prob_matrix[i][j] = prob_matrix[j][i] = prob_within if i == j else prob_between

iterations = 1

prob_within_max = 0.1
delta_prob_within = float((prob_within_max - prob_within) / iterations)
mod_list = []

for iter in range(iterations):
    prob_matrix += delta_prob_within * np.eye(blocks)
    network = ErdosRenyiBlocks(blocks_sizes, prob_matrix)
    part = network.clustering(check_result=True)
    mod_list.append(network.modularity(part))

blocks = 5
blocks_sizes = np.array([600 for i in range(blocks)])
prob_within = 0.01
prob_between = 0.01
prob_matrix = np.empty(shape=(blocks, blocks))
for i in range(blocks):
    for j in range(i, blocks):
        prob_matrix[i][j] = prob_matrix[j][i] = prob_within if i == j else prob_between


prob_within_max = 0.2
delta_prob_within = float((prob_within_max - prob_within) / iterations)
mod_list_2 = []

for iter in range(iterations):
    prob_matrix += delta_prob_within * np.eye(blocks)
    network = ErdosRenyiBlocks(blocks_sizes, prob_matrix)
    part = network.clustering(check_result=True)
    mod_list_2.append(network.modularity(part))

blocks = 5
blocks_sizes = np.array([600 for i in range(blocks)])
prob_within = 0.01
prob_between = 0.01
prob_matrix = np.empty(shape=(blocks, blocks))
for i in range(blocks):
    for j in range(i, blocks):
        prob_matrix[i][j] = prob_matrix[j][i] = prob_within if i == j else prob_between


prob_within_max = 0.3
delta_prob_within = float((prob_within_max - prob_within) / iterations)
mod_list_3 = []

for iter in range(iterations):
    prob_matrix += delta_prob_within * np.eye(blocks)
    network = ErdosRenyiBlocks(blocks_sizes, prob_matrix)
    part = network.clustering(check_result=True)
    mod_list_3.append(network.modularity(part))

points = np.arange(iterations)


fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(points, mod_list, label="p_max =0.01")
ax.plot(points, mod_list_2, label="p_max =0.02")
ax.plot(points, mod_list_3, label="p_max =0.03")
ax.set_title("Changing link density within")
ax.legend(loc="best")


''' Changing link density between'''

blocks = 5
blocks_sizes = np.array([np.random.randint(200, 600) for i in range(blocks)])
prob_within = 0.1
prob_between = 0.01
prob_matrix = np.empty(shape=(blocks, blocks))
for i in range(blocks):
    for j in range(i, blocks):
        prob_matrix[i][j] = prob_matrix[j][i] = prob_within if i == j else prob_between


prob_between_max = 0.1
delta_prob_between = float((prob_between_max - prob_between) / iterations)
mod_list = []

for iter in range(iterations):
    prob_matrix += delta_prob_between * (1 - np.eye(blocks))
    network = ErdosRenyiBlocks(blocks_sizes, prob_matrix)
    part = network.clustering(check_result=True)
    mod_list.append(network.modularity(part))


blocks = 5
blocks_sizes = np.array([np.random.randint(200, 600) for i in range(blocks)])
prob_within = 0.2
prob_between = 0.01
prob_matrix = np.empty(shape=(blocks, blocks))
for i in range(blocks):
    for j in range(i, blocks):
        prob_matrix[i][j] = prob_matrix[j][i] = prob_within if i == j else prob_between


prob_between_max = 0.2
delta_prob_between = float((prob_between_max - prob_between) / iterations)
mod_list_2 = []

for iter in range(iterations):
    prob_matrix += delta_prob_between * (1 - np.eye(blocks))
    network = ErdosRenyiBlocks(blocks_sizes, prob_matrix)
    part = network.clustering(check_result=True)
    mod_list_2.append(network.modularity(part))

blocks = 5
blocks_sizes = np.array([np.random.randint(200, 600) for i in range(blocks)])
prob_within = 0.3
prob_between = 0.01
prob_matrix = np.empty(shape=(blocks, blocks))
for i in range(blocks):
    for j in range(i, blocks):
        prob_matrix[i][j] = prob_matrix[j][i] = prob_within if i == j else prob_between


prob_between_max = 0.3
delta_prob_between = float((prob_between_max - prob_between) / iterations)
mod_list_3 = []

for iter in range(iterations):
    prob_matrix += delta_prob_between * (1 - np.eye(blocks))
    network = ErdosRenyiBlocks(blocks_sizes, prob_matrix)
    part = network.clustering(check_result=True)
    mod_list_3.append(network.modularity(part))


points = np.arange(iterations)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(points, mod_list, label="p_max =0.01")
ax.plot(points, mod_list_2, label="p_max =0.02")
ax.plot(points, mod_list_3, label="p_max =0.03")
ax.set_title("Changing link density between")
ax.legend(loc="best")
plt.show()
