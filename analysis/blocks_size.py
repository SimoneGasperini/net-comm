import numpy as np
import pylab as plt

import sys
sys.path.append("..")

from model.erdosrenyi_blocks import ErdosRenyiBlocks


blocks = 10
prob_within = 0.2
prob_between = 0.01
prob_matrix = np.empty(shape=(blocks, blocks))
for i in range(blocks):
    for j in range(i, blocks):
        prob_matrix[i][j] = prob_matrix[j][i] = prob_within if i == j else prob_between

a = b = 500
iterations = 20
mod_list = []
sigma_list = []

for iter in range(iterations):
    a -= 20 * iter
    b += 20 * iter
    blocks_sizes = np.random.randint(low=a, high=b + 1, size=blocks)
    network = ErdosRenyiBlocks(blocks_sizes, prob_matrix)
    part = network.clustering(check_result=True)
    mod_list.append(network.modularity(part))
    sigma_list.append((b - a) / np.sqrt(12))


fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(sigma_list, mod_list)
ax.set_title("modularity over heterogeneity")
plt.show()
