import numpy as np
import pylab as plt

from model.erdosrenyi_blocks import ErdosRenyiBlocks
from model.visualization import draw



blocks = 4
blocks_sizes = np.array([np.random.randint(300,600) for i in range(blocks)])
prob_within = 0.1
prob_between = 0.002

prob_matrix = np.empty(shape=(blocks,blocks))

for i in range(blocks):
    for j in range(i, blocks):
        prob_matrix[i][j] = prob_matrix[j][i] = prob_within if i == j else prob_between

prob_matrix[0,0] = 0.18
prob_matrix[2,2] = 0.25


er_blocks = ErdosRenyiBlocks(blocks_sizes, prob_matrix)

partition = er_blocks.clustering(check_result=True)
modularity = er_blocks.modularity(partition)
print(f"\nModularity = {modularity}")


fig, ax = plt.subplots(figsize=(8,8))
draw(er_blocks, partition, ax=ax)
plt.show()

fig, ax = plt.subplots(figsize=(8,8))
er_blocks.show(partition, ax=ax)
plt.show()
