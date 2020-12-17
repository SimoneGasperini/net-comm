import numpy as np
import pylab as plt

from model.randomblocks import RandomBlocks
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


random_blocks = RandomBlocks(blocks_sizes, prob_matrix)

mod = random_blocks.clustering(check_result=True)

mod = random_blocks.modularity()
print(f"\nModularity = {mod}")


fig, ax = plt.subplots(figsize=(8,8))
draw(random_blocks, ax, color_communities=True)
plt.show()

fig, ax = plt.subplots(figsize=(8,8))
random_blocks.show(ax, show_communities=True)
plt.show()
