import numpy as np
import pylab as plt
from time import time
from randblocks import RandomBlocks
from visualization import draw

blocks = 4
blocks_sizes = np.array([np.random.randint(200,600) for i in range(blocks)])
prob_within = 0.1
prob_between = 0.005
prob_matrix = np.empty(shape=(blocks,blocks))
for i in range(blocks):
    for j in range(i, blocks):
        prob_matrix[i][j] = prob_matrix[j][i] = prob_within if i == j else prob_between


ti = time()
random_blocks = RandomBlocks(blocks_sizes, prob_matrix)
tf = time()
print(f"\ntime for RBmodel generation = {tf-ti} sec", flush=True)

ti = time()
mod = random_blocks.clustering(return_modularity=True, check_result=True)
tf = time()
print(f"\ntime for clustering = {tf-ti}", flush=True)

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(mod)
plt.show()

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16,8))
draw(random_blocks, ax0, color_communities=True)
random_blocks.show(ax1, show_communities=True)
plt.show()