import numpy as np
import pylab as plt

from model.erdosrenyi_blocks import ErdosRenyiBlocks
from model.visualization import draw



blocks = 4
blocks_sizes = np.array([np.random.randint(100,300) for i in range(blocks)])
prob_within = 0.08
prob_between = 0.0005

prob_matrix = np.empty(shape=(blocks,blocks))

for i in range(blocks):
    for j in range(i, blocks):
        prob_matrix[i][j] = prob_matrix[j][i] = prob_within if i == j else prob_between

prob_matrix[0,0] = 0.1
prob_matrix[2,2] = 0.25


er_blocks = ErdosRenyiBlocks(blocks_sizes, prob_matrix)

partition = er_blocks.clustering(check_result=True)
modularity = er_blocks.modularity(partition)
print(f"\nModularity = {modularity}")


fig1, ax1 = plt.subplots(figsize=(8,8))
draw(er_blocks, partition, ax=ax1, cmap='plasma')
plt.show()

fig2, ax2 = plt.subplots(figsize=(8,8))
er_blocks.show(partition, ax=ax2, cmap='plasma')
plt.show()


#%%
fig1.savefig('images/toy_model.pdf', bbox_inches='tight', dpi=1200)

fig2.savefig('images/random_blocks.pdf', bbox_inches='tight', dpi=1200)
