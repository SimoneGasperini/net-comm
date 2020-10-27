import numpy as np
import pylab as plt
import networkx as nx
from time import time
from randblocks import RandomBlocks

def draw_nx(unet, ax, col_communities=False):
        if unet.adjacency is None:
            raise NotImplementedError("Draw method is not supported for very large networks")
        netx = nx.Graph(unet.adjacency)
        if col_communities and unet.number_of_communities > 1:
            p = unet.number_of_communities
            col = np.linspace(0,1,p)
            colors = np.empty(unet.number_of_nodes)
            for node in range(unet.number_of_nodes):
                for i,p in enumerate(unet.partition):
                    if node in p:
                        colors[node] = col[i]
        else:
            colors = "#1f78b4"
        nx.draw(netx, ax=ax, width=0.2, node_size=20, node_color=colors, cmap="viridis")


blocks = 3
n1=280; n2=330; n3=390
blocks_sizes = np.array([n1,n2,n3])
prob_matrix = np.zeros(shape=(blocks,blocks))
for i in range(blocks-1):
    for j in range(i+1, blocks):
        prob_matrix[i][j] = 0.0005
prob_matrix += prob_matrix.T
for k in range(blocks):
    prob_matrix[k][k] = 0.05

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
draw_nx(random_blocks, ax0, col_communities=True)
random_blocks.show(ax1, show_communities=True)
plt.show()