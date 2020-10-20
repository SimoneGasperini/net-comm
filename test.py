import numpy as np
import pylab as plt
from time import time
from randblocks import RandomBlocks

blocks = 3
n1=1000; n2=800; n3=900
n = n1+n2+n3
blocks_sizes = np.array([n1, n2, n3])
prob_matrix = np.zeros(shape=(blocks,blocks))
for i in range(blocks-1):
    for j in range(i+1, blocks):
        prob_matrix[i][j] = 0.001
prob_matrix += prob_matrix.T
for k in range(blocks):
    prob_matrix[k][k] = 0.01

ti = time()
random_blocks = RandomBlocks(blocks_sizes, prob_matrix)
tf = time()
print(f"\ntime for RBmodel generation = {tf-ti} sec", flush=True)
fig, ax = plt.subplots(figsize=(8,8))
random_blocks.show(ax)
plt.show()

ti = time()
random_blocks.clustering()
tf = time()
print(f"\ntime for clustering = {tf-ti}", flush=True)
fig, ax = plt.subplots(figsize=(8,8))
random_blocks.show(ax, show_communities=True)
plt.show()