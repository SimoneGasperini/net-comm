import numpy as np
import matplotlib.pylab as plt
from time import time
from randblocks import RandomBlocks

modularity_list = []
time_seq = []
prob_matrix_ij = []

blocks = 3
n1=300; n2=300; n3=300
blocks_sizes = np.array([n1, n2, n3])

p_ij_i = 0.005
p_ij_f = 0.02

numero_punti = 50
delta_p_ij = (p_ij_f - p_ij_i)/numero_punti
update_prob = np.ones(shape=(blocks,blocks))-np.eye(blocks)

prob_matrix = np.zeros(shape=(blocks,blocks))
for i in range(blocks-1):
    for j in range(i+1, blocks):
        prob_matrix[i][j] = p_ij_i
prob_matrix += prob_matrix.T
for k in range(blocks):
    prob_matrix[k][k] = 0.1

print(f"Clustering with increasing prob_matrix[i][j] of {delta_p_ij} each time")


for k in range(numero_punti):

    random_blocks = RandomBlocks(blocks_sizes, prob_matrix)

    ti = time()
    random_blocks.clustering()
    tf = time()
    time_seq.append(tf-ti)

    mod2 = random_blocks.modularity()
    modularity_list.append(mod2)

    prob_matrix_ij.append(prob_matrix[i][j])


    prob_matrix += update_prob * delta_p_ij



fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,8))
ax0.set_title("time trend")
ax0.plot(prob_matrix_ij, time_seq)
ax0.set_xlabel("prob_matrix_ij")
ax0.set_ylabel("clustering time")

ax1.set_title("modularity trend")
ax1.plot(prob_matrix_ij, modularity_list, color="orange")
ax1.set_xlabel("prob_matrix_ij")
ax1.set_ylabel("modularity after clustering")


plt.suptitle(f"Clustering with increasing prob_matrix[i][j] of {delta_p_ij} each time")



plt.show()
