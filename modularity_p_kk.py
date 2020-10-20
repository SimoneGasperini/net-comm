import numpy as np
import matplotlib.pylab as plt
from time import time
from randblocks import RandomBlocks


modularity_list = []
time_seq = []
prob_matrix_kk = []

blocks = 3
n1=300; n2=300; n3=300
blocks_sizes = np.array([n1, n2, n3])


p_kk_i = 0.025
p_kk_f = 0.2

numero_punti = 50
delta_p_kk = (p_kk_f - p_kk_i)/numero_punti

prob_matrix = np.zeros(shape=(blocks,blocks))
for i in range(blocks-1):
    for j in range(i+1, blocks):
        prob_matrix[i][j] = 0.005
prob_matrix += prob_matrix.T
for k in range(blocks):
    prob_matrix[k][k] = p_kk_i



print(f"Clustering with increasing prob_matrix[k][k] of {delta_p_kk} each time")


for i in range(numero_punti):

    random_blocks = RandomBlocks(blocks_sizes, prob_matrix)

    ti = time()
    random_blocks.clustering()
    tf = time()
    time_seq.append(tf-ti)

    mod2 = random_blocks.modularity()
    modularity_list.append(mod2)

    prob_matrix_kk.append(prob_matrix[k][k])

    np.fill_diagonal(prob_matrix, np.diag(prob_matrix) + delta_p_kk)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,8))
ax0.set_title("time trend")
ax0.plot(prob_matrix_kk, time_seq)
ax0.set_xlabel("prob_matrix_kk")
ax0.set_ylabel("clustering time")

ax1.set_title("modularity trend")
ax1.plot(prob_matrix_kk, modularity_list, color="orange")
ax1.set_xlabel("prob_matrix_kk")
ax1.set_ylabel("modularity after clustering")

plt.suptitle(f"Clustering with increasing prob_matrix[k][k] of {delta_p_kk} each time")

plt.show()
