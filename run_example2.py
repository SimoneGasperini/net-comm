import numpy as np
import pylab as plt

from model.randomnet import RandomNetwork
from model.compositenet import CompositeNetwork
from model.visualization import draw



def check_constraints (network, partition):
    '''
    Given a network and its true partition, check if the constraints
    to have a successful communities detection (formula [5], Fortunato) are satisfied.
    '''

    l = np.array([len(network._edges_within_comm(c)) * 0.5 for c in partition], dtype=int)
    a = np.array([network._outFactor_of_comm(c) for c in partition])
    l_max = network.number_of_edges * 0.25

    success = True

    if not np.all(l < l_max):
        print("The constraints are NOT satisfied (l >= L/4)")
        success = False

    if not np.all(a < 2):
        print("The constraints are NOT satisfied (a >= 2)")
        success = False

    if success:
        print("The constraints are satisfied")

    return l, l_max, a


n1 = 220
n2 = 300
n3 = 200
unetworks = [
    RandomNetwork(n=n1, m=2000),
    RandomNetwork(n=n2, m=1900),
    RandomNetwork(n=n3, m=1600)
]

num = len(unetworks)

edge_matrix = np.empty(shape=(num, num), dtype=int)
for i in range(num):
    for j in range(i, num):
        edge_matrix[i][j] = edge_matrix[j][i] = 0 if i == j else 1000


composite = CompositeNetwork(unetworks, edge_matrix)

partition = [set(range(0, n1)), set(range(n1, n1+n2)), set(range(n1+n2, n1+n2+n3))]


l, l_max, a = check_constraints(network=composite, partition=partition)
print(f"\nl = {l}")
print(f"l_max = L/4 = {l_max}")
print(f"a = {a}\n", flush=True)

partition = composite.clustering(check_result=True)

fig, ax = plt.subplots(figsize=(8,8))
draw(composite, partition, ax=ax)
plt.show()
