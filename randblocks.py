import numpy as np
from unetwork import UndirectedNetwork
from unetwork import ones_random_symm, ones_random

class RandomBlocks(UndirectedNetwork):

    def __init__(self, sizes, p_matrix, seed=None):
        self.blocks = sizes.size
        self.block_sizes = sizes
        self.p = p_matrix
        if seed is not None:
            np.random.seed(seed)
        n = np.sum(sizes)
        A = np.zeros(shape=(n,n), dtype=int)
        for i in range(self.blocks):
            row = col = np.sum(sizes[:i])
            for j in range(i, self.blocks):
                r = sizes[i]
                c = sizes[j]
                p = p_matrix[i][j]
                A[row:row+r, col:col+c] = ones_random_symm(r,p) if i==j else ones_random((r,c),p)
                col += c
            row += r
        A = np.triu(A) + np.triu(A).T - np.diag(2*np.diag(A))
        perms = np.random.permutation(range(n)) 
        self.adjacency = A[perms,:][:,perms]
        self.number_of_partitions = 1
        self.partitions = [range(n)]
        m = int(np.sum(self.adjacency)*0.5)
        node1_list, node2_list = np.nonzero(self.adjacency)
        edge_dict = {}
        for node1,node2 in zip(node1_list,node2_list):
            edge_dict.setdefault(node1,[]).append(node2)
        UndirectedNetwork.__init__(self,n,m,edge_dict, build_adjacency=True)