import numpy as np
from unetwork import UndirectedNetwork
from unetwork import ones_random_symm, ones_random


class RandomBlocks(UndirectedNetwork):

    def __init__(self, block_sizes, prob_matrix, seed=None):

        if not isinstance(block_sizes, np.ndarray): block_sizes = np.array(block_sizes)
        if not isinstance(prob_matrix, np.ndarray): prob_matrix = np.array(prob_matrix)

        self._check_parameters(block_sizes, prob_matrix)
        self.blocks = block_sizes.size

        if seed is not None: np.random.seed(seed)

        # build adjacency matrix A (symmetric and unweighted)
        n = int(np.sum(block_sizes))
        A = np.zeros(shape=(n,n), dtype=int)
        for i in range(self.blocks):
            row = col = np.sum(block_sizes[:i])
            for j in range(i, self.blocks):
                r = block_sizes[i]
                c = block_sizes[j]
                p = prob_matrix[i][j]
                A[row:row+r, col:col+c] = ones_random_symm(r,p) if i==j else ones_random((r,c),p)
                col += c
            row += r
        A = np.triu(A) + np.triu(A).T - np.diag(2*np.diag(A))
        perms = np.random.permutation(range(n)) 
        self.adjacency = A[perms,:][:,perms]

        m = int(np.sum(self.adjacency)*0.5)

        # build the edge dictionary from the adjacency matrix
        node1_list, node2_list = np.nonzero(self.adjacency)
        edge_dict = {node : [] for node in range(n)}
        for node1,node2 in zip(node1_list,node2_list):
            edge_dict[node1].append(node2)

        UndirectedNetwork.__init__(self,n,m,edge_dict, build_adjacency=True)

    def _check_parameters(self, block_sizes, prob_matrix):
        # check if all the elements in block_sizes are integers
        if not np.all(np.mod(block_sizes,1) == 0):
            raise TypeError("All the elements in block_sizes must be integer numbers")

        # check if all the elements in prob_matrix are positive (probability)
        if not np.all(np.where(prob_matrix>=0, True, False)):
            raise ValueError("All the elements in prob_matrix must be positive numbers")

        # check if all the elements in prob_matrix are equal or smaller than 1 (probability)
        if not np.all(np.where(prob_matrix<=1, True, False)):
            raise ValueError("All the elements in prob_matrix must be equal or smaller than 1")