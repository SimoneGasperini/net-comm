import numpy as np

from model.unetwork import UndirectedNetwork



class RandomBlocks (UndirectedNetwork):


    def __init__ (self, block_sizes, prob_matrix, seed=None):

        # transform parameters in numpy arrays
        if not isinstance(block_sizes, np.ndarray):
            block_sizes = np.array(block_sizes)

        if not isinstance(prob_matrix, np.ndarray):
            prob_matrix = np.array(prob_matrix)

        # check type and value of parameters
        self._check_parameters(block_sizes, prob_matrix)

        # set number of blocks
        self.blocks = block_sizes.size

        # set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # compute number of nodes
        n = int(np.sum(block_sizes))

        # build adjacency matrix
        adjacency = self._compute_adjacency(n, block_sizes, prob_matrix)

        # compute number of edges
        m = int(np.sum(adjacency) * 0.5)

        UndirectedNetwork.__init__(self, n=n, m=m, edge_dict=None, adjacency=adjacency)


    def _check_parameters (self, block_sizes, prob_matrix):

        if not np.all((block_sizes % 1) == 0):
            raise TypeError("All the elements in 'block_sizes' must be ints")

        if not np.all(block_sizes > 0):
            raise TypeError("All the elements in 'block_sizes' must be > 0")

        if not (np.all(prob_matrix >= 0) and np.all(prob_matrix <= 1)):
            raise ValueError("All the elements in 'prob_matrix' must be floats in [0,1]")

        if not np.all(prob_matrix == prob_matrix.transpose()):
            raise ValueError("'prob_matrix' must be symmetric (undirected)")


    def _compute_adjacency (self, n, block_sizes, prob_matrix):

        A = np.zeros(shape=(n,n), dtype=int)

        for i in range(self.blocks):

            row = col = np.sum(block_sizes[:i])

            for j in range(i, self.blocks):

                r = block_sizes[i]
                c = block_sizes[j]
                p = prob_matrix[i][j]

                A[row:row+r, col:col+c] = np.where(np.random.rand(r, c) < p, 1, 0)

                col += c

            row += r

        A = np.triu(A) + np.triu(A).transpose() - np.diag(2 * np.diag(A))
        perms = np.random.permutation(range(n))

        return A[perms,:][:,perms]
