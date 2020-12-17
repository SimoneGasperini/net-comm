import numpy as np

from model.unetwork import UndirectedNetwork



class RandomNetwork (UndirectedNetwork):


    def __init__ (self, n, m, seed=None):

        # check type and value of parameters
        self._check_parameters(n, m)

        # set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # build adjacency matrix
        adjacency = self._compute_adjacency(n, m)

        UndirectedNetwork.__init__(self, n=n, m=m, edge_dict=None, adjacency=adjacency)


    def _check_parameters (self, n, m):

        if not n % 1 == 0:
            raise TypeError("The number of nodes 'n' must be an int")

        if not n > 1:
            raise ValueError("The number of nodes 'n' must be > 1")

        if not m % 1 == 0:
            raise TypeError("The number of edges 'm' must be an int")

        if not 0 <= m <= (n * (n - 1)) * 0.5:
            raise ValueError("The number of edges 'm' must be in [0, n*(n-1)/2]")


    def _compute_adjacency (self, n, m):

        upper = [(i, j) for i, j in zip(*np.triu_indices(n)) if i != j]
        indices = np.random.permutation(upper)[:m]

        A = np.zeros(shape=(n, n), dtype=int)
        A[indices[:,0], indices[:,1]] = 1

        return A + A.transpose()
