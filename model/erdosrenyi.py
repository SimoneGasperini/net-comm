import numpy as np

from model.unetwork import UndirectedNetwork



class ErdosRenyi (UndirectedNetwork):


    def __init__ (self, n, p, seed=None):

        # check type and value of parameters
        self._check_parameters(n, p)

        # set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # build adjacency matrix
        adjacency = self._compute_adjacency(n, p)

        # compute number of edges
        m = int(np.sum(adjacency) * 0.5)

        UndirectedNetwork.__init__(self, n=n, m=m, edge_dict=None, adjacency=adjacency)


    def _check_parameters (self, n, p):

        if not n % 1 == 0:
            raise TypeError("The number of nodes 'n' must be an int")

        if not n > 0:
            raise ValueError("The number of nodes 'n' must be > 0")

        if not 0 <= p <= 1:
            raise ValueError("The edge probability 'p' must be a float in [0,1]")


    def _compute_adjacency (self, n, p):

        A = np.where(np.random.rand(n, n) < p, 1, 0).astype(int)

        return np.triu(A) + np.triu(A).transpose() - np.diag(2 * np.diag(A))
