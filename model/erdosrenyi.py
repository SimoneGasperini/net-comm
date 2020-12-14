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

        # build edges dictionary
        edge_dict = self._compute_edge_dict(n, adjacency)

        UndirectedNetwork.__init__(self, n=n, m=m, edge_dict=edge_dict, adjacency=adjacency)


    def _check_parameters (self, n, p):

        if not n % 1 == 0:
            raise TypeError("The number of nodes 'n' must be an int")

        if not 0 <= p <= 1:
            raise ValueError("The edge probability 'p' must be a float in [0,1]")


    def _compute_adjacency (self, n, p):

        A = np.where(np.random.rand(n, n) < p, 1, 0).astype(int)

        return np.triu(A) + np.triu(A).transpose() - np.diag(2 * np.diag(A))


    def _compute_edge_dict (self, n, A):

        node1_list, node2_list = np.nonzero(A)

        edge_dict = {node : [] for node in range(n)}

        for node1, node2 in zip(node1_list, node2_list):
            edge_dict[node1].append(node2)

        return edge_dict
