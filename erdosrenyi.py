import numpy as np
from unetwork import UndirectedNetwork


class ErdosRenyi(UndirectedNetwork):

    def __init__(self, n, p, seed=None):

        self._check_parameters(n,p)

        if seed is not None: np.random.seed(seed)

        # build adjacency matrix A (symmetric and unweighted)
        n = int(n)
        rand = np.random.rand(n,n)
        A = np.empty(shape=rand.shape, dtype=int)
        A = np.where(rand < p, 1, 0)
        self.adjacency = np.triu(A) + np.triu(A).T - np.diag(2*np.diag(A))

        m = int(np.sum(A)*0.5)

        # build the edge dictionary from the adjacency matrix
        node1_list, node2_list = np.nonzero(A)
        edge_dict = {}
        for node1,node2 in zip(node1_list,node2_list):
            edge_dict.setdefault(node1,[]).append(node2)

        UndirectedNetwork.__init__(self,n,m,edge_dict, build_adjacency=True)

    def _check_parameters(self, n, p):
        if not isinstance(n, (int,np.integer)):
            raise TypeError("n(number of nodes) must be an integer number")

        if not 0 <= p <= 1:
            raise ValueError("p(edge probability) must be in [0,1]")