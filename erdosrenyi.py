import numpy as np
from unetwork import UndirectedNetwork

class ErdosRenyi(UndirectedNetwork):

    def __init__(self, n, p, seed=None):
        if seed is not None:
            np.random.seed(seed)
        rand = np.random.rand(n,n)
        A = np.empty(shape=rand.shape, dtype=int)
        A = np.where(rand < p, 1, 0)
        self.adjacency = np.triu(A) + np.triu(A).T - np.diag(2*np.diag(A))
        m = int(np.sum(A)*0.5)
        node1_list, node2_list = np.nonzero(A)
        edge_dict = {}
        for node1,node2 in zip(node1_list,node2_list):
            edge_dict.setdefault(node1,[]).append(node2)
        UndirectedNetwork.__init__(self,n,m,edge_dict, build_adjacency=True)