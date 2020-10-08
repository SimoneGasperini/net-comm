import numpy as np
import pylab as plt
import networkx as nx
import time

class UndirectedNetwork:

    def __init__(self, adjacency, check_adjacency=True):
        self.A = adjacency
        if check_adjacency:
            self._check_adjacency()
        
    def _check_adjacency(self):
        if self.A.ndim != 2:
            raise ValueError("The array must be 2-dimensional")
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("The matrix must be square")
        if not np.allclose(self.A, self.A.T, rtol=1e-05, atol=1e-08):
            raise NotImplementedError("Only undirected networks are supported: the matrix must be symmetric")
            
    def get_numNodes(self):
        return self.A.shape[0]

    def get_numEdges(self):
        return int(np.sum(self.A)*0.5)
        
    def show(self, ax):
        ax.spy(self.A, marker=".", markersize=1)


class Erdos_Renyi(UndirectedNetwork):

    def __init__(self, n, p, seed=None, check_adjacency=True):
        if seed is not None:
            np.random.seed(seed)
        A = np.random.rand(n,n)
        A = np.where(A < p, 1, 0)
        A = np.triu(A) + np.triu(A).T - np.diag(np.diag(A))
        UndirectedNetwork.__init__(self, A, check_adjacency)



if __name__ == "__main__":
    
    nodes = 5000
    prob = 0.01
    seed = 2
    
    ti = time.time()
    net_networkx = nx.gnp_random_graph(n=nodes, p=prob, seed=seed)
    tf = time.time()
    print(f"\ntime for networkx = {tf-ti} sec")

    # this fast version of the generation algorithm performs better for sparse network (prob << 1)
    ti = time.time()
    net_networkx_fast = nx.fast_gnp_random_graph(n=nodes, p=prob, seed=seed)
    tf = time.time()
    print(f"time for networkx(fast) = {tf-ti} sec")

    ti = time.time()
    net_numpy = Erdos_Renyi(n=nodes, p=prob, seed=seed, check_adjacency=False)
    tf = time.time()
    print(f"\ntime for numpy = {tf-ti} sec")
    
    print(f"nodes = {net_numpy.get_numNodes()}")
    print(f"edges = {net_numpy.get_numEdges()}")
    #fig, ax = plt.subplots(figsize=(8,8))
    #net_numpy.show(ax)
    #plt.show()