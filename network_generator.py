import numpy as np

def ones_random_symm(dim, prob):
    mat = np.random.rand(dim,dim)
    mat = np.where(mat < prob, 1, 0)
    mat = np.triu(mat) + np.triu(mat).T - np.diag(np.diag(mat))
    return mat

def ones_random(shape, prob):
    rows, cols = shape
    mat = np.random.rand(rows,cols)
    mat = np.where(mat < prob, 1, 0)
    return mat

class UndirectedNetwork:

    def __init__(self, adjacency, check_adjacency=True):
        self.A = adjacency
        if check_adjacency:
            self._check_adjacency()

    def _check_adjacency(self):
        if self.A.ndim != 2:
            raise ValueError("The array is not 2-dimensional")
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("The matrix is not square")
        if not np.allclose(self.A, self.A.T, rtol=1e-05, atol=1e-08):
            raise NotImplementedError("Only undirected networks are supported: the matrix is not symmetric")

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
        A = ones_random_symm(dim=n, prob=p)
        UndirectedNetwork.__init__(self, A, check_adjacency)


class Random_Blocks(UndirectedNetwork):

    def __init__(self, sizes, p_matrix, seed=None, check_parameters=True, check_adjacency=True):
        self.blocks = sizes.size
        self.block_sizes = sizes
        if seed is not None:
            np.random.seed(seed)
        if check_parameters:
            self._check_parameters()
        n = np.sum(sizes)
        A = np.zeros(shape=(n,n))
        for i in range(self.blocks):
            row = col = np.sum(sizes[:i])
            for j in range(i, self.blocks):
                r = sizes[i]
                c = sizes[j]
                p = p_matrix[i][j]
                A[row:row+r, col:col+c] = ones_random_symm(r,p) if i==j else ones_random((r,c),p)
                col += c
            row += r
        A = np.triu(A) + np.triu(A).T - np.diag(np.diag(A))
        UndirectedNetwork.__init__(self, A, check_adjacency)

    def _check_parameters(self):
        if self.blocks != self.p.shape[0]:
            raise ValueError("The number of blocks and the link probability matrix dimension do not match")


if __name__ == "__main__":

    import networkx as nx
    import pylab as plt
    import time

    nodes = 5000
    prob = 0.01
    seed = 2

    ti = time.time()
    net_networkx_fast = nx.fast_gnp_random_graph(n=nodes, p=prob, seed=seed)
    tf = time.time()
    print(f"fast_gnp_random_graph (networkx) = {tf-ti} sec")

    ti = time.time()
    net_numpy = Erdos_Renyi(n=nodes, p=prob, seed=seed, check_adjacency=False)
    tf = time.time()
    print(f"\nrandom graph (numpy) = {tf-ti} sec")
    
    print(f"nodes = {net_numpy.get_numNodes()}")
    print(f"edges = {net_numpy.get_numEdges()}")


    blocks = 3
    blocks_sizes = np.array([100, 60, 160])
    prob_matrix = np.zeros(shape=(blocks,blocks))
    for i in range(blocks-1):
        for j in range(i+1, blocks):
            prob_matrix[i][j] = 0.005
    prob_matrix += prob_matrix.T
    for k in range(blocks):
        prob_matrix[k][k] = 0.1

    ti = time.time()
    blocks_networkx = nx.stochastic_block_model(blocks_sizes, prob_matrix)
    tf = time.time()
    print(f"\nstochastic_block_model (networkx) = {tf-ti} sec")
    fig, ax = plt.subplots(figsize=(8,8))
    ax.spy(nx.adjacency_matrix(blocks_networkx), marker=".", markersize=1)
    plt.show()

    ti = time.time()
    blocks_numpy = Random_Blocks(blocks_sizes, prob_matrix,
                                 check_parameters=False, check_adjacency=False)
    tf = time.time()
    print(f"stochastic_block_model (numpy) = {tf-ti} sec")
    fig, ax = plt.subplots(figsize=(8,8))
    blocks_numpy.show(ax)
    plt.show()