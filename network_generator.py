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

    def number_of_nodes(self):
        return self.A.shape[0]

    def number_of_edges(self):
        return int(np.sum(self.A)*0.5)

    def degrees_of_nodes(self):
        nodes = self.number_of_nodes()
        return np.array([np.sum(self.A[n,:]) for n in range(nodes)], dtype=int)

    def set_nodes_community(self, communities):
        self.number_of_communities = len(communities)
        nodes = self.number_of_nodes()
        self.nodes_community = np.empty(nodes, dtype=int)
        for i,comm in enumerate(communities):
            for n in comm:
                self.nodes_community[n] = i

    def _matrix_E(self, S, norm=1.):
        return norm * (S @ self.A @ S.T)

    def _vector_A(self, S, norm=1.):
            return norm * (S @ self.degrees_of_nodes())

    def modularity(self):
        n = self.number_of_nodes()
        m = self.number_of_edges()
        c = self.number_of_communities
        S = np.vstack([[i]*n for i in range(c)]) == self.nodes_community
        norm = 1/(2*m)
        return np.sum(np.diag( self._matrix_E(S,norm) ) - np.square( self._vector_A(S,norm) ))

    def show(self, ax):
        ax.imshow(self.A, cmap="binary")


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

    blocks = 3
    blocks_sizes = np.array([1000, 600, 1600])
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

    perms = np.random.permutation(range(0,3200))
    communities = [perms[:1000], perms[1000:1600], perms[1600:3200]]

    ti = time.time()
    mod = nx.algorithms.community.modularity(blocks_networkx, communities)
    tf = time.time()
    print(f"modularity (networkx) = {mod}")
    print(f"time (networkx) = {tf-ti} sec")
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(nx.to_numpy_array(blocks_networkx), cmap="binary")
    plt.show()

    ti = time.time()
    blocks_numpy = Random_Blocks(blocks_sizes, prob_matrix,
                                 check_parameters=False, check_adjacency=False)
    tf = time.time()
    print(f"\nstochastic_block_model (numpy) = {tf-ti} sec")
    blocks_numpy.set_nodes_community(communities)
    ti = time.time()
    mod = blocks_numpy.modularity()
    tf = time.time()
    print(f"modularity (numpy) = {mod}")
    print(f"time (numpy) = {tf-ti} sec")
    fig, ax = plt.subplots(figsize=(8,8))
    blocks_numpy.show(ax)
    plt.show()