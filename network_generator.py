import numpy as np
import networkx as nx

def ones_random_symm(dim, prob):
    rand = np.random.rand(dim,dim)
    mat = np.empty(shape=rand.shape, dtype=int)
    mat = np.where(rand < prob, 1, 0)
    mat = np.triu(mat) + np.triu(mat).T - np.diag(np.diag(mat))
    return mat

def ones_random(shape, prob):
    rows, cols = shape
    rand = np.random.rand(rows,cols)
    mat = np.empty(shape=rand.shape, dtype=int)
    mat = np.where(rand < prob, 1, 0)
    return mat

class UndirectedNetwork:

    def __init__(self, adjacency, check_adjacency=True):
        self.A = adjacency
        if check_adjacency:
            self._check_adjacency()
        self.edge_list = self._edge_list()
        self.netx = nx.Graph(self.A)

    def _check_adjacency(self):
        if self.A.ndim != 2:
            raise ValueError("The array is not 2-dimensional")
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("The matrix is not square")
        if not np.allclose(self.A, self.A.T, rtol=1e-05, atol=1e-08):
            raise NotImplementedError("Only undirected networks are supported: the matrix is not symmetric")

    def _edge_list(self):
        i_list, j_list = np.nonzero(np.triu(self.A))
        return list(zip(i_list,j_list))

    def number_of_nodes(self):
        return self.A.shape[0]

    def number_of_edges(self):
        return int(np.sum(self.A)*0.5)

    def degrees_of_nodes(self):
        n = self.number_of_nodes()
        return {node : np.sum(self.A[node,:]) for node in range(n)}

    def get_nodes_in_community(self, comm):
        n = self.number_of_nodes()
        return np.array([node for node in range(n) if self.node_community[node]==comm])

    def modularity_nx(self, partitions=None):
        communities = partitions if partitions is not None else self.partitions
        return nx.algorithms.community.modularity(self.netx, communities)

    def clustering_nx(self):
        communities = nx.algorithms.community.greedy_modularity_communities(self.netx)
        self.number_of_partitions = len(communities)
        self.partitions = communities

    def draw(self, ax, community_colors=False):
        if not community_colors or self.number_of_partitions == 1:
            colors = "#1f78b4"
        else:
            p = self.number_of_partitions
            col = np.linspace(0,1,p)
            colors = np.empty(self.number_of_nodes())
            for node in range(self.number_of_nodes()):
                for i,p in enumerate(self.partitions):
                    if node in p:
                        colors[node] = col[i]
        nx.draw(self.netx, ax=ax, width=0.2, node_size=50, node_color=colors, cmap="viridis")

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
        self.p = prob_matrix
        if seed is not None:
            np.random.seed(seed)
        if check_parameters:
            self._check_parameters()
        n = np.sum(sizes)
        A = np.zeros(shape=(n,n), dtype=int)
        for i in range(self.blocks):
            row = col = np.sum(sizes[:i])
            for j in range(i, self.blocks):
                r = sizes[i]
                c = sizes[j]
                p = p_matrix[i][j]
                A[row:row+r, col:col+c] = ones_random_symm(r,p) if i==j else ones_random((r,c),p)
                col += c
            row += r
        A = np.triu(A) + np.triu(A).T
        A -= np.diag(np.diag(A))
        UndirectedNetwork.__init__(self, A, check_adjacency)
        self.number_of_partitions = 1
        self.partitions = [range(self.number_of_nodes())]

    def _check_parameters(self):
        if self.blocks != self.p.shape[0]:
            raise ValueError("The number of blocks and the link probability matrix dimension do not match")


if __name__ == "__main__":

    import pylab as plt
    import time

    blocks = 3
    n1=100; n2=60; n3=160
    n = n1+n2+n3
    blocks_sizes = np.array([n1, n2, n3])
    prob_matrix = np.zeros(shape=(blocks,blocks))
    for i in range(blocks-1):
        for j in range(i+1, blocks):
            prob_matrix[i][j] = 0.005
    prob_matrix += prob_matrix.T
    for k in range(blocks):
        prob_matrix[k][k] = 0.1

    ti = time.time()
    random_blocks = Random_Blocks(blocks_sizes, prob_matrix,
                                 check_parameters=True, check_adjacency=True)
    tf = time.time()
    print(f"\ntime for RBmodel generation = {tf-ti} sec")
    fig, ax = plt.subplots(figsize=(8,8))
    random_blocks.show(ax)
    plt.show()

    mod1 = random_blocks.modularity_nx()
    print(f"\nmodularity Q(before clustering) = {mod1}")

    fig, ax = plt.subplots(figsize=(8,8))
    random_blocks.draw(ax, community_colors=True)
    plt.show()

    random_blocks.clustering_nx()

    mod2 = random_blocks.modularity_nx()
    print(f"\nmodularity Q(after clustering) = {mod2}")

    fig, ax = plt.subplots(figsize=(8,8))
    random_blocks.draw(ax, community_colors=True)
    plt.show()