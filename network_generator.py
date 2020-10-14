import numpy as np
from tqdm import trange

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

    def nodes_in_community(self, comm):
        n = self.number_of_nodes()
        return np.array([node for node in range(n) if self.nodes_community[node]==comm])

    def modularity(self):
        n = self.number_of_nodes()
        c = self.number_of_communities
        community_of_node = np.array([self.nodes_community[node] for node in range(n)])
        S = np.vstack([[i]*n for i in range(c)]) == community_of_node
        degree_dict = self.degrees_of_nodes()
        k = np.array([degree_dict[node] for node in range(n)])
        norm = 1/(2*self.number_of_edges())
        B = self.A - (norm*np.outer(k,k))
        return norm * np.trace(S @ B @ S.T)

    def clustering(self):
        n = self.number_of_nodes()
        m = self.number_of_edges()
        k = self.degrees_of_nodes()
        q = self.modularity()
        norm = 1./(2.*m)
        a = np.array([norm * k[i] for i in range(n)])
        ij_dQ = {(i,j) : 2*norm - 2*k[i]*k[j]*norm*norm for (i,j) in self.edge_list}

        mod_list = []
        for iteration in trange(n-1):
            mod_list.append(q)
            delta_q = np.max(list(ij_dQ.values()))
            if delta_q <= 0:
                print(f"\niteration = {iteration}")
                print(f"max delta_q = {delta_q}")
                break
            for ij in ij_dQ:
                if ij_dQ[ij] == delta_q:
                    comms = ij
                    break
            c1, c2 = int(min(comms)), int(max(comms))
            for node in self.nodes_in_community(c2):
                self.nodes_community[node] = c1
            self.number_of_communities -= 1
            for k in range(self.number_of_communities):
                for (i,j) in ij_dQ:
                    if (k,i) in ij_dQ:
                        k_is_connected_to_i = True if ij_dQ[(k,i)] != 0 else False
                    if (k,j) in ij_dQ:
                        k_is_connected_to_j = True if ij_dQ[(k,j)] != 0 else False
                    if (j,k) in ij_dQ and (i,k) in ij_dQ:
                        if k_is_connected_to_i and k_is_connected_to_j:
                            ij_dQ[(j,k)] = ij_dQ[(i,k)] + ij_dQ[(j,k)]
                        if k_is_connected_to_i and not k_is_connected_to_j:
                            ij_dQ[(j,k)] = ij_dQ[(i,k)] - 2*a[j]*a[k]
                    if (j,k) in ij_dQ:
                        if not k_is_connected_to_i and k_is_connected_to_j:
                            ij_dQ[(j,k)] = ij_dQ[(j,k)] - 2*a[i]*a[k]
                    if i == c2 or j == c2:
                        ij_dQ[(i,j)] = 0.
            a[c1] += a[c2]
            a[c2] = 0.
            q += delta_q

        return mod_list

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
        self.number_of_communities = self.number_of_nodes()
        self.nodes_community = {n : n for n in range(self.number_of_nodes())}

    def _check_parameters(self):
        if self.blocks != self.p.shape[0]:
            raise ValueError("The number of blocks and the link probability matrix dimension do not match")


if __name__ == "__main__":

    import networkx as nx
    import pylab as plt
    import time

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
    print(f"stochastic_block_model (networkx) = {tf-ti} sec")

    perms = np.random.permutation(range(0,320))
    partitions = [np.array([c], dtype=int) for c in range(blocks_networkx.number_of_nodes())]
    #partitions = [range(0,100), range(100,160), range(160,320)]

    ti = time.time()
    mod = nx.algorithms.community.modularity(blocks_networkx, partitions)
    tf = time.time()
    print(f"modularity (networkx) = {mod}")
    print(f"time (networkx) = {tf-ti} sec")

    ti = time.time()
    blocks_numpy = Random_Blocks(blocks_sizes, prob_matrix,
                                 check_parameters=False, check_adjacency=False)
    tf = time.time()
    print(f"\nstochastic_block_model (numpy) = {tf-ti} sec")
    ti = time.time()
    mod = blocks_numpy.modularity()
    tf = time.time()
    print(f"modularity (numpy) = {mod}")
    print(f"time (numpy) = {tf-ti} sec")
    fig, ax = plt.subplots(figsize=(8,8))
    blocks_numpy.show(ax)
    plt.show()

    #modularity_list = blocks_numpy.clustering()