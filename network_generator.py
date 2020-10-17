import numpy as np
import networkx as nx
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
        self.edge_dict = self._edge_dict()
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

    def _edge_dict(self):
        node1_list, node2_list = np.nonzero(self.A)
        edge_dict = {}
        for node1,node2 in zip(node1_list,node2_list):
            edge_dict.setdefault(node1,[]).append(node2)
        return edge_dict

    def _edges_in(self, comm):
        return ((u,v) for (u,v) in self.edge_list if u in comm and v in comm)

    def number_of_nodes(self):
        return self.A.shape[0]

    def number_of_edges(self):
        return int(np.sum(self.A)*0.5)

    def degrees_of_nodes(self):
        n = self.number_of_nodes()
        return {node : np.sum(self.A[node,:]) for node in range(n)}

    def modularity_nx(self, partitions=None):
        communities = partitions if partitions is not None else self.partitions
        return nx.algorithms.community.modularity(self.netx, communities)

    def clustering_nx(self):
        communities = nx.algorithms.community.greedy_modularity_communities(self.netx)
        self.number_of_partitions = len(communities)
        self.partitions = communities

    def modularity(self, partitions=None):
        communities = partitions if partitions is not None else self.partitions
        degree = self.degrees_of_nodes()
        deg_sum = sum(degree.values())
        m = deg_sum/2
        norm = 1/(deg_sum**2)
        def community_contribution(community):
            comm = set(community)
            L_c = sum(1 for u,v in self._edges_in(comm) if v in comm)
            degree_sum = sum(degree[u] for u in comm)
            return L_c/m - degree_sum*degree_sum*norm
        return sum(map(community_contribution, communities))

    def clustering(self):
        n = self.number_of_nodes()
        m = self.number_of_edges()
        q0 = 1./(2.*m)
        degrees = self.degrees_of_nodes()
        k = np.array([degrees[i] for i in range(n)])
        communities = {i : set([i]) for i in range(n)}
        partitions = [[node for node in comm] for comm in communities.values()]
        q = self.modularity(partitions)
        a = np.array([k[i]*q0 for i in range(n)])
        dq_matrix = np.zeros(shape=(n,n))
        for i in range(n):
            for j in self.edge_dict[i]:
                    dq_matrix[i][j] = 2.*q0 - 2.*k[i]*k[j]*q0*q0

        for _ in trange(n-1, desc="Clustering in progress"):
            delta_q = np.max(dq_matrix)
            if delta_q <= 0:
                continue
            i, j = np.where(dq_matrix == delta_q)
            if isinstance(i, np.ndarray): i = i[0]
            if isinstance(j, np.ndarray): j = j[0]
            communities[j] = set(communities[i] | communities[j])
            communities[i] = set([])
            i_set = set(np.nonzero(dq_matrix[i,:])[0])
            j_set = set(np.nonzero(dq_matrix[j,:])[0])
            all_set = (i_set | j_set) - {i,j}
            both_set = i_set & j_set
            for k in all_set:
                if k in both_set:
                    dq_jk = dq_matrix[j][k] + dq_matrix[i][k]
                elif k in j_set:
                    dq_jk = dq_matrix[j][k] - 2.*a[i]*a[k]
                else:
                    dq_jk = dq_matrix[i][k] - 2.*a[j]*a[k]
                dq_matrix[j][k] = dq_jk
                dq_matrix[k][j] = dq_jk
            dq_matrix[i,:] = 0
            dq_matrix[:,i] = 0
            a[j] += a[i]
            a[i] = 0
            q += delta_q

        communities = [set([node for node in comm]) for comm in communities.values()]
        communities = [c for c in communities if len(c) > 0]
        self.number_of_partitions = len(communities)
        self.partitions = communities

    def draw_nx(self, ax, col_communities=False):
        if col_communities and self.number_of_partitions > 1:
            p = self.number_of_partitions
            col = np.linspace(0,1,p)
            colors = np.empty(self.number_of_nodes())
            for node in range(self.number_of_nodes()):
                for i,p in enumerate(self.partitions):
                    if node in p:
                        colors[node] = col[i]
        else:
            colors = "#1f78b4"
        nx.draw(self.netx, ax=ax, width=0.2, node_size=50, node_color=colors, cmap="viridis")

    def show(self, ax, show_communities=False):
        if show_communities and self.number_of_partitions > 1:
            perms = np.empty(self.number_of_nodes(), dtype=int)
            i = 0
            for comm in self.partitions:
                for node in comm:
                    perms[i] = node
                    i += 1
            adj = self.A[perms,:][:,perms]
            ax.imshow(adj, cmap="binary")
        else:
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
        A = np.triu(A) + np.triu(A).T - np.diag(2*np.diag(A))
        perms = np.random.permutation(range(n))
        A = A[perms,:][:,perms]
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
    n1=200; n2=100; n3=260
    n = n1+n2+n3
    blocks_sizes = np.array([n1, n2, n3])
    prob_matrix = np.zeros(shape=(blocks,blocks))
    for i in range(blocks-1):
        for j in range(i+1, blocks):
            prob_matrix[i][j] = 0.005
    prob_matrix += prob_matrix.T
    for k in range(blocks):
        prob_matrix[k][k] = 0.2

    ti = time.time()
    random_blocks = Random_Blocks(blocks_sizes, prob_matrix,
                                 check_parameters=True, check_adjacency=True)
    tf = time.time()
    print(f"\ntime for RBmodel generation = {tf-ti} sec", flush=True)
    fig, ax = plt.subplots(figsize=(8,8))
    random_blocks.show(ax)
    plt.show()

    mod1 = random_blocks.modularity()
    print(f"\nmodularity Q(before clustering) = {mod1}", flush=True)
    ti = time.time()
    random_blocks.clustering()
    tf = time.time()
    print(f"\ntime for clustering = {tf-ti}", flush=True)
    mod2 = random_blocks.modularity()
    print(f"modularity Q(after clustering) = {mod2}", flush=True)
    fig, ax = plt.subplots(figsize=(8,8))
    random_blocks.draw_nx(ax, col_communities=True)
    plt.show()
    fig, ax = plt.subplots(figsize=(8,8))
    random_blocks.show(ax, show_communities=True)
    plt.show()