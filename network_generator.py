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

    def __init__(self, n, m, edge_dict, build_adjacency=False):
        self.number_of_nodes = n
        self.number_of_edges = m
        self.edge_dict = edge_dict
        if not build_adjacency:
            self.adjacency = None

    @classmethod
    def fromfile(cls, filename):
        edge_dict = {}
        file = open(filename, mode="r")
        all_lines = file.readlines()
        for i in trange(len(all_lines), desc="Reading file"):
            line = all_lines[i]
            if line.startswith("#"):
                continue
            if line.startswith("Nodes"):
                word, n = line.split()
                n = int(n)
                continue
            if line.startswith("Edges"):
                word, m = line.split()
                m = int(m)
                continue
            node1, node2 = line.split()
            node1, node2 = int(node1), int(node2)
            if node1 >= n or node2 >= n:
                continue
            edge_dict.setdefault(node1,[]).append(node2)
        file.close()
        return cls(n,m,edge_dict)

    def _edges_in(self, comm):
        edges_in_comm = []
        for u in comm:
            if u in self.edge_dict:
                for v in comm:
                    if v in self.edge_dict[u]:
                        edges_in_comm.append((u,v))
        return edges_in_comm

    def degrees_of_nodes(self):
        degree_dict = {node : 0 for node in range(self.number_of_nodes)}
        nodes_list = list(self.edge_dict.keys())
        for node in nodes_list:
            degree_dict[node] = len(self.edge_dict[node])
        return degree_dict

    def modularity(self, partitions=None):
        communities = partitions if partitions is not None else self.partitions
        degree = self.degrees_of_nodes()
        deg_sum = sum(degree.values())
        m = deg_sum/2
        norm = 1/(deg_sum**2)
        def community_contribution(community):
            L_c = sum(0.5 for u,v in self._edges_in(community))
            degree_sum = sum(degree[u] for u in community)
            return L_c/m - degree_sum*degree_sum*norm
        return sum(map(community_contribution, communities))

    def clustering(self):
        n = self.number_of_nodes
        m = self.number_of_edges
        q0 = 1./(2.*m)
        degrees = self.degrees_of_nodes()
        k = [degrees[i] for i in range(n)]
        communities = {i : frozenset([i]) for i in range(n)}
        a = [k[i]*q0 for i in range(n)]
        dq_matrix = {
        i : {
            j : [2.*q0 - 2.*k[i]*k[j]*q0*q0, i, j]
            for j in self.edge_dict[i]
            }
            for i in self.edge_dict
        }

        for _ in trange(n-1, desc="Clustering in progress"):
            delta_q = 0.
            for u in dq_matrix:
                mass = max(dq_matrix[u].values())
                if mass[0] > delta_q:
                    delta_q, i, j = mass
            if delta_q <= 0:
                break
            communities[j] = frozenset(communities[i] | communities[j])
            del communities[i]
            i_set = set(dq_matrix[i].keys()) if i in dq_matrix else set([])
            j_set = set(dq_matrix[j].keys()) if j in dq_matrix else set([])
            all_set = (i_set | j_set) - {i,j}
            both_set = i_set & j_set
            for k in all_set:
                if k in both_set:
                    dq_jk = dq_matrix[j][k][0] + dq_matrix[i][k][0]
                elif k in j_set:
                    dq_jk = dq_matrix[j][k][0] - 2.*a[i]*a[k]
                else:
                    dq_jk = dq_matrix[i][k][0] - 2.*a[j]*a[k]
                if j in dq_matrix: dq_matrix[j][k] = [dq_jk, j, k]
                if k in dq_matrix: dq_matrix[k][j] = [dq_jk, k, j]
            for k in dq_matrix[i]:
                if k in dq_matrix:
                    if i in dq_matrix[k]:
                        del dq_matrix[k][i]
            del dq_matrix[i]
            a[j] += a[i]
            a[i] = 0

        communities = [frozenset([node for node in comm]) for comm in communities.values()]
        self.number_of_partitions = len(communities)
        self.partitions = communities

    def draw_nx(self, ax, col_communities=False):
        if self.adjacency is None:
            raise NotImplementedError("Draw method is not supported for very large networks")
        netx = nx.Graph(self.adjacency)
        if col_communities and self.number_of_partitions > 1:
            p = self.number_of_partitions
            col = np.linspace(0,1,p)
            colors = np.empty(self.number_of_nodes)
            for node in range(self.number_of_nodes):
                for i,p in enumerate(self.partitions):
                    if node in p:
                        colors[node] = col[i]
        else:
            colors = "#1f78b4"
        nx.draw(netx, ax=ax, width=0.2, node_size=50, node_color=colors, cmap="viridis")

    def show(self, ax, show_communities=False):
        if self.adjacency is None:
            raise NotImplementedError("Show method is not supported for very large networks")
        if show_communities and self.number_of_partitions > 1:
            perms = np.empty(self.number_of_nodes, dtype=int)
            i = 0
            for comm in self.partitions:
                for node in comm:
                    perms[i] = node
                    i += 1
            adj = self.adjacency[perms,:][:,perms]
            ax.imshow(adj, cmap="binary")
        else:
            ax.imshow(self.adjacency, cmap="binary")


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


class RandomBlocks(UndirectedNetwork):

    def __init__(self, sizes, p_matrix, seed=None):
        self.blocks = sizes.size
        self.block_sizes = sizes
        self.p = prob_matrix
        if seed is not None:
            np.random.seed(seed)
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
        self.adjacency = A[perms,:][:,perms]
        self.number_of_partitions = 1
        self.partitions = [range(n)]
        m = int(np.sum(self.adjacency)*0.5)
        node1_list, node2_list = np.nonzero(self.adjacency)
        edge_dict = {}
        for node1,node2 in zip(node1_list,node2_list):
            edge_dict.setdefault(node1,[]).append(node2)
        UndirectedNetwork.__init__(self,n,m,edge_dict, build_adjacency=True)


if __name__ == "__main__":

    import pylab as plt
    import time

    blocks = 3
    n1=1000; n2=800; n3=900
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
    random_blocks = RandomBlocks(blocks_sizes, prob_matrix)
    tf = time.time()
    print(f"\ntime for RBmodel generation = {tf-ti} sec", flush=True)
    fig, ax = plt.subplots(figsize=(8,8))
    random_blocks.show(ax)
    plt.show()

    random_blocks.clustering()
    tf = time.time()
    print(f"\ntime for clustering = {tf-ti}", flush=True)
    fig, ax = plt.subplots(figsize=(8,8))
    random_blocks.draw_nx(ax, col_communities=True)
    plt.show()
    fig, ax = plt.subplots(figsize=(8,8))
    random_blocks.show(ax, show_communities=True)
    plt.show()