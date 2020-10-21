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
        comm = set(comm)
        u_set = comm & set(self.edge_dict.keys())
        edges_in_comm = []
        for u in u_set:
            v_set = comm & set(self.edge_dict[u])
            for v in v_set:
                edges_in_comm.append((u,v))
        return edges_in_comm

    def _check_clustering(self):
        if self.number_of_partitions != self.blocks:
            print("\nThe clustering algorithm failed -->", flush=True)
            print(f"\t# of original blocks = {self.blocks}", flush=True)
            print(f"\t# of communities found = {self.number_of_partitions}", flush=True)

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

    def clustering(self, return_modularity=False, check_result=False):
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
        
        if return_modularity:
            q = self.modularity()
            mod_list = [q]
        end = False
        for _ in trange(n-1, desc="Clustering in progress"):
            if end: continue
            delta_q = 0.
            for u in dq_matrix:
                lst = list(dq_matrix[u].values())
                if len(lst) == 0: continue
                mass = max(lst)
                if mass[0] > delta_q:
                    delta_q, i, j = mass
            if delta_q <= 0:
                end = True
                continue
            if return_modularity:
                q += delta_q
                mod_list.append(q)
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
        if check_result: self._check_clustering()
        return mod_list if return_modularity else None

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
            ax.imshow(adj)
        else:
            ax.imshow(self.adjacency)