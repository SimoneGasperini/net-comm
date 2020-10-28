import numpy as np
import pylab as plt
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
        '''
        edge_dict is a dict of int mapped to lists with the following structure:
            key --> node1
            value --> list of all nodes2 connected to node1
        _each node is represented as an integer number(node_id)
        _each connection is actually counted twice(node1--node2, node2--node1)
        _if node1 has no connections then its corresponding value is [](empty list)
        '''
        self.number_of_nodes = n
        self.number_of_edges = m
        self.edge_dict = edge_dict

        self.number_of_communities = 1
        self.partition = [set(range(n))]

        if not build_adjacency:
            self.adjacency = None

    @classmethod
    def fromfile(cls, filename, comments="#"):
        edge_dict = {}
        file = open(filename, mode="r")
        all_lines = file.readlines()
        m = 0
        for i in trange(len(all_lines), desc="Reading file"):
            line = all_lines[i]
            if line.startswith(comments):
                continue
            m += 1
            node1, node2 = line.split()
            node1, node2 = int(node1), int(node2)
            edge_dict.setdefault(node1,[]).append(node2)
            edge_dict.setdefault(node2,[]).append(node1)
        n = len(edge_dict)
        file.close()
        return cls(n,m,edge_dict)

    def _edges_within(self, comm):
        return [(u,v) for u in comm for v in comm & set(self.edge_dict[u])]

    def _edges_between(self, comm1, comm2):
        return [(u,v) for u in comm1 for v in comm2 & set(self.edge_dict[u])]

    def _check_clustering(self):
        if self.number_of_communities != self.blocks:
            print("\nThe clustering algorithm failed -->", flush=True)
            print(f"\t# of original blocks = {self.blocks}", flush=True)
            print(f"\t# of communities found = {self.number_of_communities}", flush=True)

    def degrees_of_nodes(self):
        degree_dict = {node : 0 for node in range(self.number_of_nodes)}
        nodes_list = list(self.edge_dict.keys())
        for node in nodes_list:
            degree_dict[node] = len(self.edge_dict[node])
        return degree_dict

    def modularity(self, partition=None):
        communities = partition if partition is not None else self.partition
        degree = self.degrees_of_nodes()
        deg_sum = sum(degree.values())
        m = deg_sum/2
        norm = 1/(deg_sum**2)
        def community_contribution(community):
            L_c = sum(0.5 for u,v in self._edges_within(community))
            degree_sum = sum(degree[u] for u in community)
            return L_c/m - degree_sum*degree_sum*norm
        return sum(map(community_contribution, communities))

    def clustering(self, return_modularity=False, check_result=False):
        n = self.number_of_nodes
        m = self.number_of_edges
        q0 = 1./(2.*m)
        degrees = self.degrees_of_nodes()
        k = [degrees[i] for i in range(n)]
        communities = {i : set([i]) for i in range(n)}
        a = [k[i]*q0 for i in range(n)]
        dq_matrix = {
        i : {
            j : [2.*q0 - 2.*k[i]*k[j]*q0*q0, i, j]
            for j in self.edge_dict[i]
            }
            for i in range(n)
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
            communities[j] = communities[i] | communities[j]
            del communities[i]
            i_set = set(dq_matrix[i].keys())
            j_set = set(dq_matrix[j].keys())
            all_set = (i_set | j_set) - {i,j}
            both_set = i_set & j_set
            for k in all_set:
                if k in both_set:
                    dq_jk = dq_matrix[j][k][0] + dq_matrix[i][k][0]
                elif k in j_set:
                    dq_jk = dq_matrix[j][k][0] - 2.*a[i]*a[k]
                else:
                    dq_jk = dq_matrix[i][k][0] - 2.*a[j]*a[k]
                dq_matrix[j][k] = [dq_jk, j, k]
                dq_matrix[k][j] = [dq_jk, k, j]
            for k in dq_matrix[i].keys():
                del dq_matrix[k][i]
            del dq_matrix[i]
            a[j] += a[i]
            a[i] = 0

        communities = [set(comm) for comm in communities.values()]
        self.number_of_communities = len(communities)
        self.partition = communities
        if check_result: self._check_clustering()
        return mod_list if return_modularity else None

    def show(self, ax=None, cmap="Spectral", show_communities=False):
        if self.adjacency is None:
            raise NotImplementedError("Show method is not supported for very large networks")
        if ax is None: ax = plt.gca()
        if show_communities:
            perms = np.array([node for comm in self.partition for node in comm])
            adj = self.adjacency[perms,:][:,perms]
            ax.imshow(adj, cmap=cmap)
        else:
            ax.imshow(self.adjacency, cmap=cmap)