import numpy as np
import pylab as plt
from tqdm import trange



class UndirectedNetwork:


    def __init__ (self, n, m, edge_dict, adjacency=None):

        self.number_of_nodes = n
        self.number_of_edges = m
        self.edge_dict = edge_dict
        self.adjacency = adjacency

        # build one single community containing all the nodes
        self.number_of_communities = 1
        self.partition = [set(range(n))]


    @classmethod
    def fromfile (cls, filename, comments='#'):

        edge_dict = {}

        file = open(filename, mode='r')
        all_lines = file.readlines()

        m = 0

        for i in trange(len(all_lines), desc="Reading file"):

            line = all_lines[i]
            if line.startswith(comments):
                continue
            m += 1

            node1, node2 = line.split()
            node1, node2 = int(node1), int(node2)
            edge_dict.setdefault(node1, []).append(node2)
            edge_dict.setdefault(node2, []).append(node1)

        n = len(edge_dict)

        file.close()

        return cls (n=n, m=m, edge_dict=edge_dict)


    def _relabeled_dict (self, n):

        new_dict = { node_i + n : [node_j + n for node_j in self.edge_dict[node_i]]
                                  for node_i in self.edge_dict }
        return new_dict

    def _edges_within_comm (self, comm):

        return [(u, v) for u in comm for v in comm & set(self.edge_dict[u])]


    def _edges_between_comm (self, comm1, comm2):

        return [(u, v) for u in comm1 for v in comm2 & set(self.edge_dict[u])]


    def _totdegree_of_comm (self, comm):

        return np.sum([len(self.edge_dict[u]) for u in comm])


    def _check_clustering (self):

        if self.number_of_communities != self.blocks:

            print("\nThe clustering algorithm failed -->", flush=True)
            print(f"\t# of original blocks = {self.blocks}", flush=True)
            print(f"\t# of communities found = {self.number_of_communities}", flush=True)


    def degrees_of_nodes (self):

        return { u : len(self.edge_dict[u]) for u in range(self.number_of_nodes) }


    def modularity_of_comm (self, comm):

        m = self.number_of_edges
        m_c = len(self._edges_within_comm(comm)) * 0.5
        d_c = self._totdegree_of_comm(comm)

        return (m_c / m) - (d_c / (2. * m))**2


    def modularity (self, partition=None):

        communities = partition if partition is not None else self.partition

        return np.sum([self.modularity_of_comm(comm) for comm in communities])


    def clustering (self, check_result=False):

        n = self.number_of_nodes
        m = self.number_of_edges

        q0 = 1. / (2. * m)
        degrees = self.degrees_of_nodes()
        k = [degrees[i] for i in range(n)]

        communities = {i : set([i]) for i in range(n)}

        a = [k[i]*q0 for i in range(n)]

        dq_matrix = { i : { j : [(2. * q0) - (2. * k[i] * k[j] * q0 * q0), i, j]
                                for j in self.edge_dict[i] }
                          for i in range(n) }

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
                    dq_jk = dq_matrix[j][k][0] - (2. * a[i] * a[k])
                else:
                    dq_jk = dq_matrix[i][k][0] - (2. * a[j] * a[k])
                dq_matrix[j][k] = [dq_jk, j, k]
                dq_matrix[k][j] = [dq_jk, k, j]

            i_neighbors = set(dq_matrix[i].keys())
            for k in i_neighbors:
                del dq_matrix[k][i]
            del dq_matrix[i]

            a[j] += a[i]
            a[i] = 0

        communities = [set(comm) for comm in communities.values()]
        self.number_of_communities = len(communities)
        self.partition = communities

        if check_result:
            self._check_clustering()


    def show (self, ax=None, cmap="Spectral", show_communities=False):

        if self.adjacency is None:
            raise NotImplementedError("Show method is not supported")

        if ax is None:
            ax = plt.gca()

        if show_communities:
            perms = np.array([node for comm in self.partition for node in comm])
            adjacency = self.adjacency[perms,:][:,perms]

            ax.imshow(adjacency, cmap=cmap)

        else:
            ax.imshow(self.adjacency, cmap=cmap)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
