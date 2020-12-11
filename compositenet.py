import numpy as np
from numpy.random import randint
from unetwork import UndirectedNetwork


class CompositeNetwork(UndirectedNetwork):

    def __init__(self, unetworks, edge_matrix, seed=None):

        if not isinstance(unetworks, np.ndarray): unetworks = np.array(unetworks)
        if not isinstance(edge_matrix, np.ndarray): edge_matrix = np.array(edge_matrix)

        if seed is not None: np.random.seed(seed)

        nodes_cumsum = np.cumsum([unet.number_of_nodes for unet in unetworks])
        edges_cumsum = np.cumsum([unet.number_of_edges for unet in unetworks])

        edge_dict = {}
        for i, unet in enumerate(unetworks):
            n = 0 if i == 0 else nodes_cumsum[i-1]
            current = unet._relabeled_dict(starting_id=n)
            edge_dict = {**edge_dict, **current}

        n = 0; m = 0
        edges_new = {}
        for i in range(unetworks.size-1):
            for j in range(i+1, unetworks.size):
                low_i = 0 if i == 0 else nodes_cumsum[i-1]
                high_i = nodes_cumsum[i]
                low_j = nodes_cumsum[j-1]
                high_j = nodes_cumsum[j]
                for _ in range(edge_matrix[i][j]):
                    edges_new.setdefault(randint(low_i, high_i), []).append(randint(low_j, high_j))

        edge_dict = {**edge_dict, **edges_new}
        n = nodes_cumsum[-1]
        m = edges_cumsum[-1] + np.sum(edge_matrix)

        UndirectedNetwork.__init__(self,n,m,edge_dict, build_adjacency=True)