import numpy as np

from model.unetwork import UndirectedNetwork



class CompositeNetwork (UndirectedNetwork):


    def __init__ (self, unetworks, edge_matrix, seed=None):

        # transform parameters in numpy arrays
        if not isinstance(unetworks, np.ndarray):
            unetworks = np.array(unetworks)

        if not isinstance(edge_matrix, np.ndarray):
            edge_matrix = np.array(edge_matrix)

        # check type and value of parameters
        self._check_parameters(unetworks, edge_matrix)

        # set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # compute number of nodes and edges
        n = np.sum([unet.number_of_nodes for unet in unetworks])
        m = np.sum([unet.number_of_edges for unet in unetworks]) + np.sum(edge_matrix)

        # build edges dictionary
        edge_dict = self._compute_edge_dict(unetworks, edge_matrix)

        UndirectedNetwork.__init__(self, n=n, m=m, edge_dict=edge_dict, adjacency=None)


    def _check_parameters (self, unetworks, edge_matrix):

        if not np.all([isinstance(unet, UndirectedNetwork) for unet in unetworks]):
            raise TypeError("All the elements in 'unetworks' must be UndirectedNetwork")

        if not np.all((edge_matrix % 1) == 0):
            raise TypeError("All the elements in 'edge_matrix' must be ints")

        if not np.all(edge_matrix >= 0):
            raise ValueError("All the elements in 'edge_matrix' must be >= 0")

        if not np.all(edge_matrix == edge_matrix.transpose()):
            raise ValueError("'edge_matrix' must be symmetric (undirected)")


    def _compute_edge_dict (self, unetworks, edge_matrix):

        nodes_cumsum = np.cumsum([unet.number_of_nodes for unet in unetworks])

        edge_dict = {}

        for i, unet in enumerate(unetworks):

            n = 0 if i == 0 else nodes_cumsum[i-1]
            current = unet._relabeled_dict(n)
            edge_dict = {**edge_dict, **current}

        edges_new = {}

        for i in range(unetworks.size-1):

            for j in range(i+1, unetworks.size):

                low_i = 0 if i == 0 else nodes_cumsum[i-1]
                high_i = nodes_cumsum[i]
                low_j = nodes_cumsum[j-1]
                high_j = nodes_cumsum[j]

                for _ in range(edge_matrix[i][j]):

                    node_i = np.random.randint(low_i, high_i)
                    node_j = np.random.randint(low_j, high_j)
                    edges_new.setdefault(node_i, []).append(node_j)

        return {**edge_dict, **edges_new}
