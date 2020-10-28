import numpy as np
import pylab as plt
import networkx as nx

def draw(unet, ax=None, cmap="Spectral", color_communities=False):
    if ax is None: ax = plt.gca()
    node_color = "#1f78b4"
    netx = nx.to_networkx_graph(unet.edge_dict)
    if color_communities:
        c = unet.number_of_communities
        col = np.linspace(0,1,c)
        node_color = np.array([col[i] for node in range(unet.number_of_nodes)
                               for i, comm in enumerate(unet.partition)
                               if node in comm])
    nx.draw(netx, ax=ax, width=0.1, node_size=20, node_color=node_color, cmap=cmap)


def draw_communities_graph(unet, ax=None, min_size=1, scale_size=1e6, scale="linear", cmap="plasma"):
    if ax is None: ax = plt.gca()
    communities = [comm for comm in unet.partition if len(comm) >= min_size]
    n = len(communities)
    matrix = np.eye(n, dtype=int)
    for i in range(0, n-1):
        for j in range(i+1, n):
            comm_i = communities[i]
            comm_j = communities[j]
            if len(unet._edges_between(comm_i,comm_j)) > 0:
                matrix[i][j] = matrix[j][i] = 1
    node_size = np.array([len(communities[i]) / unet.number_of_nodes * scale_size
                          for i in range(n)])
    if scale == "root": node_size = np.sqrt(node_size)
    if scale == "log": node_size = np.log(node_size)
    netx = nx.Graph(matrix)
    nx.draw(netx, ax=ax, width=0.2, node_size=node_size, vmin=0, vmax=1, cmap=cmap)


def draw_communities_barplot(unet, ax=None, min_size=1, scale="linear"):
    if ax is None: ax = plt.gca()
    sizes = np.array([len(comm) for comm in unet.partition if len(comm) >= min_size])
    ax.bar(x = range(len(sizes)), height = np.sort(sizes))
    plt.yscale(scale)
    ax.set_xlabel("community (size >= min_size)")
    ax.set_ylabel("size (number of nodes)")