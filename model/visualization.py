import numpy as np
import pylab as plt
import networkx as nx

plt.style.use("seaborn-paper")



def draw (unet, partition=None, ax=None, cmap="Spectral"):

    if ax is None:
        ax = plt.gca()

    netx = nx.to_networkx_graph(unet.edge_dict)

    if partition is not None:

        col = np.linspace(0, 1, num=unet.number_of_communities)

        node_color = np.array([col[i] for node in range(unet.number_of_nodes)
                               for i, comm in enumerate(partition)
                               if node in comm])
    else:

        node_color = "#1f78b4"

    nx.draw(netx, ax=ax, width=0.1, node_size=20, node_color=node_color, cmap=cmap)



def draw_communities_graph (unet, partition, ax=None,
                            min_size=1, scale_size=1e6, scale="linear", cmap="plasma"):

    if ax is None:
        ax = plt.gca()
    
    communities = [comm for comm in partition if len(comm) >= min_size]
    n = len(communities)

    matrix = np.eye(n, dtype=int)

    for i in range(0, n-1):

        for j in range(i+1, n):

            comm_i = communities[i]
            comm_j = communities[j]

            if len(unet._edges_between_comms(comm_i,comm_j)) > 0:
                matrix[i][j] = matrix[j][i] = 1

    node_size = np.array([len(communities[i]) / unet.number_of_nodes * scale_size
                          for i in range(n)])

    if scale == "root": node_size = np.sqrt(node_size)
    if scale == "log": node_size = np.log(node_size)

    netx = nx.Graph(matrix)

    nx.draw(netx, ax=ax, node_size=node_size, vmin=0, vmax=1, cmap=cmap)



def draw_communities_barplot (unet, partition, ax=None,
                              min_size=1, scale="linear"):

    if ax is None:
        ax = plt.gca()

    sizes = np.array([len(comm) for comm in partition if len(comm) >= min_size])

    ax.bar(x = range(len(sizes)), height = np.sort(sizes))

    if scale == "root":
        ax.set_yscale("function", functions=(lambda x : np.sqrt(x),
                                             lambda x : np.square(x)))
    else:
        plt.yscale(scale)

    ax.set_xlabel("community", fontsize=16)
    ax.set_ylabel("number of nodes", fontsize=16)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    for tx in ax.xaxis.get_major_ticks():
        tx.label.set_fontsize(12)

    for ty in ax.yaxis.get_major_ticks():
        ty.label.set_fontsize(12)
