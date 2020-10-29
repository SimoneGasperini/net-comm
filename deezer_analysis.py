import pylab as plt
from unetwork import UndirectedNetwork
from visualization import draw_communities_graph, draw_communities_barplot

def count_connections(net):
    # counts the connections in net starting from edge_dict (each edge twice)
    edges = 0
    for node in net.edge_dict:
        edges += len(net.edge_dict[node])
    return edges

deezer_net = UndirectedNetwork.fromfile(filename="data/deezer.txt")

assert deezer_net.number_of_nodes == len(deezer_net.edge_dict)
assert deezer_net.number_of_edges*2 == count_connections(deezer_net)


mod1 = deezer_net.modularity()
print(f"number of communities before clustering = {deezer_net.number_of_communities}")
print(f"modularity before clustering = {mod1}")
deezer_net.clustering()
mod2 = deezer_net.modularity()
print(f"number of communities after clustering = {deezer_net.number_of_communities}")
print(f"modularity after clustering = {mod2}")


fig, ax = plt.subplots(figsize=(8,8))
draw_communities_graph(deezer_net, ax, min_size=10, scale="root")
plt.show()

fig, ax = plt.subplots(figsize=(8,8))
draw_communities_barplot(deezer_net, ax, min_size=10, scale="log")
plt.show()
