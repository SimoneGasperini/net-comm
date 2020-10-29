import pylab as plt
from unetwork import UndirectedNetwork
from visualization import draw_communities_graph, draw_communities_barplot

def count_connections(net):
    # counts the connections in net starting from edge_dict (each edge twice)
    edges = 0
    for node in net.edge_dict:
        edges += len(net.edge_dict[node])
    return edges

github_net = UndirectedNetwork.fromfile(filename="data/github.txt")

assert github_net.number_of_nodes == len(github_net.edge_dict)
assert github_net.number_of_edges*2 == count_connections(github_net)


mod1 = github_net.modularity()
print(f"number of communities before clustering = {github_net.number_of_communities}")
print(f"modularity before clustering = {mod1}")
github_net.clustering()
mod2 = github_net.modularity()
print(f"number of communities after clustering = {github_net.number_of_communities}")
print(f"modularity after clustering = {mod2}")


fig, ax = plt.subplots(figsize=(8,8))
draw_communities_graph(github_net, ax, min_size=10, scale="root")
plt.show()

fig, ax = plt.subplots(figsize=(8,8))
draw_communities_barplot(github_net, ax, min_size=10, scale="log")
plt.show()
