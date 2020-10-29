import pylab as plt
from unetwork import UndirectedNetwork
from visualization import draw_communities_graph, draw_communities_barplot

def count_connections(net):
    # counts the connections in net starting from edge_dict (each edge twice)
    edges = 0
    for node in net.edge_dict:
        edges += len(net.edge_dict[node])
    return edges

email_net = UndirectedNetwork.fromfile(filename="data/email.txt")

assert email_net.number_of_nodes == len(email_net.edge_dict)
assert email_net.number_of_edges*2 == count_connections(email_net)


mod1 = email_net.modularity()
print(f"number of communities before clustering = {email_net.number_of_communities}")
print(f"modularity before clustering = {mod1}")
email_net.clustering()
mod2 = email_net.modularity()
print(f"number of communities after clustering = {email_net.number_of_communities}")
print(f"modularity after clustering = {mod2}")


fig, ax = plt.subplots(figsize=(8,8))
draw_communities_graph(email_net, ax, min_size=21, scale="root")
plt.show()

fig, ax = plt.subplots(figsize=(8,8))
draw_communities_barplot(email_net, ax, min_size=20, scale="log")
plt.show()
