import pylab as plt
from unetwork import UndirectedNetwork
from visualization import draw_communities_graph, draw_communities_barplot


github_net = UndirectedNetwork.fromfile(filename="data/github.txt")

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