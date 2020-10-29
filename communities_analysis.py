import pylab as plt
from unetwork import UndirectedNetwork
from visualization import draw_communities_graph, draw_communities_barplot



def analyze(filename):

    net = UndirectedNetwork.fromfile(filename)

    if net.number_of_nodes != len(net.edge_dict):
        raise ValueError("Something went wrong in reading the input file!")
    if net.number_of_edges*2 != sum([len(net.edge_dict[node]) for node in net.edge_dict]):
        raise ValueError("Something went wrong in reading the input file!")

    mod1 = net.modularity()
    print(f"\nnumber of communities before clustering = {net.number_of_communities}")
    print(f"modularity before clustering = {mod1}")
    net.clustering()
    mod2 = net.modularity()
    print(f"number of communities after clustering = {net.number_of_communities}")
    print(f"modularity after clustering = {mod2}")

    print("\nDRAW COMMUNITIES GRAPH")
    while True:
        min_size = input("\tEnter the community minimum size >>> ")
        scale = input("\tEnter the scale for nodes size (linear/root/log) >>> ")
        fig, ax = plt.subplots(figsize=(8,8))
        draw_communities_graph(net, ax, min_size=int(min_size), scale=scale)
        plt.show()
        answer = input("Do you want to plot it again? (y/n) >>> ")
        if answer == "n": break

    print("\nDRAW COMMUNITIES BARPLOT")
    while True:
        min_size = input("\tEnter the community minimum size >>> ")
        scale = input("\tEnter the scale for nodes size (linear/root/log) >>> ")
        fig, ax = plt.subplots(figsize=(8,8))
        draw_communities_barplot(net, ax, min_size=int(min_size), scale=scale)
        plt.show()
        answer = input("Do you want to plot it again? (y/n) >>> ")
        if answer == "n": break
