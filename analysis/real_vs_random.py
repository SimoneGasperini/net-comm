import sys
sys.path.append("..")

from model.unet import UndirectedNetwork
from model.erdosrenyi import ErdosRenyi


real_network = UndirectedNetwork.fromfile("../data/General_Relativity.txt")

part = real_network.clustering()
mod = real_network.modularity(part)
print(f"\nmod_real = {mod}", flush=True)

n = real_network.number_of_nodes
m = real_network.number_of_edges
p = (2. * m) / n**2

random_network = ErdosRenyi(n, p)

part = random_network.clustering()
mod = random_network.modularity(part)
print(f"\nmod_random = {mod}", flush=True)
