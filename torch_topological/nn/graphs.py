"""Layers for topological data analysis based on graphs."""

from torch_geometric.data import Batch
from torch_geometric.data import Data

from torch_geometric.loader import DataLoader

from torch_geometric.utils import erdos_renyi_graph

import torch

B = 64
N = 100
p = 0.2

data_list = [
    Data(x=torch.rand(N, 1), edge_index=erdos_renyi_graph(N, p), num_nodes=N)
    for i in range(B)
]

loader = DataLoader(data_list, batch_size=8)

for index, batch in enumerate(loader):
    vertex_slices = torch.Tensor(batch._slice_dict['x']).long()
    edge_slices = torch.Tensor(batch._slice_dict['edge_index']).long()
    
    print(vertex_slices)
    print(edge_slices)


#def calculate_persistent_homology(G, k=3):
#    """Calculate persistent homology of graph clique complex."""
#    st = gd.SimplexTree()
#
#    for v, w in G.nodes(data=True):
#        weight = w["curvature"]
#        st.insert([v], filtration=weight)
#
#    for u, v, w in G.edges(data=True):
#        weight = w["curvature"]
#        st.insert([u, v], filtration=weight)
#
#    st.make_filtration_non_decreasing()
#    st.expansion(k)
#    persistence_pairs = st.persistence()
#
#    diagrams = []
#
#    for dimension in range(k + 1):
#        diagram = [
#            (c, d) for dim, (c, d) in persistence_pairs if dim == dimension
#        ]
#
#        diagrams.append(diagram)
#
#    return diagrams
#
