"""Layers for topological data analysis based on graphs."""

import itertools

from torch_geometric.data import Data

from torch_geometric.loader import DataLoader

from torch_geometric.utils import erdos_renyi_graph

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from torch_scatter import scatter

import torch
import torch.nn as nn
import torch.functional as F

import gudhi as gd


# TODO: should be put into utils? This is available in `itertools`
# directly but only for Python 3.10+.
def pairwise(iterable):
    """Return pairwise iterator."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class DeepSetLayer(nn.Module):
    """Simple equivariant deep set layer."""

    def __init__(self, dim_in, dim_out, aggregation_fn):
        """Create new deep set layer.

        Parameters
        ----------
        dim_in : int
            Input dimension

        dim_out : int
            Output dimension

        aggregation_fn : str
            Aggregation to use for the reduction step. Must be valid for
            the ``torch_scatter.scatter()`` function, i.e. one of "sum",
            "mul", "mean", "min" or "max".
        """
        super().__init__()

        self.Gamma = nn.Linear(dim_in, dim_out)
        self.Lambda = nn.Linear(dim_in, dim_out, bias=False)

        self.aggregation_fn = aggregation_fn

    def forward(self, x, batch):
        """Implement forward pass through layer."""

        xm = scatter(x, batch, dim=0, reduce=self.aggregation_fn)
        xm = self.Lambda(xm)

        x = self.Gamma(x)
        x = x - xm[batch, :]
        return x


class TOGL(nn.Module):
    """Implementation of TOGL, a topological graph layer.

    Some caveats: this implementation only focuses on a set function
    aggregation of topological features. At the moment, it is not as
    powerful and feature-complete as the original implementation.
    """

    def __init__(
        self,
        n_features,
        n_filtrations,
        hidden_dim,
        out_dim,
        aggregation_fn,
    ):
        super().__init__()

        self.filtrations = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_filtrations),
        )

        self.n_filtrations = n_filtrations

        self.set_fn = nn.ModuleList(
            [
                nn.Linear(n_filtrations * 2, out_dim),
                nn.ReLU(),
                DeepSetLayer(out_dim, out_dim, aggregation_fn),
                nn.ReLU(),
                DeepSetLayer(
                    out_dim,
                    n_features,
                    aggregation_fn,
                ),
            ]
        )

        self.batch_norm = nn.BatchNorm1d(n_features)

    def compute_persistent_homology(
        self,
        x,
        edge_index,
        vertex_slices,
        edge_slices,
        batch,
        n_nodes,
        return_filtration=False,
    ):
        """Return persistence pairs (i.e. generators)."""
        # Apply filtrations to node attributes. For the edge values, we
        # use a sublevel set filtration.
        #
        # TODO: Support different ways of filtering?
        filtered_v = self.filtrations(x)
        filtered_e, _ = torch.max(
            torch.stack(
                (filtered_v[edge_index[0]], filtered_v[edge_index[1]])
            ),
            axis=0,
        )

        filtered_v = filtered_v.transpose(1, 0).cpu().contiguous()
        filtered_e = filtered_e.transpose(1, 0).cpu().contiguous()
        edge_index = edge_index.cpu().transpose(1, 0).contiguous()

        # TODO: Do we have to enforce contiguous indices here? 
        vertex_index = torch.arange(end=n_nodes, dtype=int)

        for (vi, vj), (ei, ej) in zip(
            pairwise(vertex_slices), pairwise(edge_slices)
        ):
            vertices = vertex_index[vi:vj]
            edges = edge_index[ei:ej]

            self._compute_persistent_homology(vertices, edges)

        # TODO: Calculate persistence tuples here. Not quite clear what
        # the output of this function should be.
        return None

    # Helper function for doing the actual calculation of topological
    # features of a graph.
    #
    # TODO: Missing filtration values for vertices and edges here.
    #
    # TODO: Need to think about how to handle different filtrations
    # etc.; what should be the best return value?
    def _compute_persistent_homology(self, vertices, edges):
        st = gd.SimplexTree()

        for v in vertices:
            st.insert([v], filtration=0.0)

        for u, v in  edges:
            st.insert([u, v], filtration=0.0)

        st.make_filtration_non_decreasing()
        st.expansion(2)
        persistence_pairs = st.persistence()

        generators = st.lower_star_persistence_generators()
        return generators

    def forward(self, x, data):
        """Implement forward pass through data."""
        # TODO: Is this the best signature? `data` is following directly
        # the convention of `PyG`.
        #
        # x : current node attributes of layer; we should not use the
        # original attributes here because they are not informed by a
        # previous layer.
        #
        # data : edge slice information etc.

        edge_index = data.edge_index

        vertex_slices = torch.Tensor(data._slice_dict["x"]).long()
        edge_slices = torch.Tensor(data._slice_dict["edge_index"]).long()
        batch = data.batch

        persistence_pairs = self.compute_persistent_homology(
            x,
            edge_index,
            vertex_slices,
            edge_slices,
            batch,
            n_nodes=data.num_nodes,
        )

        x0 = persistence_pairs.permute(1, 0, 2).reshape(
            persistence_pairs.shape[1], -1
        )

        for layer in self.set_fn:
            if isinstance(layer, DeepSetLayer):
                x0 = layer(x0, batch)
            else:
                x0 = layer(x0)

        # TODO: Residual step; could be made optional. Plus, the optimal
        # order of operations is not clear.
        x = x + self.batch_norm(F.relu(x0))
        return x


class TopoGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([GCNConv(1, 16), GCNConv(16, 2)])

        self.pooling_fn = global_mean_pool
        self.togl = TOGL(16, 16, 32, 16, "mean")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for layer in self.layers[:1]:
            x = layer(x, edge_index)

        x = self.togl(x, data)

        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        x = self.pooling_fn(x, data.batch)
        return x


B = 64
N = 100
p = 0.2

data_list = [
    Data(x=torch.rand(N, 1), edge_index=erdos_renyi_graph(N, p), num_nodes=N)
    for i in range(B)
]

loader = DataLoader(data_list, batch_size=8)

model = TopoGCN()

for index, batch in enumerate(loader):
    print(batch)

    vertex_slices = torch.Tensor(batch._slice_dict["x"]).long()
    edge_slices = torch.Tensor(batch._slice_dict["edge_index"]).long()

    model(batch)


# def calculate_persistent_homology(G, k=3):
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
