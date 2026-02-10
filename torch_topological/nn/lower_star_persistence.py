"""Lower star persistence calculation module."""

import torch
from torch import nn
from torch_topological.nn import PersistenceInformation
from torch_geometric.data import Batch
import gudhi


class LowerStarPersistence(nn.Module):
    """Calculate persistence diagrams of a simplicial complex given a
    lower star filtration.

    This module calculates persistence diagrams of a simplicial complex,
    e.g. graphs with no self loops and double edges, given a filtration
    function on the vertices of the complex.  The filtration extends to
    higher dimensional simplicies by taking the maximum among the
    vertices.
    """

    def __init__(
            self,
            clip_inf=False,
            include_max_dim=False,
            list_of_lists=True,
            **kwargs
    ):
        """Initialize new module.

        Parameters
        ----------
        clip_inf : bool
            Indicates whether to clip the essential features (infinite
            destruction value) with the maximum value of the filtration.
            Default is False.
        
        include_max_dim : bool
            Indicates whether to include the homology of the highest
            dimension.  Default is False.

        list_of_lists : bool
            Indicates whether to return a list of lists of Persistence
            Information when the input is a Batch object, or return a
            list of tuples of Persistence Information with a pairs_batch
            tensor.  Default is True.

        **kwargs
            Additional arguments passed to gudhi backend.  Please refer
            to `the manual 
            <https://gudhi.inria.fr/python/latest/simplex_tree_ref.html#gudhi.SimplexTree.compute_persistence>` # noqa
            for more details. 
        
        """
        super().__init__()
        self.clip_inf = clip_inf
        self.include_max_dim = include_max_dim
        self.list_of_lists = list_of_lists

        self.compute_persistence_kwargs = {
            'persistence_dim_max': self.include_max_dim
        }

        self.compute_persistence_kwargs.update(kwargs)

    def forward(self, x, filtration):
        """Implement forward pass.

        The forward pass calculates the persistence homology of a
        simplicial complex equipped with a lower star filtration.
        Currently there are two types of inputs allowed: a Batch object
        in Pytorch Geometric or an iterable.  For the latter, the first
        entry is an integer and the other entries are tensors with
        integer entries.

        Parameters
        ----------
        x : Iterable or Batch
            Input simplicial complex.  If it is an iterable, x[0] has to
            be an integer which indicates the number of vertices.  x[i]
            for i > 0 has to be a tensor of size (i + 1, n_i) where i is
            the dimension of the simplices and n_i is the number of
            i-simplices.  x can also be a Batch object from Pytorch
            Geometric.  When the data is a Pytorch Geometric graph, the
            input should be (data.num_nodes, data.edge_index).

        filtration : Tensor
            Filtration values of the vertices.  Should be of size (n, )
            where n is the number of vertices.

        Returns
        -------
        list of PersistenceInformation or list of tuples or
        list of lists
            Returns a list of PersistenceInformation, with both the
            generators and the persistence diagrams.  If x is a Batch
            object and list_of_lists is True, returns a list of lists
            where the first dimension is the batch dimension.  Else
            returns a list of tuples where the second entry of the tuple
            is the pairs batch tensor.  Each generator will be given by
            a creator-destroyer pair of simplices, which in dimension k
            will be given by a k-simplex then a (k+1)-simplex.  If a
            generator is essential, i.e. not destroyed, then the
            destroyer simplex will be given by a tensor with all -1
            entries.
        """
        if isinstance(x, Batch):

            if self.list_of_lists:

                return [
                    self._forward(
                        (x_.num_nodes, x_.edge_index), filtration[x.batch == i]
                    ) for i, x_ in enumerate(x.to_data_list())
                ]

            else:

                return self._forward((x.num_nodes, x.edge_index), filtration,
                                     batch=x.batch)

        else:

            return self._forward(x, filtration)

    def _forward(self, x, filtration, batch=None):

        st = gudhi.SimplexTree()
        device = filtration.device
        filtration = filtration.cpu()

        for v in torch.arange(x[0], device=torch.device('cpu'))[:, None]:
            st.insert(v, filtration=filtration[v])

        for i in range(1, len(x)):
            for f in x[i].t().cpu():
                st.insert(f, filtration=torch.max(filtration[f]))

        st.compute_persistence(**self.compute_persistence_kwargs)
        persistence_pairs = st.persistence_pairs()

        if self.include_max_dim:
            max_dim = x[-1].size(dim=0) - 1
        else:
            max_dim = x[-1].size(dim=0) - 2

        persistence_information = [
            self._extract_generators_and_diagrams(
                filtration,
                persistence_pairs,
                dim,
                device,
                batch
            ) for dim in range(max_dim + 1)
        ]

        return persistence_information

    def _extract_generators_and_diagrams(self, filtration, persistence_pairs,
                                         dim, device, batch):

        pairs = []
        for p in persistence_pairs:
            if len(p[0]) == dim + 1:
                if len(p[1]) != 0:
                    pairs.append(torch.cat((torch.as_tensor(p[0]),
                                            torch.as_tensor(p[1]))))
                else:
                    pairs.append(torch.cat((torch.as_tensor(p[0]),
                                            -1 * torch.ones(
                                                dim + 2, dtype=torch.long
                                            ))))

        if len(pairs) == 0:
            empty = PersistenceInformation(
                pairing=torch.empty((0, 2 * dim + 3), dtype=torch.long,
                                    device=device),
                diagram=torch.empty((0, 2), device=device),
                dimension=dim
            )

            if batch is not None:
                return empty, torch.empty((0, ), dtype=torch.long,
                                          device=device)
            else:
                return empty

        else:
            pairs = torch.stack(pairs).to(device)

            if batch is not None:
                pairs_batch = torch.zeros(pairs.size(0))
                for i in range(pairs.size(0)):
                    pairs_batch[i] = batch[pairs[i, 0]]

            filtration = filtration.to(device)

            birth = torch.amax(filtration[pairs[:, :dim + 1]], 1)[:, None]
            death = filtration[pairs[:, dim + 1:]]

            if self.clip_inf:
                death[pairs[:, dim + 1:] == -1] = torch.max(filtration)
            else:
                death[pairs[:, dim + 1:] == -1] = torch.inf

            death = torch.amax(death, 1)[:, None]

            diagram = torch.cat((birth, death), dim=1)

            persistence = PersistenceInformation(
                pairing=pairs,
                diagram=diagram,
                dimension=dim
            )
            if batch is not None:
                return persistence, pairs_batch
            else:
                return persistence
