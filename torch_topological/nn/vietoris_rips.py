"""Minimise functions based on Vietoris--Rips complexes."""

from gph import ripser_parallel
from torch import nn

import torch


class VietorisRips(nn.Module):
    """Calculate Vietoris--Rips persistence diagrams.

    This module calculates 'differentiable' persistence diagrams for
    point clouds. The underlying topological approximations are done
    by calculating a Vietoris--Rips complex of the data.
    """

    def __init__(self, dim=1):
        """Initialise new module.

        Parameters
        ----------
        dim : int
            Calculates persistent homology up to (and including) the
            prescribed dimension.
        """
        super().__init__()

        # Ensures that the same parameters are used whenever calling
        # `ripser`.
        self.ripser_params = {
            'return_generators': True,
            'maxdim': dim,
        }

    def forward(self, x):
        """Implement forward pass for persistence diagram calculation.

        The forward pass entails calculating persistent homology on
        a point cloud and returning a set of persistence diagrams.

        Parameters
        ----------
        x : `np.array` or `torch.tensor`
            Input point cloud

        Returns
        -------
        List of tuples of the form `(gen, pd)`, where `gen` refers to
        the set of generators for the respective dimension, while `pd`
        denotes the persistence diagram.
        """
        generators = ripser_parallel(
            x.detach(),
            **self.ripser_params
        )['gens']

        # TODO: Is this always required? Can we calculate this in
        # a smarter fashion?
        #
        # Calculate distances in the source space and select the
        # appropriate tuples later on.
        source_distances = torch.cdist(x, x, p=2)

        generators_0d = generators[0]
        generators_1d = generators[1]

        edge_indices_0d = generators_0d[:, 1:]
        edge_indices_1d = generators_1d[0][:, 0:]

        destroyers_0d = source_distances[
            edge_indices_0d[:, 0], edge_indices_0d[:, 1]
        ]

        creators_1d = source_distances[
            edge_indices_1d[:, 0], edge_indices_1d[:, 1]
        ]

        destroyers_1d = source_distances[
            edge_indices_1d[:, 2], edge_indices_1d[:, 3]
        ]

        persistence_0d = destroyers_0d
        persistence_1d = destroyers_1d - creators_1d

        persistence_diagram_0d = torch.stack(
            (torch.zeros_like(persistence_0d), destroyers_0d), 1
        )

        persistence_diagram_1d = torch.stack(
            (creators_1d, destroyers_1d), 1
        )

        return (
            (generators_0d, persistence_diagram_0d),
            (generators_1d, persistence_diagram_1d)
        )
