"""Minimise functions based on Vietoris--Rips complexes."""

from gph import ripser_parallel
from torch import nn

import torch


class VietorisRips(nn.Module):
    """Calculate Vietoris--Rips persistence diagrams.

    This module calculates 'differentiable' persistence diagrams between
    up to two spaces. The first space is treated as the source or input,
    while the second space is treated as the non-trainable target, whose
    persistence diagram information will be cached.
    """

    def __init__(self, X, Y=None):
        """Initialise new module.

        Parameters
        ----------
        X : `np.array` or `torch.tensor`
            Input point cloud, typically denoted as the 'source' point
            cloud that can be adjusted.

        Y : `np.array`, `torch.tensor`, or `None`
            Target point cloud. Will be considered fixed and not
            trainable at all.
        """
        super().__init__()

        self.X = X
        self.Y = Y

        info = ripser_parallel(self.Y, return_generators=True)

        self.pd_target = [
            torch.as_tensor(pd) for pd in info['dgms']
        ]

        self.pp_target = [
            torch.tensor(gens) for gens in info['gens']
        ]

    def forward(self):
        """Implement forward pass for persistence diagram calculation.

        The forward pass entails calculating persistent homology on the
        current point cloud and returning a set of persistence diagrams.

        Returns
        -------
        Tuple containing persistence information for the source and
        target space, respectively. Each tuple is a list of tuples of
        the form `(gen, pd)`, where `gen` refers to the set of
        generators for the respective dimension, and `pd` is the usual
        persistence diagram.
        """
        X = self.X

        generators = ripser_parallel(
            X.detach(),
            return_generators=True
        )['gens']

        # TODO: Is this always required? Can we calculate this in
        # a smarter fashion?
        #
        # Calculate distances in the source space and select the
        # appropriate tuples later on.
        source_distances = torch.cdist(X, X, p=2)

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
            [
                (generators_0d, persistence_diagram_0d),
                (generators_1d, persistence_diagram_1d)
            ],
            [
                (gens, pd) for gens, pd in zip(self.pp_target, self.pd_target)
            ]
        )
