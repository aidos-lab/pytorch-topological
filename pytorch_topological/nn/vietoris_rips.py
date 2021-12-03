"""Minimise functions based on Vietoris--Rips complexes."""

import numpy as np

from gph import ripser_parallel
from torch import nn

import torch


def _total_persistence(D, p=2):
    """Calculate total persistence of a persistence diagram.

    Parameters
    ----------
    D : `np.array`
        Persistence diagram, assumed to be in the usual `giotto-ph`
        format: each entry is supposed to be a tuple of the form $(x, y,
        d)$, with $(x, y)$ being the usual creation--destruction pair,
        and $d$ denoting the dimension.

    p : float
        Weight parameter for the total persistence calculation.

    Returns
    -------
    Total persistence of `D`.
    """
    persistence = np.diff(D[:, 0:2])
    persistence = persistence[np.isfinite(persistence)]

    # TODO: Normalise?
    return np.sum(np.power(np.abs(persistence), p))


class ModelSpaceLoss(nn.Module):
    """Optimise persistence-based functions on a Vietoris--Rips complex.

    This module should be used if a 'model space' $Y$ is present, and
    a loss function between the two spaces should be optimised.
    """

    def __init__(self, X, Y):
        """Initialise new module.

        Parameters
        ----------
        X : `np.array`
            Input point cloud, typically denoted as the 'source' point
            cloud that can be adjusted.

        Y : `np.array`
            Target point cloud. Will be considered fixed and not
            trainable at all.
        """
        super().__init__()

        self.X = nn.Parameter(torch.as_tensor(X), requires_grad=True)
        self.Y = Y

        # TODO: make configurable
        self.pd_target = ripser_parallel(self.Y)['dgms']

    def forward(self):
        """Implement forward pass of the loss.

        The forward pass entails evaluating the provided function on the
        persistence diagrams obtained from the current source point
        cloud.

        Returns
        -------
        Loss
        """
        X = self.X
        pd_target = self.pd_target

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

        source_total_persistence = persistence_0d.pow(2).sum() + persistence_1d.pow(2).sum()
        target_total_persistence = _total_persistence(pd_target[0]) + _total_persistence(pd_target[1])

        loss = torch.abs(source_total_persistence - target_total_persistence)
        return loss
