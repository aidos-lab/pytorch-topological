"""Cubical complex calculation module(s)."""

from torch import nn

import gudhi
import torch

import numpy as np


class Cubical(nn.Module):
    """Calculate cubical complex persisistence diagrams.

    This module calculates 'differentiable' persistence diagrams for
    point clouds. The underlying topological approximations are done
    by calculating a Vietoris--Rips complex of the data.
    """

    # TODO: Handle different dimensions?
    def __init__(self):
        """Initialise new module."""
        super().__init__()

    # TODO: Handle batches?
    def forward(self, x):
        """Implement forward pass for persistence diagram calculation.

        The forward pass entails calculating persistent homology on a
        cubical complex and returning a set of persistence diagrams.

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
        cubical_complex = gudhi.CubicalComplex(
            dimensions=x.shape,
            top_dimensional_cells=x.flatten()
        )

        # We need the persistence pairs first, even though we are *not*
        # using them directly here.
        dgm  =cubical_complex.persistence()
        cofaces = cubical_complex.cofaces_of_persistence_pairs()

        print(dgm)

        # TODO: Make this configurable
        dim = 0

        persistence_information = \
            self._extract_generators_and_diagrams(
                x,
                cofaces,
                dim
            )

    def _extract_generators_and_diagrams(self, x, cofaces, dim):
        pairs = torch.empty((0, 2), dtype=torch.long)

        try:
            regular_pairs = torch.as_tensor(
                cofaces[0][dim], dtype=torch.long
            )
            pairs = torch.cat(
                (pairs, regular_pairs)
            )
        except IndexError:
            pass

        try:
            infinite_pairs = torch.as_tensor(
                cofaces[1][dim], dtype=torch.long
            )
        except IndexError:
            infinite_pairs = None

        if infinite_pairs is not None:
            # 'Pair off' all the indices
            max_index = torch.argmax(x)
            fake_destroyers = torch.empty_like(infinite_pairs).fill_(max_index)

            infinite_pairs = torch.stack(
                (infinite_pairs, fake_destroyers), 1
            )

            pairs = torch.cat(
                (pairs, infinite_pairs)
            )

        return self._create_tensors_from_pairs(x, pairs)

    # Internal utility function to handle the 'heavy lifting:'
    # creates tensors from sets of persistence pairs.
    def _create_tensors_from_pairs(self, x, pairs):

        xs = x.shape

        # Notice that `creators` and `destroyers` refer to pixel
        # coordinates in the image.
        creators = torch.as_tensor(
                np.column_stack(
                    np.unravel_index(pairs[:, 0], xs)
                ),
                dtype=torch.long
        )
        destroyers = torch.as_tensor(
                np.column_stack(
                    np.unravel_index(pairs[:, 1], xs)
                ),
                dtype=torch.long
        )
        gens = torch.as_tensor(torch.hstack((creators, destroyers)))

        # TODO: Most efficient way to generate diagram again?
        persistence_diagram = torch.stack((
            x.ravel()[pairs[:, 0]],
            x.ravel()[pairs[:, 1]]
        ), 1)

        return (gens, persistence_diagram)
