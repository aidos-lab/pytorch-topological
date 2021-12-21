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
        cubical_complex.persistence()
        cofaces = cubical_complex.cofaces_of_persistence_pairs()

        # TODO: Make this configurable
        dim = 0

        persistence_information = \
            self._extract_generators_and_diagrams(
                x,
                cofaces,
                dim
            )

    def _extract_generators_and_diagrams(self, x, cofaces, dim):
        xs = x.shape

        # Handle regular pairs first
        try:
            regular_pairs = cofaces[0][dim]
        except IndexError:
            regular_pairs = None

        if regular_pairs is not None:
            # Notice that `creators` and `destroyers` refer to pixel
            # coordinates in the image.
            creators = torch.as_tensor(
                    np.column_stack(
                        np.unravel_index(regular_pairs[:, 0], xs)
                    ),
                    dtype=torch.int
            )
            destroyers = torch.as_tensor(
                    np.column_stack(
                        np.unravel_index(regular_pairs[:, 1], xs)
                    ),
                    dtype=torch.int
            )
            gens = torch.as_tensor(torch.hstack((creators, destroyers)))

            print('creators =', creators)
            print('destroyers =', destroyers)

            print('gens =', gens)

            print(x.shape)
            print(creators.shape)

            print(x)
            print(x[1, 3], x[1, 0], x[3, 0])

            # TODO: Most efficient way to generate diagram again?
            print(x.ravel()[regular_pairs[:, 0]])

            # Create a persistence diagram. We need access to the
            # original input tensor here.
            #persistence_diagram = torch.hstack(
            #    (x[creators], x[destroyers])
            #)

            print(persistence_diagram)
