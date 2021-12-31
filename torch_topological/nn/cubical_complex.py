"""Cubical complex calculation module."""

from torch import nn

from torch_topological.nn import PersistenceInformation

import gudhi
import torch

import numpy as np


class CubicalComplex(nn.Module):
    """Calculate cubical complex persistence diagrams.

    This module calculates 'differentiable' persistence diagrams for
    point clouds. The underlying topological approximations are done
    by calculating a cubical complex of the data.

    Cubical complexes are an excellent choice whenever data exhibits
    a highly-structured form, such as *images*.
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
        x : array_like
            Input image(s). `x` can either be a 2D array of shape `(H, W)`,
            which is treated as a single image, or a 3D array/tensor of the
            form `(C, H, W)`, with `C` representing the number of channels,
            or a 4D array/tensor of the form `(B, C, H, W)`, with `B` being
            the batch size.

        Returns
        -------
        list of :class:`PersistenceInformation`
            List of :class:`PersistenceInformation`, containing both the
            persistence diagrams and the generators, i.e. the
            *pairings*, of a certain dimension of topological features.
            If `x` is a 3D array, returns a list of lists, in which the
            first dimension denotes the batch and the second dimension
            refers to the individual instances of
            :class:`PersistenceInformation` elements.

        """
        # Check which shape to handle.
        if len(x.shape) == 2:
            return self._forward(x)

        # Handle channels
        elif len(x.shape) == 3:
            return [
                self._forward(x_) for x_ in x
            ]

        # Handle full batch
        elif len(x.shape) == 4:
            return [
                    [self._forward(x__) for x__ in x_] for x_ in x
            ]

    def _forward(self, x):
        """Handle a single-channel image.

        This internal function handles the calculation of topological
        features for a single-channel image, i.e. an `array_like`  of
        2D shape.

        Parameters
        ----------
        x : array_like of shape `(H, W)`
            Single-channel input image.

        Returns
        -------
        list of class:`PersistenceInformation`
            List of persistence information data structures, containing
            the persistence diagram and the persistence pairing of some
            dimension in the input data set.
        """
        cubical_complex = gudhi.CubicalComplex(
            dimensions=x.shape,
            top_dimensional_cells=x.flatten()
        )

        # We need the persistence pairs first, even though we are *not*
        # using them directly here.
        cubical_complex.persistence()
        cofaces = cubical_complex.cofaces_of_persistence_pairs()

        max_dim = len(x.shape)

        # TODO: Make this configurable; is it possible that users only
        # want to return a *part* of the data?
        persistence_information = [
            self._extract_generators_and_diagrams(
                x,
                cofaces,
                dim
            ) for dim in range(0, max_dim)
        ]

        return persistence_information

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

        return self._create_tensors_from_pairs(x, pairs, dim)

    # Internal utility function to handle the 'heavy lifting:'
    # creates tensors from sets of persistence pairs.
    def _create_tensors_from_pairs(self, x, pairs, dim):

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

        return PersistenceInformation(
                pairing=gens,
                diagram=persistence_diagram,
                dimension=dim
        )
