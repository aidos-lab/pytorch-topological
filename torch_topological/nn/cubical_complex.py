"""Cubical complex calculation module."""

from torch import nn

from torch_topological.nn import PersistenceInformation

import gudhi
import torch

import numpy as np


class CubicalComplex(nn.Module):
    """Calculate cubical complex persistence diagrams.

    This module calculates 'differentiable' persistence diagrams for
    structured data, such as images. This is achieved by calculating
    a *cubical complex*.

    Cubical complexes are the natural choice for calculating topological
    features of highly-structured inputs. See [Rieck20a]_ for an example
    of how to apply such topological features in practice.

    References
    ----------
    .. [Rieck20a] B. Rieck et al., "Uncovering the Topology of
       Time-Varying fMRI Data Using Cubical Complex", *Advances in
       Neural Information Processing Systems 33*, pp. 6900--6912, 2020.
    """

    # TODO: Handle different dimensions?
    def __init__(self, superlevel=False, dim=None):
        """Initialise new module.

        Parameters
        ----------
        superlevel : bool
            Indicates whether to calculate topological features based on
            superlevel sets. By default, *sublevel set filtrations* are
            used.

        dim : int or `None`
            If set, describes dimension of input data. This is meant to
            be the dimension of an individual image **without** channel
            information, if any. The value of `dim` will change the way
            an input tensor is being handled: additional dimensions, if
            present, will be treated as batches or channels. If not set
            to an integer value, :func:`forward` will just *guess* what
            to do with an input (which should work in most cases).
        """
        super().__init__()

        # TODO: This is handled somewhat inelegantly below. Might be
        # smarter to update.
        self.superlevel = superlevel
        self.dim = dim

    def forward(self, x):
        """Implement forward pass for persistence diagram calculation.

        The forward pass entails calculating persistent homology on a
        cubical complex and returning a set of persistence diagrams.
        The way the input will be interpreted depends on the presence
        of the `dim` attribute of this class. If `dim` is set, the
        *last* `dim` dimensions of an input tensor will be considered to
        contain the image data. If `dim` is not set, image dimensions
        will be guessed as follows:

        1. Tensor of `dim = 2`: a single 2D image
        2. Tensor of `dim = 3`: a single 2D image with channels
        3. Tensor of `dim = 4`: a batch of 2D images with channels

        See parameters for more details.

        Parameters
        ----------
        x : array_like
            Input image(s). If `dim` has not been set, will *guess* how
            to handle the input as follows: `x` can either be a 2D array
            of shape `(H, W)`, which is treated as a single image, or
            a 3D array/tensor of the form `(C, H, W)`, with `C`
            representing the number of channels, or a 4D array/tensor of
            the form `(B, C, H, W)`, with `B` being the batch size. If
            `dim` has been set, the same handling strategy applies, but
            the *last* `dim` dimensions of the tensor are being used for
            the cubical complex calculation. All subsequent dimensions
            will be assumed to represent batches or channels (in this
            order). Hence, if `dim` is set, the tensor must at most have
            `dim + 2` dimensions.

        Returns
        -------
        list of :class:`PersistenceInformation`
            List of :class:`PersistenceInformation`, containing both the
            persistence diagrams and the generators, i.e. the
            *pairings*, of a certain dimension of topological features.
            If `x` is a 3D array, returns a list of lists, in which the
            first dimension denotes the batch and the second dimension
            refers to the individual instances of
            :class:`PersistenceInformation` elements. Similar for
            higher-order tensors.

        """
        # Dimension was provided; this makes calculating the *effective*
        # dimension of the tensor much easier: take everything but the
        # last `self.dim` dimensions.
        if self.dim is not None:
            shape = x.shape[:-self.dim]
            dims = len(shape)

        # No dimension was provided; just use the shape provided by the
        # client.
        else:
            dims = len(x.shape) - 2

        # No additional dimensions present: a single image
        if dims == 0:
            return self._forward(x)

        # Handle image with channels, such as a tensor of the form `(C, H, W)`
        elif dims == 1:
            return [
                self._forward(x_) for x_ in x
            ]

        # Handle image with channels and batch index, such as a tensor of
        # the form `(B, C, H, W)`.
        elif dims == 2:
            return [
                    [self._forward(x__) for x__ in x_] for x_ in x
            ]

    def _forward(self, x):
        """Handle a single-channel image.

        This internal function handles the calculation of topological
        features for a single-channel image, i.e. an `array_like`.

        Parameters
        ----------
        x : array_like of shape `(d_1, d_2, ..., d_d)`
            Single-channel input image of arbitrary dimensions. Batch
            dimensions and channel dimensions have to to be handled by
            the calling function explicitly. This function interprets
            its input as a high-dimensional image.

        Returns
        -------
        list of class:`PersistenceInformation`
            List of persistence information data structures, containing
            the persistence diagram and the persistence pairing of some
            dimension in the input data set.
        """
        if self.superlevel:
            x = -x

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
