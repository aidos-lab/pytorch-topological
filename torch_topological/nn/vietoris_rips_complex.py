"""Vietoris--Rips complex calculation module(s)."""

from itertools import starmap

from gph import ripser_parallel
from torch import nn

from torch_topological.nn import PersistenceInformation
from torch_topological.nn.data import batch_handler

import numpy
import torch


class VietorisRipsComplex(nn.Module):
    """Calculate Vietoris--Rips complex of a data set.

    This module calculates 'differentiable' persistence diagrams for
    point clouds. The underlying topological approximations are done
    by calculating a Vietoris--Rips complex of the data.
    """

    def __init__(
        self,
        dim=1,
        p=2,
        threshold=numpy.inf,
        keep_infinite_features=False,
        **kwargs
    ):
        """Initialise new module.

        Parameters
        ----------
        dim : int
            Calculates persistent homology up to (and including) the
            prescribed dimension.

        p : float
            Exponent indicating which Minkowski `p`-norm to use for the
            calculation of pairwise distances between points. Note that
            if `treat_as_distances` is supplied to :func:`forward`, the
            parameter is ignored and will have no effect. The rationale
            is to permit clients to use a pre-computed distance matrix,
            while always falling back to Minkowski norms.

        threshold : float
            If set to a finite number, only calculates topological
            features up to the specified distance threshold. Thus,
            any persistence pairings may contain infinite features
            as well.

        keep_infinite_features : bool
            If set, keeps infinite features. This flag is disabled by
            default. The rationale for this is that infinite features
            require more deliberate handling and, in case `threshold`
            is not changed, only a *single* infinite feature will not
            be considered in subsequent calculations.

        **kwargs
            Additional arguments to be provided to ``ripser``, i.e. the
            backend for calculating persistent homology. The `n_threads`
            parameter, which controls parallelisation, is probably the
            most relevant parameter to be adjusted.
            Please refer to the `the gitto-ph documentation
            <https://giotto-ai.github.io/giotto-ph/build/html/index.html>`_
            for more details on admissible parameters.

        Notes
        -----
        This module currently only supports Minkowski norms. It does not
        yet support other metrics internally. To use custom metrics, you
        need to set `treat_as_distances` in the :func:`forward` function
        instead.
        """
        super().__init__()

        self.dim = dim
        self.p = p
        self.threshold = threshold
        self.keep_infinite_features = keep_infinite_features

        # Ensures that the same parameters are used whenever calling
        # `ripser`.
        self.ripser_params = {
            'return_generators': True,
            'maxdim': self.dim,
            'thresh': self.threshold
        }

        self.ripser_params.update(kwargs)

    def forward(self, x, treat_as_distances=False):
        """Implement forward pass for persistence diagram calculation.

        The forward pass entails calculating persistent homology on
        a point cloud and returning a set of persistence diagrams.

        Parameters
        ----------
        x : array_like
            Input point cloud(s). `x` can either be a 2D array of shape
            `(n, d)`, which is treated as a single point cloud, or a 3D
            array/tensor of the form `(b, n, d)`, with `b` representing
            the batch size. Alternatively, you may also specify a list,
            possibly containing point clouds of non-uniform sizes.

        treat_as_distances : bool
            If set, treats `x` as containing pre-computed distances
            between points. The semantics of how `x` is handled are
            not changed; the only difference is that when `x` has a
            shape of `(n, d)`, the values of `n` and `d` need to be
            the same.

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

            Generators will be represented in the persistence pairing
            based on vertex--edge pairs (dimension 0) or edge--edge
            pairs. Thus, the persistence pairing in dimension zero will
            have three components, corresponding to a vertex and an
            edge, respectively, while the persistence pairing for higher
            dimensions will have four components.
        """
        return batch_handler(
            x,
            self._forward,
            treat_as_distances=treat_as_distances
        )

    def _forward(self, x, treat_as_distances=False):
        """Handle a *single* point cloud.

        This internal function handles the calculation of topological
        features for a single point cloud, i.e. an `array_like` of 2D
        shape.

        Parameters
        ----------
        x : array_like of shape `(n, d)`
            Single input point cloud.

        treat_as_distances : bool
            Flag indicating whether `x` should be treated as a distance
            matrix. See :func:`forward` for more information.

        Returns
        -------
        list of class:`PersistenceInformation`
            List of persistence information data structures, containing
            the persistence diagram and the persistence pairing of some
            dimension in the input data set.
        """
        if treat_as_distances:
            distances = x
        else:
            distances = torch.cdist(x, x, p=self.p)

        generators = ripser_parallel(
            distances.cpu().detach().numpy(),
            metric='precomputed',
            **self.ripser_params
        )['gens']

        # We always have 0D information.
        persistence_information = \
            self._extract_generators_and_diagrams(
                distances,
                generators,
                dim0=True,
            )

        if self.keep_infinite_features:
            persistence_information_inf = \
                self._extract_generators_and_diagrams(
                    distances,
                    generators,
                    finite=False,
                    dim0=True,
                )

        # Check whether we have any higher-dimensional information that
        # we should return.
        if self.dim >= 1:
            persistence_information.extend(
                self._extract_generators_and_diagrams(
                    distances,
                    generators,
                    dim0=False,
                )
            )

            if self.keep_infinite_features:
                persistence_information_inf.extend(
                    self._extract_generators_and_diagrams(
                        distances,
                        generators,
                        finite=False,
                        dim0=False,
                    )
                )

        # Concatenation is only necessary if we want to keep infinite
        # features.
        if self.keep_infinite_features:
            persistence_information = self._concatenate_features(
                persistence_information, persistence_information_inf
            )

        return persistence_information

    def _extract_generators_and_diagrams(
            self,
            dist,
            gens,
            finite=True,
            dim0=False
    ):
        """Extract generators and persistence diagrams from raw data.

        This convenience function translates between the output of
        `ripser_parallel` and the required output of this function.
        """
        index = 1 if not dim0 else 0

        # Perform index shift to find infinite features in the tensor.
        if not finite:
            index += 2

        gens = gens[index]

        if dim0:
            if finite:
                # In a Vietoris--Rips complex, all vertices are created at
                # time zero.
                creators = torch.zeros_like(
                    torch.as_tensor(gens)[:, 0],
                    device=dist.device
                )

                destroyers = dist[gens[:, 1], gens[:, 2]]
            else:
                creators = torch.zeros_like(
                    torch.as_tensor(gens)[:],
                    device=dist.device
                )

                destroyers = torch.full_like(
                    torch.as_tensor(gens)[:],
                    torch.inf,
                    dtype=torch.float,
                    device=dist.device
                )

                inf_pairs = numpy.full(
                    shape=(gens.shape[0], 2), fill_value=-1
                )
                gens = numpy.column_stack((gens, inf_pairs))

            persistence_diagram = torch.stack(
                (creators, destroyers), 1
            )

            return [PersistenceInformation(gens, persistence_diagram, 0)]
        else:
            result = []

            for index, gens_ in enumerate(gens):
                # Dimension zero is handled differently, so we need to
                # use an offset here. Note that this is not used as an
                # index into the `gens` array any more.
                dimension = index + 1

                if finite:
                    creators = dist[gens_[:, 0], gens_[:, 1]]
                    destroyers = dist[gens_[:, 2], gens_[:, 3]]

                    persistence_diagram = torch.stack(
                        (creators, destroyers), 1
                    )
                else:
                    creators = dist[gens_[:, 0], gens_[:, 1]]

                    destroyers = torch.full_like(
                        torch.as_tensor(gens_)[:, 0],
                        torch.inf,
                        dtype=torch.float,
                        device=dist.device
                    )

                    # Create special infinite pairs; we pretend that we
                    # are concatenating with unknown edges here.
                    inf_pairs = numpy.full(
                        shape=(gens_.shape[0], 2), fill_value=-1
                    )
                    gens_ = numpy.column_stack((gens_, inf_pairs))

                persistence_diagram = torch.stack(
                    (creators, destroyers), 1
                )

                result.append(
                    PersistenceInformation(
                        gens_,
                        persistence_diagram,
                        dimension)
                )

        return result

    def _concatenate_features(self, pers_info_finite, pers_info_infinite):
        """Concatenate finite and infinite features."""
        def _apply(fin, inf):
            assert fin.dimension == inf.dimension

            diagram = torch.concat((fin.diagram, inf.diagram))
            pairing = numpy.concatenate((fin.pairing, inf.pairing), axis=0)
            dimension = fin.dimension

            return PersistenceInformation(
                pairing=pairing,
                diagram=diagram,
                dimension=dimension
            )

        return list(starmap(_apply, zip(pers_info_finite, pers_info_infinite)))
