"""Vietoris--Rips complex calculation module(s)."""

from gph import ripser_parallel
from torch import nn

from torch_topological.nn import PersistenceInformation

import torch


class VietorisRipsComplex(nn.Module):
    """Calculate Vietoris--Rips complex of a data set.

    This module calculates 'differentiable' persistence diagrams for
    point clouds. The underlying topological approximations are done
    by calculating a Vietoris--Rips complex of the data.
    """

    def __init__(self, dim=1, p=2, **kwargs):
        """Initialise new module.

        Parameters
        ----------
        dim : int
            Calculates persistent homology up to (and including) the
            prescribed dimension.

        p : float
            Exponent for the `p`-norm calculation of distances.

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
        yet support other metrics.
        """
        super().__init__()

        self.dim = dim
        self.p = p

        # Ensures that the same parameters are used whenever calling
        # `ripser`.
        self.ripser_params = {
            'return_generators': True,
            'maxdim': self.dim,
        }

        self.ripser_params.update(kwargs)

    def forward(self, x):
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
        # Check whether individual batches need to be handled (3D array)
        # or not (2D array). We default to this type of processing for a
        # list as well.
        if isinstance(x, list) or len(x.shape) == 3:

            # TODO: This is rather ugly and inefficient but it is the
            # easiest workaround for now.
            return [
                self._forward(torch.as_tensor(x_)) for x_ in x
            ]
        else:
            return self._forward(torch.as_tensor(x))

    def _forward(self, x):
        """Handle a *single* point cloud.

        This internal function handles the calculation of topological
        features for a single point cloud, i.e. an `array_like` of 2D
        shape.

        Parameters
        ----------
        x : array_like of shape `(n, d)`
            Single input point cloud.

        Returns
        -------
        list of class:`PersistenceInformation`
            List of persistence information data structures, containing
            the persistence diagram and the persistence pairing of some
            dimension in the input data set.
        """
        generators = ripser_parallel(
            x.cpu().detach(),
            **self.ripser_params
        )['gens']

        # TODO: Is this always required? Can we calculate this in
        # a smarter fashion?
        distances = torch.cdist(x, x, p=self.p)

        # We always have 0D information.
        persistence_information = \
            self._extract_generators_and_diagrams(
                distances,
                generators,
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
        gens = gens[index]

        # TODO: Handling of infinite features not provided yet, but the
        # index shift is already correct.
        if not finite:
            index += 1

        if dim0:
            # In a Vietoris--Rips complex, all vertices are created at
            # time zero.
            creators = torch.zeros_like(torch.as_tensor(gens)[:, 0])
            destroyers = dist[gens[:, 1], gens[:, 2]]

            persistence_diagram = torch.stack(
                (creators, destroyers), 1
            )

            return [PersistenceInformation(gens, persistence_diagram, 0)]
        else:
            result = []

            for index, gens_ in enumerate(gens):
                creators = dist[gens_[:, 0], gens_[:, 1]]
                destroyers = dist[gens_[:, 2], gens_[:, 3]]

                persistence_diagram = torch.stack(
                    (creators, destroyers), 1
                )

                # Dimension zero is handled differently, so we need to
                # use an offset here.
                dimension = index + 1

                result.append(
                    PersistenceInformation(
                        gens_,
                        persistence_diagram,
                        dimension)
                )

        return result
