"""Alpha complex calculation module(s)."""

from torch import nn

from torch_topological.nn import PersistenceInformation

import gudhi
import itertools
import torch


class AlphaComplex(nn.Module):
    """Calculate persistence diagrams of an alpha complex.

    This module calculates persistence diagrams of an alpha complex,
    i.e. a subcomplex of the Delaunay triangulation, which is sparse
    and thus often substantially smaller than other complex.

    It was first described in [Edelsbrunner94]_ and is particularly
    useful when analysing low-dimensional data.

    Notes
    -----
    At the moment, this alpha complex implementation, following other
    implementations, provides *distance-based filtrations* only. This
    means that the resulting persistence diagrams do *not* correspond
    to the circumradius of a simplex.

    In addition, this implementation is **work in progress**. Some of
    the core features, such as handling of infinite features, are not
    available at the moment.

    References
    ----------
    .. [Edelsbrunner94] H. Edelsbrunner and E.P. Mücke,
       "Three-dimensional alpha shapes", *ACM Transactions on Graphics*,
       Volume 13, Number 1, pp. 43--72, 1994.
    """

    def __init__(self, p=2):
        """Initialise new alpha complex calculation module.

        Parameters
        ----------
        p : float
            Exponent for the `p`-norm calculation of distances.

        Notes
        -----
        This module currently only supports Minkowski norms. It does not
        yet support other metrics.
        """
        super().__init__()

        self.p = p

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
            based on proper creator--destroyer pairs of simplices. In
            dimension `k`, for instance, every generator is stored as
            a `k`-simplex followed by a `k+1` simplex.
        """
        # TODO: Copied from `VietorisRipsComplex`; needs to be
        # refactored into a unified interface.
        #
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
        alpha_complex = gudhi.alpha_complex.AlphaComplex(
            x.cpu().detach(),
            precision='fast',
        )

        st = alpha_complex.create_simplex_tree()
        st.persistence()
        persistence_pairs = st.persistence_pairs()

        max_dim = x.shape[-1]
        dist = torch.cdist(x.contiguous(), x.contiguous(), p=self.p)

        return [
            self._extract_generators_and_diagrams(
                dist,
                persistence_pairs,
                dim
            )
            for dim in range(0, max_dim)
        ]

    def _extract_generators_and_diagrams(self, dist, persistence_pairs, dim):
        pairs = [
            torch.cat((torch.as_tensor(p[0]), torch.as_tensor(p[1])), 0)
            for p in persistence_pairs if len(p[0]) == (dim + 1)
        ]

        # TODO: Ignore infinite features for now. Will require different
        # handling in the future.
        pairs = [p for p in pairs if len(p) == 2 * dim + 3]

        if not pairs:
            return PersistenceInformation(
                    pairing=[],
                    diagram=[],
                    dim=dim
            )

        # Create tensor of shape `(n, 2 * dim + 3)`, with `n` being the
        # number of finite persistence pairs.
        pairs = torch.stack(pairs)

        # We have to branch here because the creation of
        # zero-dimensional persistence diagrams is easy,
        # whereas higher-dimensional diagrams require an
        # involved lookup strategy.
        if dim == 0:
            creators = torch.zeros_like(torch.as_tensor(pairs)[:, 0])

        # Iterate over the flag complex in order to get (a) the distance
        # of the creator simplex, and (b) the distance of the destroyer.
        # We *cannot* look this information up in the filtration itself,
        # because we have to use gradient-imbued information such as the
        # set of pairwise distances.
        else:
            creators = torch.stack(
                    [
                        self._get_filtration_weight(creator, dist)
                        for creator in pairs[:, :dim+1]
                    ]
            )

        # For the destroyers, we can always rely on the same
        # construction, regardless of dimensionality.
        destroyers = torch.stack(
                [
                    self._get_filtration_weight(destroyer, dist)
                    for destroyer in pairs[:, dim+1:]
                ]
        )

        # Create the persistence diagram from creator and destroyer
        # information. This step is the same for all dimensions.
        persistence_diagram = torch.stack(
            (creators, destroyers), 1
        )

        return PersistenceInformation(
                pairing=pairs,
                diagram=persistence_diagram,
                dimension=dim
        )

    def _get_filtration_weight(self, simplex, dist):
        """Auxiliary function for querying simplex weights.

        This function returns the filtration weight of an arbitrary
        simplex under a distance-based filtration, i.e. the maximum
        weight of its cofaces. This function is crucial for getting
        persistence diagrams that are differentiable.

        Parameters
        ----------
        simplex : torch.tensor
            Simplex to query; must be a sequence of numbers, i.e. of
            shape `(n, 1)` or of shape `(n, )`.

        dist : torch.tensor
            Matrix of pairwise distances between points.

        Returns
        -------
        torch.tensor
            Scalar tensor containing the filtration weight.
        """
        weights = torch.stack([
            dist[edge] for edge in itertools.combinations(simplex, 2)
        ])

        # TODO: Might have to be adjusted depending on filtration
        # ordering?
        return torch.max(weights)
