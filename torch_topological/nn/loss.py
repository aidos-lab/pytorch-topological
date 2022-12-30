"""Loss terms for various optimisation objectives."""

import torch

from torch_topological.utils import is_iterable


class SummaryStatisticLoss(torch.nn.Module):
    r"""Implement loss based on summary statistic.

    This is a generic loss function based on topological summary
    statistics. It implements a loss of the following form:

    .. math:: \|s(X) - s(Y)\|^p

    In the preceding equation, `s` refers to a function that results in
    a scalar-valued summary of a persistence diagram.
    """

    def __init__(self, summary_statistic="total_persistence", **kwargs):
        """Create new loss function based on summary statistic.

        Parameters
        ----------
        summary_statistic : str
            Indicates which summary statistic function to use. Must be
            a summary statistics function that exists in the utilities
            module, i.e. :mod:`torch_topological.utils`.

            At present, the following choices are valid:

            - `torch_topological.utils.persistent_entropy`
            - `torch_topological.utils.polynomial_function`
            - `torch_topological.utils.total_persistence`
            - `torch_topological.utils.p_norm`

        **kwargs
            Optional keyword arguments, to be passed to the
            summary statistic function.
        """
        super().__init__()

        self.p = kwargs.get("p", 1.0)
        self.kwargs = kwargs

        import torch_topological.utils.summary_statistics as stat

        self.stat_fn = getattr(stat, summary_statistic, None)

    def forward(self, X, Y=None):
        r"""Calculate loss based on input tensor(s).

        Parameters
        ----------
        X : list of :class:`PersistenceInformation`
            Source information. Supposed to contain persistence diagrams
            and persistence pairings.

        Y : list of :class:`PersistenceInformation` or `None`
            Optional target information. If set, evaluates a difference
            in loss functions as shown in the introduction. If `None`,
            a simpler variant of the loss will be evaluated.

        Returns
        -------
        torch.tensor
            Loss based on the summary statistic selected by the client.
            Given a statistic :math:`s`, the function returns the
            following expression:

            .. math:: \|s(X) - s(Y)\|^p

            In case no target tensor `Y` has been provided, the latter part
            of the expression amounts to `0`.
        """
        stat_src = self._evaluate_stat_fn(X)

        if Y is not None:
            stat_target = self._evaluate_stat_fn(Y)
            return (stat_target - stat_src).abs().pow(self.p)
        else:
            return stat_src.abs().pow(self.p)

    def _evaluate_stat_fn(self, X):
        """Evaluate statistic function for a given tensor."""
        return torch.sum(
            torch.stack(
                [
                    self.stat_fn(pers_info.diagram, **self.kwargs)
                    for pers_info in X
                ]
            )
        )


class SignatureLoss(torch.nn.Module):
    """Implement topological signature loss.

    This module implements the topological signature loss first
    described in [Moor20a]_. In contrast to the original code provided
    by the authors, this module also provides extensions to
    higher-dimensional generators if desired.

    The module can be used in conjunction with any set of generators and
    persistence diagrams, i.e. with any set of persistence pairings and
    persistence diagrams. At the moment, it is restricted to calculating
    a Minkowski distances for the loss calculation.

    References
    ----------
    .. [Moor20a] M. Moor et al., "Topological Autoencoders",
        *Proceedings of the 37th International Conference on Machine
        Learning*, PMLR 119, pp. 7045--7054, 2020.
    """

    def __init__(self, p=2, normalise=True, dimensions=0):
        """Create new loss instance.

        Parameters
        ----------
        p : float
            Exponent for the `p`-norm calculation of distances.

        normalise : bool
            If set, normalises distances for each point cloud. This can
            be useful when working with batches.

        dimensions : int or tuple of int
            Dimensions to use in the signature calculation. Following
            [Moor20a]_, this is set by default to `0`.
        """
        super().__init__()

        self.p = p
        self.normalise = normalise
        self.dimensions = dimensions

        # Ensure that we can iterate over the dimensions later on, as
        # this simplifies the code.
        if not is_iterable(self.dimensions):
            self.dimensions = [self.dimensions]

    def forward(self, X, Y):
        """Calculate the signature loss between two point clouds.

        This loss function uses the persistent homology from each point
        cloud in order to retrieve the topologically relevant distances
        from a distance matrix calculated from the point clouds. For
        more information, see [Moor20a]_.

        Parameters
        ----------
        X: Tuple[torch.tensor, PersistenceInformation]
            A tuple consisting of the point cloud and the persistence
            information of the point cloud. The persistent information
            is calculated by performing persistent homology calculation
            to retrieve a list of topologically relevant edges.

        Y: Tuple[torch.tensor, PersistenceInformation]
            A tuple consisting of the point cloud and the persistence
            information of the point cloud. The persistent information
            is calculated by performing persistent homology calculation
            to retrieve a list of topologically relevant edges.

        Returns
        -------
        torch.tensor
            A scalar representing the topological loss term for the two
            data sets.
        """
        X_point_cloud, X_persistence_info = X
        Y_point_cloud, Y_persistence_info = Y

        # Calculate the pairwise distance matrix between points in the
        # point cloud. Distances are calculated using the p-norm.
        X_pairwise_dist = torch.cdist(X_point_cloud, X_point_cloud, self.p)
        Y_pairwise_dist = torch.cdist(Y_point_cloud, Y_point_cloud, self.p)

        if self.normalise:
            X_pairwise_dist = X_pairwise_dist / X_pairwise_dist.max()
            Y_pairwise_dist = Y_pairwise_dist / Y_pairwise_dist.max()

        # Using the topologically relevant edges from point cloud X,
        # retrieve the corresponding distances from the pairwise
        # distance matrix of X.
        X_sig_X = [
            self._select_distances(
                X_pairwise_dist, X_persistence_info[dim].pairing
            )
            for dim in self.dimensions
        ]

        # Using the topologically relevant edges from point cloud Y,
        # retrieve the corresponding distances from the pairwise
        # distance matrix of X.
        X_sig_Y = [
            self._select_distances(
                X_pairwise_dist, Y_persistence_info[dim].pairing
            )
            for dim in self.dimensions
        ]

        # Using the topologically relevant edges from point cloud X,
        # retrieve the corresponding distances from the pairwise
        # distance matrix of Y.
        Y_sig_X = [
            self._select_distances(
                Y_pairwise_dist, X_persistence_info[dim].pairing
            )
            for dim in self.dimensions
        ]

        # Using the topologically relevant edges from point cloud Y,
        # retrieve the corresponding distances from the pairwise
        # distance matrix of Y.
        Y_sig_Y = [
            self._select_distances(
                Y_pairwise_dist, Y_persistence_info[dim].pairing
            )
            for dim in self.dimensions
        ]

        XY_dist = self._partial_distance(X_sig_X, Y_sig_X)
        YX_dist = self._partial_distance(Y_sig_Y, X_sig_Y)

        return torch.stack(XY_dist).sum() + torch.stack(YX_dist).sum()

    def _select_distances(self, pairwise_distance_matrix, generators):
        """Select topologically relevant edges from a pairwise distance matrix.

        Parameters
        ----------
        pairwise_distance_matrix: torch.tensor
            NxN pairwise distance matrix of a point cloud.

        generators: np.ndarray
            A 2D array consisting of indices corresponding to edges that
            correspond to the birth/destruction of some topological
            feature during persistent homology calculation. If the
            generator corresponds to topological features in
            0-dimension, i.e. connected components, we only consider the
            edges that destroy connected components (we do not consider
            vertices). If the generator corresponds to topological
            features in > 0 dimensions, e.g holes or voids, we consider
            edges that create/destroy such topological features.

        Returns
        -------
        torch.tensor
            A vector that contains all of the topologically relevant
            distances.
        """
        # Dimension 0: only a mapping of vertices--edges is present, and
        # we must *only* access the edges.
        if generators.shape[1] == 3:
            selected_distances = pairwise_distance_matrix[
                generators[:, 1], generators[:, 2]
            ]

        # Dimension > 0: we can access all distances
        else:
            creator_distances = pairwise_distance_matrix[
                generators[:, 0], generators[:, 1]
            ]
            destroyer_distances = pairwise_distance_matrix[
                generators[:, 2], generators[:, 3]
            ]

            # Need to use `torch.abs` here because of the way the
            # signature lookup works. We are *not* guaranteed  to
            # get 'valid' persistence values when using a pairing
            # from space X to access distances from space Y,  for
            # instance, hence some of values could be *negative*.
            selected_distances = torch.abs(
                destroyer_distances - creator_distances
            )

        return selected_distances

    def _partial_distance(self, A, B):
        """
        Calculate partial distances between pairings.

        The purpose of this function is to calculate a partial distance
        for the loss, depending on distances selected from the pairing.
        """
        dist = [
            0.5 * torch.linalg.vector_norm(a - b, ord=self.p)
            for a, b in zip(A, B)
        ]

        return dist
