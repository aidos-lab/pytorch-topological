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

    def __init__(self, summary_statistic='total_persistence', **kwargs):
        """Create new loss function based on summary statistic.

        Parameters
        ----------
        summary_statistic : str
            Indicates which summary statistic function to use.

        **kwargs
            Optional keyword arguments, to be passed to the
            summary statistic function.
        """
        super().__init__()

        self.p = kwargs.get('p', 1.0)
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
        stat_src = torch.sum(
            torch.stack([
                self.stat_fn(pi.diagram, **self.kwargs) for pi in X
            ])
        )

        if Y is not None:
            stat_target = torch.sum(
                torch.stack([
                    self.stat_fn(pi.diagram, **self.kwargs) for pi in Y
                ])
            )

            return (stat_target - stat_src).abs().pow(self.p)
        else:
            return stat_src.abs().pow(self.p)


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

    # TODO: Improve documentation and nomenclature.
    def forward(self, X, Y):
        """Calculate signature loss between two data sets."""
        X_pc, X_pi = X
        Y_pc, Y_pi = Y

        X_dist = torch.cdist(X_pc, X_pc, self.p)
        Y_dist = torch.cdist(Y_pc, Y_pc, self.p)

        if self.normalise:
            X_dist = X_dist / X_dist.max()
            Y_dist = Y_dist / Y_dist.max()

        X_sig_X = [
            self._select_distances(X_dist, X_pi[dim].pairing)
            for dim in self.dimensions
        ]
        X_sig_Y = [
            self._select_distances(X_dist, Y_pi[dim].pairing)
            for dim in self.dimensions
        ]
        Y_sig_X = [
            self._select_distances(Y_dist, X_pi[dim].pairing)
            for dim in self.dimensions
        ]
        Y_sig_Y = [
            self._select_distances(Y_dist, Y_pi[dim].pairing)
            for dim in self.dimensions
        ]

        XY_dist = [
            0.5 * torch.linalg.vector_norm(XX - YX, ord=self.p)
            for XX, YX in zip(X_sig_X, Y_sig_X)
        ]

        YX_dist = [
            0.5 * torch.linalg.vector_norm(YY - XY, ord=self.p)
            for YY, XY in zip(Y_sig_Y, X_sig_Y)
        ]

        return torch.stack(XY_dist).sum() + torch.stack(YX_dist).sum()

    def _select_distances(self, dist, gens):
        # Dimension 0: only a mapping of vertices--edges is present, and
        # we must *only* access the edges.
        if gens.shape[1] == 3:
            selected_distances = dist[gens[:, 1], gens[:, 2]]

        # Dimension > 0: we can access all distances
        else:
            creator_distances = dist[gens[:, 0], gens[:, 1]]
            destroyer_distances = dist[gens[:, 2], gens[:, 3]]

            # Need to use `torch.abs` here because of the way the
            # signature lookup works. We are *not* guaranteed  to
            # get 'valid' persistence values when using a pairing
            # from space X to access distances from space Y,  for
            # instance, hence some of values could be *negative*.
            selected_distances = torch.abs(
                destroyer_distances - creator_distances
            )

        return selected_distances
