"""Loss terms for various optimisation objectives."""

import torch


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
        X : `torch.tensor` or `None`
            Source tensor. Supposed to contain persistence diagrams and
            persistence pairings.

        Y : `torch.tensor` or `None`
            Optional target tensor. If set, evaluates a difference in
            loss functions as shown in the introduction. If `None`, a
            simpler variant of the loss will be evaluated.

        Returns
        -------
        Loss based on the summary statistic selected by the client.
        Given a statistic :math:`s`, the function returns the following
        expression:

        .. math:: \|s(X) - s(Y)\|^p

        In case no target tensor `Y` has been provided, the latter part
        of the expression amounts to `0`.
        """
        stat_src = torch.sum(
            torch.stack([
                self.stat_fn(D, **self.kwargs) for _, D in X
            ])
        )

        if Y is not None:
            stat_target = torch.sum(
                torch.stack([
                    self.stat_fn(D, **self.kwargs) for _, D in Y
                ])
            )

            return (stat_target - stat_src).abs().pow(self.p)
        else:
            return stat_src.abs().pow(self.p)


class SignatureLoss(torch.nn.Module):
    def __init__(self, Y=None):
        super().__init__()

    def forward(self, X, Y):
        X_pc, X_pi = X
        Y_pc, Y_pi = Y

        # TODO: make configurable
        X_dist = torch.cdist(X_pc, X_pc)
        Y_dist = torch.cdist(Y_pc, Y_pc)

        X_dist = X_dist / X_dist.max()
        Y_dist = Y_dist / Y_dist.max()

        X_sig_X = self._select_distances_from_generators(X_dist, X_pi[0][0])
        X_sig_Y = self._select_distances_from_generators(X_dist, Y_pi[0][0])
        Y_sig_X = self._select_distances_from_generators(Y_dist, X_pi[0][0])
        Y_sig_Y = self._select_distances_from_generators(Y_dist, Y_pi[0][0])

        XY_dist = (X_sig_X - Y_sig_X).pow(2).sum()
        YX_dist = (Y_sig_Y - X_sig_Y).pow(2).sum()

        return XY_dist + YX_dist

    def _select_distances_from_generators(self, dist, gens):
        # TODO: Incorporate more than edge information.
        selected_distances = dist[gens[:, 1], gens[:, 2]]
        return selected_distances
