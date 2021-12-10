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

    # TODO: Add weight functions
    def __init__(self, Y=None, summary_statistic='total_persistence'):
        """Create new loss function based on summary statistic.

        Parameters
        ----------
        Y : `torch.tensor` or `None`
            Optional target tensor. If set, evaluates a difference in
            loss functions as shown in the introduction. If `None`, a
            simpler variant of the loss will be evaluated.

        summary_statistic : str
            Indicates which summary statistic function to use.
        """
        super().__init__()

        import pytorch_topological.utils.summary_statistics as stat

        self.Y = Y
        self.stat_fn = getattr(stat, summary_statistic, None)

    # TODO: improve documentation
    def forward(self, X):
        """Calculate loss based on input tensor."""
        stat_src = torch.sum(
            torch.stack([
                self.stat_fn(D) for D in X
            ])
        )

        if self.Y is not None:
            stat_target = torch.sum(
                torch.stack([
                    self.stat_fn(D) for D in self.Y
                ])
            )

            return (stat_target - stat_src).abs()
        else:
            return stat_src.abs()
