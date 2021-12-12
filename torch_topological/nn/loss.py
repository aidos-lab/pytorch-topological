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

    # TODO: improve documentation
    def forward(self, X, Y=None):
        """Calculate loss based on input tensor(s).

        Parameters
        ----------
        Y : `torch.tensor` or `None`
            Optional target tensor. If set, evaluates a difference in
            loss functions as shown in the introduction. If `None`, a
            simpler variant of the loss will be evaluated.
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
