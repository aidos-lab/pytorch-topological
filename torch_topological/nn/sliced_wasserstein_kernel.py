"""Sliced Wasserstein kernel implementation."""


import torch

from torch_topological.nn import SlicedWassersteinDistance

from torch_topological.utils import wrap_if_not_iterable


class SlicedWassersteinKernel(torch.nn.Module):
    """Calculate sliced Wasserstein kernel between persistence diagrams.

    This is an implementation of the sliced Wasserstein kernel between
    persistence diagrams, following [Carriere17a]_.

    References
    ----------
    .. [Carriere17a] M. Carrière et al., "Sliced Wasserstein Kernel for
       Persistence Diagrams", *Proceedings of the 34th International
       Conference on Machine Learning*, PMLR 70, pp. 664--673, 2017.
    """

    def __init__(self, num_directions=10, sigma=1.0):
        """Create new sliced Wasserstein kernel module.

        Parameters
        ----------
        num_directions : int
            Specifies the number of random directions to be sampled for
            computation of the sliced Wasserstein distance.

        sigma : int
            Variance term of the sliced Wasserstein kernel expression.
        """
        super().__init__()

        self.num_directions = num_directions
        self.sigma = sigma

    def forward(self, X, Y):
        """Calculate sliced Wasserstein kernel based on input tensors.

        Parameters
        ----------
        X : list or instance of :class:`PersistenceInformation`
            Topological features of the first space. Supposed to contain
            persistence diagrams and persistence pairings.

        Y : list or instance of :class:`PersistenceInformation`
            Topological features of the second space. Supposed to
            contain persistence diagrams and persistence pairings.

        Returns
        -------
        torch.tensor
            A single scalar tensor containing the sliced Wasserstein kernel
            between the persistence diagram(s) contained in `X` and `Y`.
        """
        total_cost = 0.0

        X = wrap_if_not_iterable(X)
        Y = wrap_if_not_iterable(Y)

        swd = SlicedWassersteinDistance(num_directions=self.num_directions)

        for pers_info in zip(X, Y):
            D1 = pers_info[0]
            D2 = pers_info[1]

            total_cost += torch.exp(-swd(D1, D2)) / self.sigma

        return total_cost
