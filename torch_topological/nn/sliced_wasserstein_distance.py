"""Sliced Wasserstein distance implementation."""


import torch

import numpy as np

from torch_topological.utils import wrap_if_not_iterable


class SlicedWassersteinDistance(torch.nn.Module):
    """Calculate sliced Wasserstein distance between persistence diagrams.

    This is an implementation of the sliced Wasserstein distance between
    persistence diagrams, following [Carriere17a]_.

    This module calculates the sliced Wasserstein distance between two
    persistence diagrams. It is an efficient variant of the Wasserstein
    distance, and it is commonly used in the Sliced Wasserstein Kernel.
    It computes the expected value of the Wasserstein distance when the
    persistence diagram is projected on a random line passing through
    the origin.

    References
    ----------
    .. [Carriere17a] M. Carrière et al., "Sliced Wasserstein Kernel for
       Persistence Diagrams", *Proceedings of the 34th International
       Conference on Machine Learning*, PMLR 70, pp. 664--673, 2017.
    """

    def __init__(self, num_directions=10):
        """Create new sliced Wasserstein distance calculation module.

        Parameters
        ----------
        num_directions : int
            Specifies the number of random directions to be sampled for
            computation of the sliced Wasserstein distance.
        """
        super().__init__()

        # Generates num_directions number of lines with slopes randomly sampled
        # between -pi/2 and pi/2.
        self.num_directions = num_directions
        thetas = torch.linspace(-np.pi/2, np.pi/2, steps=self.num_directions+1)
        thetas = thetas[:-1]
        self.lines = torch.vstack([torch.tensor([torch.cos(i), torch.sin(i)],
                                  dtype=torch.float32) for i in thetas])

    def _emd1d(self, X, Y):
        # Compute Wasserstein Distance between two 1d-distributions.
        X, ind = torch.sort(X, dim=0)
        Y, ind = torch.sort(Y, dim=0)
        return torch.sum(torch.abs(torch.sub(X, Y)))

    def _project_diagram(self, D1, L):
        # Project persistence diagram D1 onto a given line L.
        return torch.stack([torch.dot(x, L)/torch.dot(L, L) for x in D1])

    def forward(self, X, Y):
        """Calculate sliced Wasserstein metric based on input tensors.

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
            A single scalar tensor containing the sliced Wasserstein distance
            between the persistence diagram(s) contained in `X` and `Y`.
        """
        total_cost = 0.0

        X = wrap_if_not_iterable(X)
        Y = wrap_if_not_iterable(Y)

        for pers_info in zip(X, Y):
            D1 = pers_info[0].diagram.float()
            D2 = pers_info[1].diagram.float()

            # Auxiliary array to project onto diagonal.
            diag = torch.tensor([0.5, 0.5], dtype=torch.float32)

            # Project both the diagrams onto the diagonals.
            D1_diag = torch.vstack([torch.sum(x) * diag for x in D1])
            D2_diag = torch.vstack([torch.sum(x) * diag for x in D2])

            cost = 0.0

            for line in self.lines:
                proj_d1 = self._project_diagram(D1, line)
                proj_d2 = self._project_diagram(D2, line)

                proj_diag_d1 = self._project_diagram(D1_diag, line)
                proj_diag_d2 = self._project_diagram(D2_diag, line)

                cost += self._emd1d(torch.cat([proj_d1, proj_diag_d2]),
                                    torch.cat([proj_d2, proj_diag_d1]))

            cost /= self.num_directions

            total_cost += cost

        return total_cost
