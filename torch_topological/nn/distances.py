"""Distance calculation modules between topological descriptors."""

import ot
import torch

from torch_topological.utils import wrap_if_not_iterable


class WassersteinDistance(torch.nn.Module):
    """Implement Wasserstein distance between persistence diagrams.

    This module calculates the Wasserstein between two persistence
    diagrams. The Wasserstein distance is arguably the most common
    metric that is applied when dealing with such diagrams. Notice
    that calculating the metric involves solving optimal transport
    problems, which are known to suffer from scalability problems.
    When dealing with large persistence diagrams, other losses may
    be more appropriate.
    """

    def __init__(self, p=torch.inf, q=1):
        """Create new Wasserstein distance calculation module.

        Parameters
        ----------
        p : float or `inf`
            Specifies the exponent of the norm to calculate. By default,
            `p = torch.inf`, corresponding to the *maximum norm*.

        q: float
            Specifies the order of Wasserstein metric to calculate. This
            raises all internal matching costs to the power of `q`, hence
            subsequently returning the `q`-th root of the total cost.
        """
        super().__init__()

        self.p = p
        self.q = q

    def _project_to_diagonal(self, diagram):
        x = diagram[:, 0]
        y = diagram[:, 1]

        # TODO: Is this the closest point in all p-norms?
        return 0.5 * torch.stack(((x + y), (x + y)), 1)

    def _distance_to_diagonal(self, diagram):
        return torch.linalg.vector_norm(
            diagram - self._project_to_diagonal(diagram),
            self.p,
            dim=1
        )

    def _make_distance_matrix(self, D1, D2):
        dist_D11 = self._distance_to_diagonal(D1)
        dist_D22 = self._distance_to_diagonal(D2)

        # n x m matrix containing the distances between 'regular'
        # persistence pairs of both persistence diagrams.
        dist = torch.cdist(D1, D2, p=torch.inf)

        # Extend the matrix with a column of distances of samples in D1
        # to their respective projection on the diagonal.
        upper_blocks = torch.hstack((dist, dist_D11[:, None]))

        # Create a lower row of distances of samples in D2 to their
        # respective projection on the diagonal. The ordering needs
        # to follow the ordering of samples in D2. Note how one `0`
        # needs to be added to the row in order to balance it. The
        # entry intuitively describes the cost between *projected*
        # points, so it has to be zero.
        lower_blocks = torch.cat(
            (dist_D22, torch.tensor(0, device=dist_D22.device).unsqueeze(0))
        )

        # Full (n + 1 ) x (m + 1) matrix containing *all* distances. By
        # construction, M[[i, n] contains distances to projected points
        # in D1, whereas M[m, j] does the same for points in D2. Only a
        # cell M[i, j] with 0 <= i < n and 0 <= j < m contains a proper
        # distance.
        M = torch.vstack((upper_blocks, lower_blocks))
        M = M.pow(self.q)

        return M

    def forward(self, X, Y):
        """Calculate Wasserstein metric based on input tensors.

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
            A single scalar tensor containing the distance between the
            persistence diagram(s) contained in `X` and `Y`.
        """
        total_cost = 0.0

        X = wrap_if_not_iterable(X)
        Y = wrap_if_not_iterable(Y)

        for pers_info in zip(X, Y):
            D1 = pers_info[0].diagram
            D2 = pers_info[1].diagram

            n = len(D1)
            m = len(D2)

            dist = self._make_distance_matrix(D1, D2)

            # Create weight vectors. Since the last entries of entries
            # describe the m points coming from D2, we have to set the
            # last entry accordingly.

            a = torch.ones(n + 1, device=dist.device)
            b = torch.ones(m + 1, device=dist.device)

            a[-1] = m
            b[-1] = n

            # TODO: Make settings configurable?
            total_cost += ot.emd2(a, b, dist)

        return total_cost.pow(1.0 / self.q)
