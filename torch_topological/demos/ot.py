"""Check optimal transport integration."""


import numpy as np

import ot
import torch

from ot.datasets import make_1D_gauss as gauss


class WassersteinDistanceLoss(torch.nn.Module):
    # TODO: q is still unused
    def __init__(self, p=torch.inf, q=1):
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

        dist = torch.cdist(D1, D2, p=torch.inf)

        print('dist =', dist)

        upper_blocks = torch.hstack((dist, dist_D11[:, None]))
        lower_blocks = torch.cat((dist_D22, torch.tensor(0).unsqueeze(0)))

        M = torch.vstack((upper_blocks, lower_blocks))
        print('all_dist =', M)
        print(M.shape)

        return M

    def forward(self, D1, D2):
        n = len(D1)
        m = len(D2)
        M = self._make_distance_matrix(D1, D2)
        a = torch.ones(n + 1)
        b = torch.ones(m + 1)

        return ot.emd2(a, b, M)



PD1 = [
    (0, 0),
    (1, 2),
    (3, 4)
]

PD2 = [
    (0, 0),
    (3, 4),
    (5, 3),
    (7, 4)
]


PD1 = torch.as_tensor(PD1, dtype=torch.float)
PD2 = torch.as_tensor(PD2, dtype=torch.float)
print(PD1)
print(PD2)
print(WassersteinDistanceLoss()._make_distance_matrix(PD1, PD2))
print(WassersteinDistanceLoss()(PD1, PD2))
