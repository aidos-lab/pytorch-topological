"""Loss terms for various optimisation objectives."""

from torch import nn

from pytorch_topological.utils import total_persistence

import torch


class TotalPersistenceLoss(nn.Module):
    """Implement loss based on total persistence."""

    def __init__(self, X, Y=None):
        super().__init__()

        self.X = X
        self.Y = Y

    def forward(self, X):
        total_persistence_src = torch.sum([
                total_persistence(D) for D in self.X
            ]
        )

        if self.Y is not None:
            total_persistence_target = torch.sum([
                    total_persistence(D) for D in self.Y
                ]
            )

            return total_persistence_target - total_persistence_src
        else:
            return total_persistence_src
