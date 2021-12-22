"""Demonstrate image smoothing based on topology."""

import numpy as np
import matplotlib.pyplot as plt

from torch_topological.nn import Cubical
from torch_topological.nn import SummaryStatisticLoss

from sklearn.datasets import make_circles

import torch


def _make_data(n_cells, n_samples=1000):
    X = make_circles(n_samples, shuffle=True, noise=0.05)[0]

    heatmap, *_ = np.histogram2d(X[:, 0], X[:, 1], bins=n_cells)
    heatmap -= heatmap.mean()
    heatmap /= heatmap.max()

    return heatmap


class TopologicalSimplification(torch.nn.Module):
    def __init__(self, theta):
        super().__init__()

        self.theta = theta

    def forward(self, x):
        persistence_information = cubical(x)
        persistence_information = [persistence_information[0]]

        gens, pd = persistence_information[0]

        persistence = (pd[:, 1] - pd[:, 0]).abs()
        indices = persistence <= self.theta

        gens = gens[indices]

        indices = torch.vstack((gens[:, 0:2], gens[:, 2:]))

        indices = np.ravel_multi_index(
            (indices[:, 0], indices[:, 1]), x.shape
        )

        x.ravel()[indices] = 0.0

        persistence_information = cubical(x)
        persistence_information = [persistence_information[0]]

        return x, persistence_information


if __name__ == '__main__':

    np.random.seed(23)

    Y = _make_data(50)
    Y = torch.as_tensor(Y, dtype=torch.float)
    X = torch.as_tensor(
        Y + np.random.normal(scale=0.05, size=Y.shape), dtype=torch.float
    )

    theta = torch.nn.Parameter(
        torch.as_tensor(1.0), requires_grad=True,
    )

    topological_simplification = TopologicalSimplification(theta)

    optimizer = torch.optim.Adam(
        [theta], lr=1e-2
    )
    loss_fn = SummaryStatisticLoss('total_persistence', p=1)

    cubical = Cubical()

    persistence_information_target = cubical(Y)
    persistence_information_target = [persistence_information_target[0]]

    for i in range(500):
        X, persistence_information = topological_simplification(X)

        optimizer.zero_grad()

        loss = loss_fn(
            persistence_information,
            persistence_information_target
        )

        print(loss.item(), theta.item())

        theta.backward()
        optimizer.step()

    X = X.detach().numpy()

    plt.imshow(X)
    plt.show()
