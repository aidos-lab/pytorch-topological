"""Demo for calculating cubical complexes.

This example demonstrates how to perform topological operations on
a structured array, such as a grey-scale image.
"""

import numpy as np
import matplotlib.pyplot as plt

from torch_topological.nn import CubicalComplex
from torch_topological.nn import SummaryStatisticLoss
from torch_topological.nn import WassersteinDistance

from tqdm import tqdm

import torch


def sample_circles(n_cells, n_samples=1000):
    """Sample two nested circles and bin them.

    Parameters
    ----------
    n_cells : int
        Number of cells for the 2D histogram, i.e. the 'resolution' of
        the histogram.

    n_samples : int
        Number of samples to use for creating the nested circles
        coordinates.

    Returns
    -------
    np.ndarray of shape ``(n_cells, n_cells)``
        Structured array containing intensity values for the data set.
    """
    from sklearn.datasets import make_circles
    X = make_circles(n_samples, shuffle=True, noise=0.01)[0]

    heatmap, *_ = np.histogram2d(X[:, 0], X[:, 1], bins=n_cells)
    heatmap -= heatmap.mean()
    heatmap /= heatmap.max()

    return heatmap


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device', device)

    np.random.seed(23)

    Y = sample_circles(50)
    Y = torch.as_tensor(Y, dtype=torch.float)
    X = torch.as_tensor(
        Y + np.random.normal(scale=0.05, size=Y.shape),
        dtype=torch.float,
        device=device,
    )
    Y = Y.to(device)
    X = torch.nn.Parameter(X, requires_grad=True).to(device)

    optimizer = torch.optim.Adam([X], lr=1e-3)
    loss_fn = SummaryStatisticLoss('total_persistence', p=1)
    loss_fn = WassersteinDistance()

    cubical_complex = CubicalComplex()

    persistence_information_target = cubical_complex(Y)
    persistence_information_target = [persistence_information_target[0]]

    n_iter = 500
    progress = tqdm(range(n_iter))

    for i in progress:
        persistence_information = cubical_complex(X)
        persistence_information = [persistence_information[0]]

        optimizer.zero_grad()

        loss = loss_fn(
            persistence_information,
            persistence_information_target
        )

        loss.backward()
        optimizer.step()

        progress.set_postfix(loss=loss.item())

    X = X.cpu().detach().numpy()

    plt.imshow(X)
    plt.show()
