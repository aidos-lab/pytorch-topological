"""Example demonstrating the computation of alpha complexes.

This simple example demonstrates how to use alpha complexes to change
the appearance of a point cloud, following the `TopologyLayer
<https://github.com/bruel-gabrielsson/TopologyLayer>`_ package.

This example is still a **work in progress**.
"""

from torch_topological.nn import AlphaComplex
from torch_topological.nn import SummaryStatisticLoss

import numpy as np
import matplotlib.pyplot as plt

import torch

if __name__ == '__main__':
    np.random.seed(42)
    data = np.random.rand(100, 2)

    alpha_complex = AlphaComplex()

    loss_fn = SummaryStatisticLoss(
        summary_statistic='polynomial_function',
        p=1,
        q=2
    )

    X = torch.nn.Parameter(torch.as_tensor(data), requires_grad=True)
    opt = torch.optim.Adam([X], lr=1e-2)

    for i in range(100):
        loss = loss_fn(alpha_complex(X))

        opt.zero_grad()
        loss.backward()
        opt.step()

    X = X.detach().numpy()

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
