"""Example demonstrating the computation of alpha complexes.

This simple example demonstrates how to use alpha complexes to change
the appearance of a point cloud, following the `TopologyLayer
<https://github.com/bruel-gabrielsson/TopologyLayer>`_ package.

This example is still a **work in progress**.
"""

from torch_topological.nn import AlphaComplex
from torch_topological.nn import SummaryStatisticLoss

from torch_topological.utils import SelectByDimension

import numpy as np
import matplotlib.pyplot as plt

import torch

if __name__ == '__main__':
    np.random.seed(42)
    data = np.random.rand(100, 2)

    alpha_complex = AlphaComplex()

    loss_fn = SummaryStatisticLoss(
        summary_statistic='polynomial_function',
        p=2,
        q=0
    )

    X = torch.nn.Parameter(torch.as_tensor(data), requires_grad=True)
    opt = torch.optim.Adam([X], lr=1e-2)

    for i in range(100):
        # We are only interested in working with persistence diagrams of
        # dimension 1.
        selector = SelectByDimension(1)

        # Let's think step by step; apparently, AIs like that! So let's
        # first get the persistence information of our complex. We pass
        # it through the selector to remove diagrams we do not need.
        pers_info = alpha_complex(X)
        pers_info = selector(pers_info)

        # Evaluate the loss; notice that we want to *maximise* it in
        # order to improve the holes in the data.
        loss = -loss_fn(pers_info)

        opt.zero_grad()
        loss.backward()
        opt.step()

    X = X.detach().numpy()

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
