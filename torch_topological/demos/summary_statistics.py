"""Demo for summary statistics minimisation of a point cloud.

This example demonstrates how to use various topological summary
statistics in order to change the shape of an input point cloud.
"""

import logging

from torch_topological.data import sample_from_disk

from torch_topological.nn import SummaryStatisticLoss
from torch_topological.nn import VietorisRipsComplex

from tqdm import tqdm

import torch
import torch.optim as optim

import matplotlib.pyplot as plt


if __name__ == '__main__':
    n = 100

    X = sample_from_disk(n=n, r=0.5, R=0.6)
    Y = sample_from_disk(n=n, r=0.9, R=1.0)

    X = torch.nn.Parameter(torch.as_tensor(X), requires_grad=True)

    # loss = ModelSpaceLoss(X, Y, loss=SummaryStatisticLoss)
    vr = VietorisRipsComplex(dim=2)
    pi_target = vr(Y)
    loss = SummaryStatisticLoss('polynomial_function', p=2, q=2)
    #loss = SummaryStatisticLoss('polynomial_function', p=2, q=2)
    opt = optim.SGD([X], lr=0.05)

    logging.basicConfig(
        level=logging.INFO
    )

    losses = []

    for i in range(500):
        pi_source = vr(X)

        l = loss(pi_source, pi_target)

        opt.zero_grad()

        l.backward()
        opt.step()

        l = l.detach().numpy()

        logging.info(l)
        losses.append(l)

    X = X.detach().numpy()

    plt.scatter(X[:, 0], X[:, 1], label='Source')
    plt.scatter(Y[:, 0], Y[:, 1], label='Target')

    plt.legend()
    plt.show()

