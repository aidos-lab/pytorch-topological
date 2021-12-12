"""Demo for total persistence minimisation of a point cloud."""


import logging

import pytorch_topological.utils as utils

from pytorch_topological.nn import SummaryStatisticLoss
from pytorch_topological.nn import VietorisRips

import torch
import torch.optim as optim


import matplotlib.pyplot as plt


if __name__ == '__main__':
    n = 100

    X = utils.make_disk(r=0.5, R=0.6, n=n)
    Y = utils.make_disk(r=0.9, R=1.0, n=n)

    X = torch.nn.Parameter(torch.as_tensor(X), requires_grad=True)

    # loss = ModelSpaceLoss(X, Y, loss=SummaryStatisticLoss)
    vr = VietorisRips(X, Y)
    loss = SummaryStatisticLoss()
    opt = optim.SGD([X], lr=0.05)

    logging.basicConfig(
        level=logging.INFO
    )

    losses = []

    for i in range(100):
        pd_source, pd_target = vr()

        l = loss(pd_source, pd_target)

        l.backward()
        opt.step()
        opt.zero_grad()

        l = l.detach().numpy()

        logging.info(l)
        losses.append(l)

    X = X.detach().numpy()

    plt.scatter(X[:, 0], X[:, 1], label='Source')
    plt.scatter(Y[:, 0], Y[:, 1], label='Target')

    plt.legend()
    plt.show()

