"""Demo for total persistence minimisation of a point cloud."""


import logging

import pytorch_topological.utils as utils
import pytorch_topological.nn.vietoris_rips as vr

import torch.optim as optim


if __name__ == '__main__':
    n = 100

    X = utils.make_disk(r=0.01, R=0.5, n=n)
    Y = utils.make_disk(r=0.90, R=1.0, n=n)

    loss = vr.ModelSpaceLoss(X, Y)
    opt = optim.SGD(loss.parameters(), lr=0.05)

    logging.basicConfig(
        level=logging.INFO
    )

    losses = []

    for i in range(100):
        l = loss()
        l.backward()
        opt.step()
        opt.zero_grad()

        l = l.detach().numpy()

        logging.info(l)
        losses.append(l)