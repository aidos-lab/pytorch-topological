"""Demo for distance minimisations of a point cloud.

Note
----
This demonstration is a work in progress. It is not fully documented and
tested yet.
"""

from tqdm import tqdm

from torch_topological.data import sample_from_disk

from torch_topological.nn import VietorisRipsComplex
from torch_topological.nn import WassersteinDistance

import torch
import torch.optim as optim


import matplotlib.pyplot as plt


if __name__ == '__main__':
    n = 100

    X = sample_from_disk(r=0.5, R=0.6, n=n)
    Y = sample_from_disk(r=0.9, R=1.0, n=n)

    X = torch.nn.Parameter(torch.as_tensor(X), requires_grad=True)

    vr = VietorisRipsComplex(dim=1)

    pi_target = vr(Y)
    loss_fn = WassersteinDistance(q=2)

    opt = optim.SGD([X], lr=0.1)

    n_iterations = 500
    progress = tqdm(range(n_iterations))

    for i in progress:

        opt.zero_grad()

        pi_source = vr(X)
        loss = loss_fn(pi_source, pi_target)

        loss.backward()
        opt.step()

        progress.set_postfix(loss=loss.item())

    X = X.detach().numpy()

    plt.scatter(X[:, 0], X[:, 1], label='Source')
    plt.scatter(Y[:, 0], Y[:, 1], label='Target')

    plt.legend()
    plt.show()
