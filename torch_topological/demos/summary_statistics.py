"""Demo for summary statistics minimisation of a point cloud.

This example demonstrates how to use various topological summary
statistics in order to change the shape of an input point cloud.
"""

import argparse

import matplotlib.pyplot as plt

from torch_topological.data import sample_from_disk

from torch_topological.nn import SummaryStatisticLoss
from torch_topological.nn import VietorisRipsComplex

from tqdm import tqdm

import torch
import torch.optim as optim


def main(args):
    """Run example."""
    n = args.n_samples
    n_iterations = args.n_iterations

    X = sample_from_disk(n=n, r=0.5, R=0.6)
    Y = sample_from_disk(n=n, r=0.9, R=1.0)

    X = torch.nn.Parameter(torch.as_tensor(X), requires_grad=True)

    # loss = ModelSpaceLoss(X, Y, loss=SummaryStatisticLoss)
    vr = VietorisRipsComplex(dim=2)
    pi_target = vr(Y)
    loss_fn = SummaryStatisticLoss('polynomial_function', p=2, q=2)
    #loss = SummaryStatisticLoss('polynomial_function', p=2, q=2)
    opt = optim.SGD([X], lr=0.05)

    progress = tqdm(range(n_iterations))

    for i in progress:
        pi_source = vr(X)

        loss = loss_fn(pi_source, pi_target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        progress.set_postfix(loss=f'{loss.item():.08f}')

    X = X.detach().numpy()

    plt.scatter(X[:, 0], X[:, 1], label='Source')
    plt.scatter(Y[:, 0], Y[:, 1], label='Target')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--n-iterations',
        default=250,
        type=int,
        help='Number of iterations'
    )

    parser.add_argument(
        '-n', '--n-samples',
        default=100,
        type=int,
        help='Number of samples in point clouds'
    )

    parser.add_argument(
        '-s', '--statistic',
        choices=[
            'persistent_entropy',
            'polynomial_function',
            'total_persistence',
        ],
        help='Name of summary statistic to use for the loss'
    )

    parser.add_argument(
        '-p',
        type=float,
        help='Outer exponent for summary statistic loss calculation'
    )

    parser.add_argument(
        '-q',
        type=float,
        help='Inner exponent for summary statistic loss calculation. Will '
             'only be used for certain summary statistics.'
    )

    args = parser.parse_args()
    main(args)
