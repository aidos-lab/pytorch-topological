"""Demo for summary statistics minimisation of a point cloud.

This example demonstrates how to use various topological summary
statistics in order to change the shape of an input point cloud.
The script can either demonstrate how to adjust the shape of two
point clouds, i.e. using a summary statistic as a loss function,
or how to change the shape of a *single* point cloud. By default
two point clouds will be used.
"""

import argparse

import matplotlib.pyplot as plt

from torch_topological.data import sample_from_disk
from torch_topological.data import sample_from_unit_cube

from torch_topological.nn import SummaryStatisticLoss
from torch_topological.nn import VietorisRipsComplex

from tqdm import tqdm

import torch
import torch.optim as optim


def main(args):
    """Run example."""
    n = args.n_samples
    n_iterations = args.n_iterations
    statistic = args.statistic
    p = args.p
    q = args.q

    vr = VietorisRipsComplex(dim=2)

    if args.single:
        X = sample_from_unit_cube(n=n, d=2)
        Y = X.clone()
    else:
        X = sample_from_disk(n=n, r=0.5, R=0.6)
        Y = sample_from_disk(n=n, r=0.9, R=1.0)
        pi_target = vr(Y)

    # Make source point cloud adjustable by treating it as a parameter.
    # This enables topological loss functions to influence the shape of
    # `X`.
    X = torch.nn.Parameter(torch.as_tensor(X), requires_grad=True)

    loss_fn = SummaryStatisticLoss(
        summary_statistic=statistic,
        p=p,
        q=q
    )

    opt = optim.SGD([X], lr=0.05)

    progress = tqdm(range(n_iterations))

    for i in progress:
        pi_source = vr(X)

        if not args.single:
            loss = loss_fn(pi_source, pi_target)
        else:
            loss = loss_fn(pi_source)

        opt.zero_grad()
        loss.backward()
        opt.step()

        progress.set_postfix(loss=f'{loss.item():.08f}')

    X = X.detach().numpy()

    if args.single:
        plt.scatter(X[:, 0], X[:, 1], label='Result')
        plt.scatter(Y[:, 0], Y[:, 1], label='Initial')
    else:
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
        default='polynomial_function',
        help='Name of summary statistic to use for the loss'
    )

    parser.add_argument(
        '-S', '--single',
        action='store_true',
        help='If set, uses only a single point cloud'
    )

    parser.add_argument(
        '-p',
        type=float,
        default=2.0,
        help='Outer exponent for summary statistic loss calculation'
    )

    parser.add_argument(
        '-q',
        type=float,
        default=2.0,
        help='Inner exponent for summary statistic loss calculation. Will '
             'only be used for certain summary statistics.'
    )

    args = parser.parse_args()
    main(args)
