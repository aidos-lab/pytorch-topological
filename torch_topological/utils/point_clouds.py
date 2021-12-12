"""Generate point clouds for demo purposes."""

import numpy as np

import torch


def make_uniform_blob(low=0.0, high=1.0, n=100):
    """Create blob by uniform sampling.

    Parameters
    ----------
    n : int
        Number of points to sample.

    Returns
    -------
    Tensor of shape `(n, 2)`.
    """
    assert low <= high, RuntimeError('low > high')

    X = np.random.uniform(low=low, high=high, size=(n, 2))
    return torch.as_tensor(X)


def make_disk(r=0.9, R=1.0, n=100):
    """Create disk by uniform sampling.

    Parameters
    ----------
    r: float
        Minimum radius, i.e. the radius of the inner circle of a perfect
        sampling.

    R : float
        Maximum radius, i.e. the radius of the outer circle of a perfect
        sampling.

    n : int
        Number of points to sample.

    Returns
    -------
    Tensor of shape `(n, 2)`.
    """
    assert r <= R, RuntimeError('r > R')

    length = np.random.uniform(r, R, size=n)
    angle = np.pi * np.random.uniform(0, 2, size=n)

    x = np.sqrt(length) * np.cos(angle)
    y = np.sqrt(length) * np.sin(angle)

    X = np.vstack((x, y)).T
    return torch.as_tensor(X)
