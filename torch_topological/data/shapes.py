"""Contains sampling routines for various simple geometric objects."""

import numpy as np
import torch

from .utils import embed


def sample_from_disk(n=100, r=0.9, R=1.0):
    """Sample points from disk.

    Parameters
    ----------
    n : int
        Number of points to sample.

    r: float
        Minimum radius, i.e. the radius of the inner circle of a perfect
        sampling.

    R : float
        Maximum radius, i.e. the radius of the outer circle of a perfect
        sampling.

    Returns
    -------
    torch.tensor of shape `(n, 2)`
        Tensor containing the sampled coordinates.
    """
    assert r <= R, RuntimeError('r > R')

    length = np.random.uniform(r, R, size=n)
    angle = np.pi * np.random.uniform(0, 2, size=n)

    x = np.sqrt(length) * np.cos(angle)
    y = np.sqrt(length) * np.sin(angle)

    X = np.vstack((x, y)).T
    return torch.as_tensor(X)


def sample_from_unit_cube(n, d=3, random_state=None):
    """Sample points uniformly from unit cube in `d` dimensions.

    Parameters
    ----------
    n : int
        Number of points to sample

    d : int
        Number of dimensions.

    random_state : `np.random.RandomState` or int
        Optional random state to use for the pseudo-random number
        generator.

    Returns
    -------
    torch.tensor of shape `(n, d)`
        Tensor containing the sampled coordinates.
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.random.uniform(size=(n, d))
    return torch.as_tensor(X)


def sample_from_sphere(n=100, d=2, r=1, noise=None, ambient=None):
    """Sample `n` data points from a `d`-sphere in `d + 1` dimensions.

    Parameters
    -----------
    n : int
        Number of data points in shape.

    d : int
        Dimension of the sphere.

    r : float
        Radius of sphere.

    noise : float or None
        Optional noise factor. If set, data coordinates will be
        perturbed by a standard normal distribution, scaled by
        `noise`.

    ambient : int or None
        Embed the sphere into a space with ambient dimension equal to
        `ambient`. The sphere is randomly rotated into this
        high-dimensional space.

    Returns
    -------
    torch.tensor
        Tensor of sampled coordinates. If `ambient` is set, array will be
        of shape `(n, ambient)`. Else, array will be of shape `(n, d + 1)`.

    Notes
    -----
    This function was originally authored by Nathaniel Saul as part of
    the `tadasets` package. [tadasets]_

    References
    ----------
    .. [tadasets] https://github.com/scikit-tda/tadasets

    """
    data = np.random.randn(n, d+1)

    # Normalize points to the sphere
    data = r * data / np.sqrt(np.sum(data**2, 1)[:, None])

    if noise:
        data += noise * np.random.randn(*data.shape)

    if ambient is not None:
        assert ambient > d
        data = embed(data, ambient)

    return torch.as_tensor(data)


def sample_from_torus(n, d=3, r=1.0, R=2.0, random_state=None):
    """Sample points uniformly from torus and embed it in `d` dimensions.

    Parameters
    ----------
    n : int
        Number of points to sample

    d : int
        Number of dimensions.

    r : float
        Radius of the 'tube' of the torus.

    R : float
        Radius of the torus, i.e. the distance from the centre of the
        'tube' to the centre of the torus.

    random_state : `np.random.RandomState` or int
        Optional random state to use for the pseudo-random number
        generator.

    Returns
    -------
    torch.tensor of shape `(n, d)`
        Tensor of sampled coordinates.
    """
    if random_state is not None:
        np.random.seed(random_state)

    angles = []

    while len(angles) < n:
        x = np.random.uniform(0, 2 * np.pi)
        y = np.random.uniform(0, 1 / np.pi)

        f = (1.0 + (r/R) * np.cos(x)) / (2 * np.pi)

        if y < f:
            psi = np.random.uniform(0, 2 * np.pi)
            angles.append((x, psi))

    X = []

    for theta, psi in angles:
        x = (R + r * np.cos(theta)) * np.cos(psi)
        y = (R + r * np.cos(theta)) * np.sin(psi)
        z = r * np.sin(theta)

        X.append((x, y, z))

    X = np.asarray(X)
    return torch.as_tensor(X)


def sample_from_annulus(n, r, R, random_state=None):
    """Sample points from a 2D annulus.

    This function samples `N` points from an annulus with inner radius `r`
    and outer radius `R`.

    Parameters
    ----------
    n : int
        Number of points to sample

    r : float
        Inner radius of annulus

    R : float
        Outer radius of annulus

    random_state : `np.random.RandomState` or int
        Optional random state to use for the pseudo-random number
        generator.

    Returns
    -------
    torch.tensor of shape `(n, 2)`
        Tensor containing sampled coordinates.
    """
    if r >= R:
        raise RuntimeError(
            'Inner radius must be less than or equal to outer radius'
        )

    if random_state is not None:
        np.random.seed(random_state)

    thetas = np.random.uniform(0, 2 * np.pi, n)

    # Need to sample based on squared radii to account for density
    # differences.
    radii = np.sqrt(np.random.uniform(r ** 2, R ** 2, n))

    X = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
    return torch.as_tensor(X)


# TODO: Improve documentation
def sample_from_double_annulus(n, random_state=None):
    """Sample n points from a double annulus."""
    if random_state is not None:
        np.random.seed(random_state)

    X = []
    y = []

    for i in range(n):
        while True:
            t = [
                    np.random.uniform(-50, 50, 1)[0],
                    np.random.uniform(-50, 140, 1)[0]
            ]

            d = np.sqrt(np.dot(t, t))
            if d <= 50 and d >= 20:
                X.append(t)
                y.append(0)
                break

            d = np.sqrt(t[0] ** 2 + (t[1] - 90) ** 2)
            if d <= 50 and d >= 40:
                X.append(t)
                y.append(1)
                break

    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return np.asarray(X), np.asarray(y)
