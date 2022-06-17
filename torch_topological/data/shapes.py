"""Contains sampling routines for various simple geometric objects."""

import numpy as np
import torch

from .utils import embed


def sample_from_disk(n=100, r=0.9, R=1.0, seed=None):
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

    seed : int, instance of `np.random.Generator`, or `None`
        Seed for the random number generator, or an instance of such
        a generator. If set to `None`, the default random number
        generator will be used.

    Returns
    -------
    torch.tensor of shape `(n, 2)`
        Tensor containing the sampled coordinates.
    """
    assert r <= R, RuntimeError('r > R')

    rng = np.random.default_rng(seed)

    length = rng.uniform(r, R, size=n)
    angle = np.pi * rng.uniform(0, 2, size=n)

    x = np.sqrt(length) * np.cos(angle)
    y = np.sqrt(length) * np.sin(angle)

    X = np.vstack((x, y)).T
    return torch.as_tensor(X)


def sample_from_unit_cube(n, d=3, seed=None):
    """Sample points uniformly from unit cube in `d` dimensions.

    Parameters
    ----------
    n : int
        Number of points to sample

    d : int
        Number of dimensions.

    seed : int, instance of `np.random.Generator`, or `None`
        Seed for the random number generator, or an instance of such
        a generator. If set to `None`, the default random number
        generator will be used.

    Returns
    -------
    torch.tensor of shape `(n, d)`
        Tensor containing the sampled coordinates.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))

    return torch.as_tensor(X)


def sample_from_sphere(n=100, d=2, r=1, noise=None, ambient=None, seed=None):
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

    seed : int, instance of `np.random.Generator`, or `None`
        Seed for the random number generator, or an instance of such
        a generator. If set to `None`, the default random number
        generator will be used.

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
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d+1))

    # Normalize points to the sphere
    data = r * data / np.sqrt(np.sum(data**2, 1)[:, None])

    if noise:
        data += noise * rng.standard_normal(data.shape)

    if ambient is not None:
        assert ambient > d
        data = embed(data, ambient)

    return torch.as_tensor(data)


def sample_from_torus(n, d=3, r=1.0, R=2.0, seed=None):
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

    seed : int, instance of `np.random.Generator`, or `None`
        Seed for the random number generator, or an instance of such
        a generator. If set to `None`, the default random number
        generator will be used.

    Returns
    -------
    torch.tensor of shape `(n, d)`
        Tensor of sampled coordinates.
    """
    rng = np.random.default_rng(seed)
    angles = []

    while len(angles) < n:
        x = rng.uniform(0, 2 * np.pi)
        y = rng.uniform(0, 1 / np.pi)

        f = (1.0 + (r/R) * np.cos(x)) / (2 * np.pi)

        if y < f:
            psi = rng.uniform(0, 2 * np.pi)
            angles.append((x, psi))

    X = []

    for theta, psi in angles:
        a = R + r * np.cos(theta)
        x = a * np.cos(psi)
        y = a * np.sin(psi)
        z = r * np.sin(theta)

        X.append((x, y, z))

    X = np.asarray(X)
    return torch.as_tensor(X)


def sample_from_annulus(n, r, R, seed=None):
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

    seed : int, instance of `np.random.Generator`, or `None`
        Seed for the random number generator, or an instance of such
        a generator. If set to `None`, the default random number
        generator will be used.

    Returns
    -------
    torch.tensor of shape `(n, 2)`
        Tensor containing sampled coordinates.
    """
    if r >= R:
        raise RuntimeError(
            'Inner radius must be less than or equal to outer radius'
        )

    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0, 2 * np.pi, n)

    # Need to sample based on squared radii to account for density
    # differences.
    radii = np.sqrt(rng.uniform(r ** 2, R ** 2, n))

    X = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
    return torch.as_tensor(X)
