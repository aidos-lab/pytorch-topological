"""Contains sampling routines for various simple geometric objects."""

import numpy as np


def sample_from_unit_cube(N, d=3, random_state=None):
    """Sample points uniformly from unit cube in `d` dimensions.

    Parameters
    ----------
    N : int
        Number of points to sample

    d : int
        Number of dimensions.

    random_state : `np.random.RandomState` or int
        Optional random state to use for the pseudo-random number
        generator.

    Returns
    -------
    np.array of shape `(N, d)`
        Array of sampled coordinates.
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.random.uniform(size=(N, d))
    return X


def sample_from_torus(N, d=3, r=1.0, R=2.0, random_state=None):
    """Sample points uniformly from torus and embed it in `d` dimensions.

    Parameters
    ----------
    N : int
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
    np.array of shape `(N, d)`
        Array of sampled coordinates.
    """
    if random_state is not None:
        np.random.seed(random_state)

    angles = []

    while len(angles) < N:
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
    return X


def sample_from_annulus(N, r, R, random_state=None):
    """Sample points from a 2D annulus.

    This function samples `N` points from an annulus with inner radius `r`
    and outer radius `R`.

    Parameters
    ----------
    N : int
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
    np.array of shape `(N, 2)`
        Array of sampled coordinates.
    """
    if r >= R:
        raise RuntimeError(
            'Inner radius must be less than or equal to outer radius'
        )

    if random_state is not None:
        np.random.seed(random_state)

    thetas = np.random.uniform(0, 2 * np.pi, N)

    # Need to sample based on squared radii to account for density
    # differences.
    radii = np.sqrt(np.random.uniform(r ** 2, R ** 2, N))

    X = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
    return X, np.linspace(0, 1, N)


# TODO: Improve documentation
def sample_from_double_annulus(N, random_state=None):
    """Sample N points from a double annulus."""
    if random_state is not None:
        np.random.seed(random_state)

    X = []
    y = []

    for i in range(N):
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
