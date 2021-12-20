"""Contains sampling routines for various simple geometric objects."""

import numpy as np


def make_annulus(N, r, R, **kwargs):
    """Sample points from annulus.

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

    **kwargs:
        Optional keyword arguments, such as a fixed random state for the
        pseudo-random number generator.

    Returns
    -------
    Array of (x, y) coordinates.
    """
    if r >= R:
        raise RuntimeError(
            'Inner radius must be less than or equal to outer radius'
        )

    if kwargs.get('random_state'):
        np.random.seed(kwargs['random_state'])

    thetas = np.random.uniform(0, 2 * np.pi, N)

    # Need to sample based on squared radii to account for density
    # differences.
    radii = np.sqrt(np.random.uniform(r ** 2, R ** 2, N))

    X = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
    return X, np.linspace(0, 1, N)


# TODO: Improve documentation
# TODO: Improve `random_state` handling
def make_double_annulus(N, random_state=None):
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
