"""Utility functions for data set generation."""

import numpy as np


def embed(data, ambient=50):
    """Embed `data` in `ambient` dimensions.

    Parameters
    ----------
    data : array_like
        Input data set; needs to have shape `(n, d)`, i.e. samples are
        in the rows, dimensions are in the columns.

    ambient : int
        Dimension of embedding space. Must be greater than
        dimensionality of data.

    Returns
    -------
    array_like
        Input array of shape `(n, D)`, with `D = ambient`.

    Notes
    -----
    This function was originally authored by Nathaniel Saul as part of
    the `tadasets` package. [tadasets]_

    References
    ----------
    .. [tadasets] https://github.com/scikit-tda/tadasets
    """
    n, d = data.shape
    assert ambient > d

    base = np.zeros((n, ambient))
    base[:, :d] = data

    # construct a rotation matrix of dimension `ambient`.
    random_rotation = np.random.random((ambient, ambient))
    q, r = np.linalg.qr(random_rotation)

    base = np.dot(base, q)
    return base
