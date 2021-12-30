"""Sphere(s) data set generation functions."""

import numpy as np

from torch_topological.data.utils import embed


def sample_from_sphere(n=100, d=2, r=1, noise=None, ambient=None):
    """Sample `n` data points from a sphere in `d` dimensions.

    Parameters
    -----------
    n : int
        Number of data points in shape.

    d : int
        Dimension of the sphere.

    r : float
        Radius of sphere.

    ambient : int or None
        Embed the sphere into a space with ambient dimension equal to
        `ambient`. The sphere is randomly rotated into this
        high-dimensional space.

    Returns
    -------
    np.array
        Array of sampled coordinates. If `ambient` is set, array will be
        of shape `(n, ambient)`. Else, array will be of shape `(n, d)`.

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

    return data


# TODO: Finish documentation
def create_sphere_dataset(n_samples=500, d=100, n_spheres=11, r=5, seed=42):
    """Create data set of high-dimensional spheres.

    Create 'SPHERES' data set described in Moor et al. [Moor20a]_.

    Notes
    -----
    This code was originally authored by Michael Moor.

    References
    ----------
    .. [Moor20a] M. Moor et al., "Topological Autoencoders",
        *Proceedings of the 37th International Conference on Machine
        Learning*, PMLR 119, pp. 7045--7054, 2020.
    """
    np.random.seed(seed)

    variance = 10 / np.sqrt(d)
    shift_matrix = np.random.normal(0, variance, [n_spheres, d+1])

    spheres = []
    n_datapoints = 0
    for i in np.arange(n_spheres-1):
        sphere = sample_from_sphere(n=n_samples, d=d, r=r)
        spheres.append(sphere + shift_matrix[i, :])
        n_datapoints += n_samples

    # Build additional large surrounding sphere:
    n_samples_big = 10 * n_samples
    big = sample_from_sphere(n=n_samples_big, d=d, r=r*5)
    spheres.append(big)
    n_datapoints += n_samples_big

    X = np.concatenate(spheres, axis=0)
    y = np.zeros(n_datapoints)

    label_index = 0

    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        y[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples

    return X, y
