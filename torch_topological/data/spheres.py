"""Sphere(s) data set generation functions."""

import numpy as np

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
