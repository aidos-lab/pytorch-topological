"""Create `SPHERES` data set."""

import numpy as np

import torch

from torch.utils.data import Dataset
from torch.utils.data import random_split

from torch_topological.data import sample_from_sphere


# TODO: Finish documentation
# TODO: Harmonise API
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


class Spheres(Dataset):
    def __init__(
        self,
        train=True,
        n_samples=100,
        n_spheres=11,
        r=5,
        test_fraction=0.1,
        seed=42
    ):
        X, y = create_sphere_dataset(
                n_samples=n_samples,
                n_spheres=n_spheres,
                r=r,
                seed=seed)

        X = torch.as_tensor(X, dtype=torch.float)

        test_size = int(test_fraction * len(X))
        train_size = len(X) - test_size

        indices = torch.as_tensor(np.arange(len(X)), dtype=torch.long)

        train_indices, test_indices = random_split(
            indices, [train_size, test_size]
        )

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        self.data = X_train if train else X_test
        self.labels = y_train if train else y_test

        self.dimension = X.shape[1]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

