"""Data sets based on simple topological shapes."""

from torch_topological.data import sample_from_sphere
from torch_topological.data import sample_from_torus

from torch.utils.data import Dataset

import torch


class SphereVsTorus(Dataset):
    """Data set containing point cloud samples from spheres and tori."""

    def __init__(
        self,
        n_point_clouds=500,
        n_samples=100,
        shuffle=True
    ):
        """Create new instance of the data set.

        Parameters
        ----------
        n_point_clouds : int
            Number of point clouds to generate. Each point cloud will
            consist of `n_samples` samples.

        n_samples : int
            Number of samples to use for each of the `n_point_clouds`
            point clouds contained in the data set.

        shuffle : bool
            If set, shuffles point clouds. Else, point clouds will be
            stored in the order of their creation.
        """
        self.n_point_clouds = n_point_clouds
        self.n_samples = n_samples

        n_spheres = n_point_clouds // 2
        n_tori = n_point_clouds - n_spheres

        spheres = torch.stack([
            sample_from_sphere(self.n_samples) for i in range(n_spheres)
        ])

        tori = torch.stack([
            sample_from_torus(self.n_samples) for i in range(n_tori)
        ])

        labels = torch.as_tensor(
            [0] * n_spheres + [1] * n_tori,
            dtype=torch.long
        )

        self.data = torch.vstack((spheres, tori))
        self.labels = labels

        if shuffle:
            perm = torch.randperm(self.n_point_clouds)

            self.data = self.data[perm]
            self.labels = self.labels[perm]

    def __getitem__(self, index):
        """Get point cloud at `index`.

        Accesses the point cloud stored at `index` and returns it as
        well as its corresponding label.

        Parameters
        ----------
        index : int
            Index of samples to access.

        Returns
        -------
        Tuple of torch.tensor, torch.tensor
            Point cloud at index `index` and its label.
        """
        return self.data[index], self.labels[index]

    def __len__(self):
        """Get number of point clouds stored in data set.

        Returns
        -------
        int
            Number of point clouds stored in this instance of the class.
        """
        return len(self.data)
