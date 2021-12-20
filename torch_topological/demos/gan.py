"""Example of topology-based GANs."""

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch_topological.data import make_annulus

import numpy as np


class AnnulusDataset(Dataset):
    def __init__(self, n_samples=100, N=100, r=0.75, R=1.0):
        X = [
            torch.as_tensor(make_annulus(N, r, R)[0]) for i in range(n_samples)
        ]

        X = torch.stack(X)

        self.data = X
        self.labels = [1] * len(X)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class Generator(torch.nn.Module):
    def __init__(self, latent_dim=100, shape=(100, 2)):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = shape[0] * shape[1]

        self.shape = shape

        def _make_layer(input_dim, output_dim):
            layers = [
                torch.nn.Linear(input_dim, output_dim),
                torch.nn.ReLU()
            ]
            return layers

        self.model = torch.nn.Sequential(
            *_make_layer(self.latent_dim, 64),
            *_make_layer(64, 128),
            *_make_layer(128, 256),
            torch.nn.Linear(256, self.output_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, z):
        point_cloud = self.model(z)
        point_cloud = point_cloud.view(point_cloud.size(0), *self.shape)

        return point_cloud


if __name__ == '__main__':

    n_epochs = 10
    shape = (100, 2)
    latent_dim = 100

    data_loader = DataLoader(
        AnnulusDataset(),
        shuffle=True,
        batch_size=32,
    )

    generator = Generator(shape=shape, latent_dim=latent_dim)

    for epoch in range(n_epochs):
        for batch, (point_cloud, _) in enumerate(data_loader):
            z = torch.autograd.Variable(
                torch.Tensor(
                    np.random.normal(
                        0,
                        1,
                        (point_cloud.shape[0], latent_dim)
                    )
                )
            )

            print('z.shape =', z.shape)

            point_clouds = generator(z)

            print('point_clouds.shape =', point_clouds.shape)
