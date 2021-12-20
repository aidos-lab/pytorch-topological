"""Example of topology-based GANs."""

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch_topological.data import make_annulus

import numpy as np
import matplotlib.pyplot as plt


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


class Discriminator(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()

        input_dim = np.prod(shape)

        # Inspired by the original GAN. THERE CAN ONLY BE ONE!
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        # Flatten point cloud
        x = x.view(x.size(0), -1)
        return self.model(x)


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
    discriminator = Discriminator(shape=shape)

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

            point_clouds = generator(z)
            quasi_probs = discriminator(point_clouds)

            print('z.shape =', z.shape)
            print('point_clouds.shape =', point_clouds.shape)
            print('quasi_probs.shape =', quasi_probs.shape)

    # TODO: This is a rather stupid way of visualising the output of
    # this training routine while writing it. I hope to *eventually*
    # get rid of this...
    pc = point_clouds.detach().numpy()[0]

    plt.scatter(pc[:, 0], pc[:, 1])
    plt.show()
