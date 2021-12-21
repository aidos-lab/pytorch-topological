"""Example of topology-based GANs."""

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch_topological.data import make_annulus

from torch_topological.nn import SummaryStatisticLoss
from torch_topological.nn import VietorisRips

import numpy as np
import matplotlib.pyplot as plt


class AnnulusDataset(Dataset):
    def __init__(self, n_samples=100, N=100, r=0.75, R=1.0):
        X = [
            torch.as_tensor(make_annulus(N, r, R)[0]) for i in range(n_samples)
        ]

        X = torch.stack(X)
        X = torch.as_tensor(X, dtype=torch.float)

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
                torch.nn.LeakyReLU(0.2)
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
            torch.nn.Linear(input_dim, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        # Flatten point cloud
        x = x.view(x.size(0), -1)
        return self.model(x)


class AdversarialLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.vr = VietorisRips(dim=1)
        self.loss = SummaryStatisticLoss('polynomial_function', p=2, q=2)

    def forward(self, real, synthetic):

        loss = 0.0

        for x, y in zip(real, synthetic):
            pi_real = self.vr(x)
            pi_synthetic = self.vr(y)

            loss += self.loss(pi_real, pi_synthetic)

        return loss


if __name__ == '__main__':

    n_epochs = 200
    shape = (100, 2)
    latent_dim = 200

    data_loader = DataLoader(
        AnnulusDataset(),
        shuffle=True,
        batch_size=8,
    )

    generator = Generator(shape=shape, latent_dim=latent_dim)
    discriminator = Discriminator(shape=shape)
    adversarial_loss = torch.nn.BCELoss() # AdversarialLoss()

    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    for epoch in range(n_epochs):
        for batch, (pc_real, _) in enumerate(data_loader):
            z = torch.autograd.Variable(
                torch.Tensor(
                    np.random.normal(
                        0,
                        1,
                        (pc_real.shape[0], latent_dim)
                    )
                )
            )

            real_labels = torch.as_tensor([1.0] * len(pc_real)).view(-1, 1)
            fake_labels = torch.as_tensor([0.0] * len(pc_real)).view(-1, 1)

            opt_g.zero_grad()

            pc_synthetic = generator(z)

            #generator_loss = adversarial_loss(pc_real, pc_synthetic)
            generator_loss = adversarial_loss(
                discriminator(pc_synthetic), real_labels
            )
            generator_loss.backward()

            opt_g.step()

            opt_d.zero_grad()

            real_loss = adversarial_loss(discriminator(pc_real), real_labels)
            fake_loss = adversarial_loss(
                discriminator(pc_synthetic).detach(), fake_labels
            )

            discriminator_loss = 0.5 * (real_loss + fake_loss)
            discriminator_loss.backward()

            opt_d.step()


    output = pc_synthetic.detach().numpy()

    for X_ in output:
        plt.scatter(X_[:, 0], X_[:, 1])

    plt.show()
