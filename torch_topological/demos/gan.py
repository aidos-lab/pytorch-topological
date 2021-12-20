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

    n_epochs = 5
    shape = (100, 2)
    latent_dim = 200

    data_loader = DataLoader(
        AnnulusDataset(),
        shuffle=True,
        batch_size=8,
    )

    generator = Generator(shape=shape, latent_dim=latent_dim)
    discriminator = Discriminator(shape=shape)
    adversarial_loss = AdversarialLoss()

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

            opt_g.zero_grad()

            pc_synthetic = generator(z)
            quasi_probs = discriminator(pc_synthetic)

            loss = adversarial_loss(pc_real, pc_synthetic)
            loss.backward()

            print(loss.item())

            opt_g.step()

            print('z.shape =', z.shape)
            print('pc_real.shape =', pc_real.shape)
            print('quasi_probs.shape =', quasi_probs.shape)

    # TODO: This is a rather stupid way of visualising the output of
    # this training routine while writing it. I hope to *eventually*
    # get rid of this...
    pc = pc_synthetic.detach().numpy()[0]

    plt.scatter(pc[:, 0], pc[:, 1])
    plt.show()
