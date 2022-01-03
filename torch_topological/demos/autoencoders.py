"""Demo for topology-regularised autoencoders."""

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from tqdm import tqdm

from torch.utils.data import DataLoader

from torch_topological.datasets import Spheres

from torch_topological.nn import SignatureLoss
from torch_topological.nn import VietorisRipsComplex


class LinearAutoencoder(torch.nn.Module):
    """Simple linear autoencoder."""
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.latent_dim)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.input_dim)
        )

        self.loss = torch.nn.MSELoss()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        reconstruction_error = self.loss(x, x_hat)
        return reconstruction_error


class TopologicalAutoencoder(torch.nn.Module):
    """Wrapper for a topologically-regularised autoencoder."""
    def __init__(self, model, lam=1.0):
        super().__init__()

        self.lam = lam
        self.model = model
        self.loss = SignatureLoss()

        # TODO: Decrease dimensionality...
        self.vr = VietorisRipsComplex(dim=0)

    def forward(self, x):
        z = self.model.encode(x)

        pi_x = self.vr(x)
        pi_z = self.vr(z)

        geom_loss = self.model(x)
        topo_loss = self.loss([x, pi_x], [z, pi_z])

        loss = geom_loss + self.lam * topo_loss
        return loss


if __name__ == '__main__':
    n_spheres = 11
    data_set = Spheres(n_spheres=n_spheres)

    train_loader = DataLoader(
        data_set,
        batch_size=32,
        shuffle=True,
        drop_last=True
    )

    model = LinearAutoencoder(input_dim=data_set.dimension)
    topo_model = TopologicalAutoencoder(model, lam=10)

    optimizer = optim.Adam(topo_model.parameters(), lr=1e-2)

    n_epochs = 5

    progress = tqdm(range(n_epochs))

    for i in progress:
        topo_model.train()

        for batch, (x, y) in enumerate(train_loader):
            loss = topo_model(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        progress.set_postfix(loss=loss.item())

    data_set = Spheres(
        train=False,
        n_samples=2000,
        n_spheres=n_spheres,
    )

    test_loader = DataLoader(
            data_set,
            shuffle=False,
            batch_size=len(data_set)
    )

    X, y = next(iter(test_loader))
    Z = model.encode(X).detach().numpy()

    plt.scatter(
        Z[:, 0], Z[:, 1],
        c=y,
        cmap='Set1',
        marker='o',
        alpha=0.9,
    )
    plt.show()
