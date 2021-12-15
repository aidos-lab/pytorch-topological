"""Demo for topology-regularised autoencoders."""

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from torch_topological.data import create_sphere_dataset

from torch_topological.nn import SignatureLoss
from torch_topological.nn import VietorisRips


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

    def forward(self, x):
        z = self.model.encode(x)

        # TODO: I don't like this syntax at all.
        pi_x, _ = VietorisRips(x)()
        pi_z, _ = VietorisRips(z)()

        geom_loss = self.model(x)
        topo_loss = self.loss([x, pi_x], [z, pi_z])

        loss = geom_loss + self.lam * topo_loss
        return loss



if __name__ == '__main__':
    X, y = create_sphere_dataset(n_samples=50, n_spheres=3)
    X = torch.as_tensor(X, dtype=torch.float)

    model = LinearAutoencoder(input_dim=X.shape[1])
    topo_model = TopologicalAutoencoder(model, lam=0.1)

    optimizer = optim.Adam(topo_model.parameters(), lr=0.1)

    for i in range(10):
        topo_model.train()

        loss = topo_model(X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    Z = model.encode(X).detach().numpy()

    plt.scatter(
        Z[:, 0], Z[:, 1],
        c=y,
        cmap='Set1',
        marker='.',
        alpha=0.9, s=10.0
    )
    plt.show()
