"""Demo for topology-regularised autoencoders."""

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from torch_topological.data import create_sphere_dataset


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


if __name__ == '__main__':
    X, y = create_sphere_dataset()
    X = torch.as_tensor(X, dtype=torch.float)

    model = LinearAutoencoder(input_dim=X.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for i in range(500):
        model.train()

        loss = model(X)

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
