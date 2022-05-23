"""Example of topology-based GANs."""

import torch
import torchvision

from torch_topological.nn import Cubical
from torch_topological.nn import SummaryStatisticLoss

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt


class Generator(torch.nn.Module):
    def __init__(self, latent_dim, shape):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = np.prod(shape)

        self.shape = shape

        def _make_layer(input_dim, output_dim):
            layers = [
                torch.nn.Linear(input_dim, output_dim),
                torch.nn.BatchNorm1d(output_dim, 0.8),
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


class TopologicalAdversarialLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.cubical = Cubical()
        self.loss = SummaryStatisticLoss('polynomial_function', p=2, q=2)

    def forward(self, real, synthetic):

        loss = 0.0

        for x, y in zip(real, synthetic):
            x = x.squeeze()
            y = y.squeeze()

            pi_real = self.cubical(x)[0]
            pi_synthetic = self.cubical(y)[0]

            loss += self.loss([pi_real], [pi_synthetic])

        return loss


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


if __name__ == '__main__':

    n_epochs = 5

    img_size = 16
    shape = (1, img_size, img_size)
    batch_size = 32
    latent_dim = 200

    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "./data/MNIST",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(img_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.5], [0.5])
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    generator = Generator(shape=shape, latent_dim=latent_dim)
    discriminator = Discriminator(shape=shape)
    adversarial_loss = torch.nn.BCELoss()
    topo_loss = TopologicalAdversarialLoss()

    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    for epoch in range(n_epochs):
        for batch, (imgs, _) in tqdm(enumerate(data_loader), desc='Batch'):
            z = torch.autograd.Variable(
                torch.Tensor(
                    np.random.normal(
                        0,
                        1,
                        (imgs.shape[0], latent_dim)
                    )
                )
            )

            real_labels = torch.as_tensor([1.0] * len(imgs)).view(-1, 1)
            fake_labels = torch.as_tensor([0.0] * len(imgs)).view(-1, 1)

            opt_g.zero_grad()

            imgs_synthetic = generator(z)

            generator_loss = adversarial_loss(
                discriminator(imgs_synthetic), real_labels
            ) + 0.01 * topo_loss(imgs, imgs_synthetic)

            generator_loss.backward()

            opt_g.step()

            opt_d.zero_grad()

            real_loss = adversarial_loss(discriminator(imgs), real_labels)
            fake_loss = adversarial_loss(
                discriminator(imgs_synthetic).detach(), fake_labels
            )

            discriminator_loss = 0.5 * (real_loss + fake_loss)
            discriminator_loss.backward()

            opt_d.step()

    output = imgs_synthetic.detach()
    grid = torchvision.utils.make_grid(output)

    show(grid)

    plt.show()
