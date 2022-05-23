"""Layers for processing persistence diagrams."""

import torch


class StructureElementLayer(torch.nn.Module):
    def __init__(
        self,
        n_elements
    ):
        super().__init__()

        self.n_elements = n_elements
        self.dim = 2    # TODO: Make configurable

        size = (self.n_elements, self.dim)

        self.centres = torch.nn.Parameter(
            torch.rand(*size)
        )

        self.sharpness = torch.nn.Parameter(
            torch.ones(*size) * 3
        )

    def forward(self, x):
        batch = torch.cat([x] * self.n_elements, 1)

        B, N, D = x.shape

        # This is a 'butchered' variant of the much nicer `SLayerExponential`
        # class by C. Hofer and R. Kwitt.
        #
        # https://c-hofer.github.io/torchph/_modules/torchph/nn/slayer.html#SLayerExponential

        centres = torch.cat([self.centres] * N, 1)
        centres = centres.view(-1, self.dim)
        centres = torch.stack([centres] * B, 0)
        centres = torch.cat((centres, 2 * batch[..., -1].unsqueeze(-1)), 2)

        sharpness = torch.pow(self.sharpness, 2)
        sharpness = torch.cat([sharpness] * N, 1)
        sharpness = sharpness.view(-1, self.dim)
        sharpness = torch.stack([sharpness] * B, 0)
        sharpness = torch.cat(
            (
                sharpness,
                torch.ones_like(batch[..., -1].unsqueeze(-1))
            ),
            2
        )

        x = centres - batch
        x = x.pow(2)
        x = torch.mul(x, sharpness)
        x = torch.nansum(x, 2)
        x = torch.exp(-x)
        x = x.view(B, self.n_elements, -1)
        x = torch.sum(x, 2)
        x = x.squeeze()

        return x
