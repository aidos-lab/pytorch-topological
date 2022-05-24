"""Alpha complex calculation module(s)."""

from torch import nn

from torch_topological.nn import PersistenceInformation

import gudhi
import torch

class AlphaComplex(nn.Module):
    """Calculate persistence diagrams of an alpha complex."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # TODO: Copied from `VietorisRipsComplex`; needs to be
        # refactored into a unified interface.
        #
        # Check whether individual batches need to be handled (3D array)
        # or not (2D array). We default to this type of processing for a
        # list as well.
        if isinstance(x, list) or len(x.shape) == 3:

            # TODO: This is rather ugly and inefficient but it is the
            # easiest workaround for now.
            return [
                self._forward(torch.as_tensor(x_)) for x_ in x
            ]
        else:
            return self._forward(torch.as_tensor(x))

    def _forward(self, x):
        alpha_complex = gudhi.alpha_complex.AlphaComplex(
            x.cpu().detach()
        )

        st = alpha_complex.create_simplex_tree()
        persistence_points = st.persistence()

        max_dim = x.shape[-1]

        return [
            self._extract_diagrams(persistence_points, dim)
            for dim in range(0, max_dim)
        ]

    def _extract_diagrams(self, persistence_points, dim):
        points = [
            torch.as_tensor(p) for d, p in persistence_points if d == dim
        ]

        persistence_diagram = torch.stack(points, 0)

        return PersistenceInformation(
                pairing=None,
                diagram=persistence_diagram,
                dimension=dim
        )
