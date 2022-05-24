"""Alpha complex calculation module(s)."""

from torch import nn

from torch_topological.nn import PersistenceInformation

import gudhi
import torch

class AlphaComplex(nn.Module):
    """Calculate persistence diagrams of an alpha complex."""

    def __init__(self):
        super().__init__()

        # TODO: Make configurable
        self.p = 2

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
        st.persistence()
        persistence_pairs = st.persistence_pairs()

        max_dim = 0  # TODO: Should be x.shape[-1]

        dist = torch.cdist(x, x, p=self.p)

        return [
            self._extract_generators_and_diagrams(
                dist,
                persistence_pairs,
                dim
            )
            for dim in range(0, max_dim + 1)
        ]

    def _extract_generators_and_diagrams(self, dist, persistence_pairs, dim):
        pairs = [
            torch.cat((torch.as_tensor(p[0]), torch.as_tensor(p[1])), 0)
            for p in persistence_pairs if len(p[0]) == (dim + 1)
        ]

        # TODO: Ignore infinite features for now. Will require different
        # handling in the future.
        pairs = [p for p in pairs if len(p) == 2 * dim + 3]

        # Create tensor of shape `(n, 2 * dim + 3)`, with `n` being the
        # number of finite persistence pairs.
        pairs = torch.stack(pairs)

        if dim == 0:
            creators = torch.zeros_like(torch.as_tensor(pairs)[:, 0])
            destroyers = dist[pairs[:, 1], pairs[:, 2]]

            persistence_diagram = torch.stack(
                (creators, destroyers), 1
            )

            return PersistenceInformation(
                    pairing=pairs,
                    diagram=persistence_diagram,
                    dimension=0
            )

        # Iterate over the flag complex in order to get (a) the distance
        # of the creator simplex, and (b) the distance of the destroyer.
        # We *cannot* look this information up in the filtration itself,
        # because we have to use gradient-imbued information such as the
        # set of pairwise distances.
        else:
            pass

        #return PersistenceInformation(
        #        pairing=None,
        #        diagram=persistence_diagram,
        #        dimension=dim
        #)
