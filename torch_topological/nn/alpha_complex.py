"""Alpha complex calculation module(s)."""

from torch import nn

from torch_topological.nn import PersistenceInformation

import gudhi
import itertools
import torch


class AlphaComplex(nn.Module):
    """Calculate persistence diagrams of an alpha complex."""

    def __init__(self, p=2):
        """Initialise new alpha complex calculation module.

        Parameters
        ----------
        p : float
            Exponent for the `p`-norm calculation of distances.

        Notes
        -----
        This module currently only supports Minkowski norms. It does not
        yet support other metrics.
        """
        super().__init__()

        self.p = p

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
            x.cpu().detach(),
            precision='fast',
        )

        st = alpha_complex.create_simplex_tree()
        st.persistence()
        persistence_pairs = st.persistence_pairs()

        max_dim = x.shape[-1]
        dist = torch.cdist(x.contiguous(), x.contiguous(), p=self.p)

        return [
            self._extract_generators_and_diagrams(
                dist,
                persistence_pairs,
                dim
            )
            for dim in range(0, max_dim)
        ]

    def _extract_generators_and_diagrams(self, dist, persistence_pairs, dim):
        pairs = [
            torch.cat((torch.as_tensor(p[0]), torch.as_tensor(p[1])), 0)
            for p in persistence_pairs if len(p[0]) == (dim + 1)
        ]

        # TODO: Ignore infinite features for now. Will require different
        # handling in the future.
        pairs = [p for p in pairs if len(p) == 2 * dim + 3]

        if not pairs:
            return PersistenceInformation(
                    pairing=[],
                    diagram=[],
                    dim=dim
            )

        # Create tensor of shape `(n, 2 * dim + 3)`, with `n` being the
        # number of finite persistence pairs.
        pairs = torch.stack(pairs)

        if dim == 0:
            creators = torch.zeros_like(torch.as_tensor(pairs)[:, 0])
            destroyers = dist[pairs[:, 1], pairs[:, 2]]

        # Iterate over the flag complex in order to get (a) the distance
        # of the creator simplex, and (b) the distance of the destroyer.
        # We *cannot* look this information up in the filtration itself,
        # because we have to use gradient-imbued information such as the
        # set of pairwise distances.
        else:
            creators = torch.stack(
                    [
                        self._get_filtration_weight(creator, dist)
                        for creator in pairs[:, :dim+1]
                    ]
            )
            destroyers = torch.stack(
                    [
                        self._get_filtration_weight(destroyer, dist)
                        for destroyer in pairs[:, dim+1:]
                    ]
            )

        # Create the persistence diagram from creator and destroyer
        # information. This step is the same for all dimensions.

        persistence_diagram = torch.stack(
            (creators, destroyers), 1
        )

        return PersistenceInformation(
                pairing=pairs,
                diagram=persistence_diagram,
                dimension=dim
        )

    def _get_filtration_weight(self, simplex, dist):
        """Auxiliary function for querying simplex weights.

        This function returns the filtration weight of an arbitrary
        simplex under a distance-based filtration, i.e. the maximum
        weight of its cofaces. This function is crucial for getting
        persistence diagrams that are differentiable.

        Parameters
        ----------
        simplex : torch.tensor
            Simplex to query; must be a sequence of numbers, i.e. of
            shape `(n, 1)` or of shape `(n, )`.

        dist : torch.tensor
            Matrix of pairwise distances between points.

        Returns
        -------
        torch.tensor
            Scalar tensor containing the filtration weight.
        """
        weights = torch.as_tensor([
            dist[edge] for edge in itertools.combinations(simplex, 2)
        ])

        # TODO: Might have to be adjusted depending on filtration
        # ordering?
        return torch.max(weights)
