"""Vietoris--Rips complex calculation module(s)."""

from gph import ripser_parallel
from torch import nn

import torch


class VietorisRips(nn.Module):
    """Calculate Vietoris--Rips persistence diagrams.

    This module calculates 'differentiable' persistence diagrams for
    point clouds. The underlying topological approximations are done
    by calculating a Vietoris--Rips complex of the data.
    """

    def __init__(self, dim=1, p=2):
        """Initialise new module.

        Parameters
        ----------
        dim : int
            Calculates persistent homology up to (and including) the
            prescribed dimension.

        p : float
            Exponent for the `p`-norm calculation of distances.

        Notes
        -----
        This module currently only supports Minkowski norms. It does not
        yet support other metrics.
        """
        super().__init__()

        self.dim = dim
        self.p = p

        # Ensures that the same parameters are used whenever calling
        # `ripser`.
        self.ripser_params = {
            'return_generators': True,
            'maxdim': self.dim,
        }

    def forward(self, x):
        """Implement forward pass for persistence diagram calculation.

        The forward pass entails calculating persistent homology on
        a point cloud and returning a set of persistence diagrams.

        Parameters
        ----------
        x : `np.array` or `torch.tensor`
            Input point cloud

        Returns
        -------
        List of tuples of the form `(gen, pd)`, where `gen` refers to
        the set of generators for the respective dimension, while `pd`
        denotes the persistence diagram.
        """
        generators = ripser_parallel(
            x.detach(),
            **self.ripser_params
        )['gens']

        # TODO: Is this always required? Can we calculate this in
        # a smarter fashion?
        distances = torch.cdist(x, x, p=self.p)

        # We always have 0D information.
        persistence_information = \
            self._extract_generators_and_diagrams(
                distances,
                generators,
                dim0=True,
            )

        # Check whether we have any higher-dimensional information that
        # we should return.
        if self.dim >= 1:
            persistence_information.extend(
                self._extract_generators_and_diagrams(
                    distances,
                    generators,
                    dim0=False,
                )
            )

        return persistence_information

    def _extract_generators_and_diagrams(
            self,
            dist,
            gens,
            finite=True,
            dim0=False
    ):
        """Extract generators and persistence diagrams from raw data.

        This convenience function translates between the output of
        `ripser_parallel` and the required output of this function.
        """
        index = 1 if not dim0 else 0
        gens = gens[index]

        # TODO: Handling of infinite features not provided yet, but the
        # index shift is already correct.
        if not finite:
            index += 1

        if dim0:
            # In a Vietoris--Rips complex, all vertices are created at
            # time zero.
            creators = torch.zeros_like(torch.as_tensor(gens)[:, 0])
            destroyers = dist[gens[:, 1], gens[:, 2]]

            persistence_diagram = torch.stack(
                (creators, destroyers), 1
            )

            return [(gens, persistence_diagram)]
        else:
            result = []

            for gens_ in gens:
                creators = dist[gens_[:, 0], gens_[:, 1]]
                destroyers = dist[gens_[:, 2], gens_[:, 3]]

                persistence_diagram = torch.stack(
                    (creators, destroyers), 1
                )

                result.append((gens_, persistence_diagram))

        return result
