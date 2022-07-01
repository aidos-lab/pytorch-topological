"""Filter functions for algorithmic outputs."""

import torch


class SelectByDimension(torch.nn.Module):
    """Select persistence diagrams by dimension.

    This is a simple selector that enables filtering the outputs of
    persistent homology calculation algorithms by dimension: often,
    one is really only interested in a *specific* dimension but the
    corresponding algorithm yields more diagrams. To this end, this
    module can be applied as a lightweight filter.
    """

    def __init__(self, min_dim, max_dim=None):
        """Prepare filter for subsequent usage.

        Provides the filter with the required parameters. A minimum
        dimension must be provided. There is also the option to use
        a maximum dimension, thus permitting filtering by ranges.

        Parameters
        ----------
        min_dim : int
            Minimum dimension to allow through the filter. If this is
            the sole provided parameter, only diagrams satisfying the
            dimension requirement will be selected.

        max_dim : int
            Optional upper dimension. If set, the selection returns a
            diagram whose dimension is within `[min_dim, max_dim]`.
        """
        super().__init__()

        self.min_dim = min_dim
        self.max_dim = max_dim

    def forward(self, X):
        """Apply selection parameters to input.

        Iterate over input and select diagrams according to the
        pre-defined parameters.

        Parameters
        ----------
        X : iterable of `PersistenceInformation`
            An iterable containing `PersistenceInformation` objects at
            its lowest level.

        Returns
        -------
        iterable
            Input, i.e. `X`, but with all non-matching persistence
            diagrams removed.
        """
        return [
            pers_info for pers_info in X if self._is_valid(pers_info)
        ]

    def _is_valid(self, pers_info):
        if self.max_dim is not None:
            return self.min_dim <= pers_info.dimension <= self.max_dim
        else:
            return self.min_dim == pers_info.dimension
