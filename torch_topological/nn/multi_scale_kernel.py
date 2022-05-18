import torch

from torch_topological.utils import wrap_if_not_iterable


class MultiScaleKernel(torch.nn.Module):
    # TODO: more detailed description
    r"""Implement the multi-scale kernel between two persistence diagrams

    This class implements the multi-scale kernel between two persistence
    diagrams as defined by Reininghaus et al. as

    .. math::
        k_\sigma(F,G) = \frac{1}{8 \pi \sigma} \sum_{\substack{p \in F\\q \in G}} exp{-\frac{\|p-q\|^2}{8\sigma}} - exp{-\frac{\|p-\overline{q}\|^2}{8\sigma}}

    where :math:`z=(z_1, z_2)` and :math:`\overline{z}=(z_2, z_1)`

    Attributes
    ----------
    sigma : float
        Scale parameter

    Methods
    -------
    forward(X, Y)
        Compute the kernel value between two persistence diagrams

    References
    ----------
    .. [Reininghaus15] J. Reininghaus, U. Bauer and R. Kwitt, "A Stable
       Multi-Scale Kernel for Topological Machine Learning", *Proceedings
       of the IEEE conference on computer vision and pattern recognition*,
       pp. 4741--4748, 2015.
    """
    def __init__(self, sigma):
        """Create a new MultiScaleKernel instance

        Parameters
        ----------
        sigma : float
            scale parameter of the kernel
        """
        super().__init__()

        self.sigma = sigma

    @staticmethod
    def _check_upper(d):
        """
        Check if all points in the diagram are above the diagonal.
        All points below the diagonal are 'swapped'.
        """
        is_upper = d[:, 0] < d[:, 1]
        if not torch.all(is_upper):
            d[~is_upper, 0] = d[~is_upper, 1]
            d[~is_upper, 1] = d[~is_upper, 0]
        return d

    @staticmethod
    def _mirror(x):
        """Mirror one or multiple points of a persistence diagram at the
        diagonal
        """
        if len(x.shape) > 1:
            return x[:, [1, 0]]
        # only a single point in the diagram
        return x[[1, 0]]

    @staticmethod
    def _dist(x, y):
        """Compute the pairwise distance between two persistence diagrams

        The returned tensor is the point-wise squared Euclidean distance
        """
        dist = torch.cdist(x, y, p=2)
        return dist.pow(2)

    def forward(self, X, Y):
        """Calculate the multi-scale kernel metric between two persistence
        diagrams

        Parameters
        ----------
        X : list or instance of :class:`PersistenceInformation`
            Topological features of the first space. Supposed to
            contain persistence diagrams and persistence pairings.

        Y : list or instance of :class:`PersistenceInformation`
            Topological features of the second space. Supposed to
            contain persistence diagrams and persistence pairings.

        Returns
        -------
        torch.tensor
            A single scalar tensor containing the kernel value between the
            persistence diagram(s) contained in `X` and `Y`.

        Examples
        --------
        >>> from torch_topological.data.shapes import sample_from_disk
        >>> from torch_topological.nn import VietorisRipsComplex
        >>> # sample randomly from two disks
        >>> x = sample_from_disk(r=0.5, R=0.6, n=100)
        >>> y = sample_from_disk(r=0.9, R=1.0, n=100)
        >>> # compute vietoris rips filtration for both point clouds
        >>> vr = VietorisRipsComplex(dim=1)
        >>> vr_x = vr(x)
        >>> vr_y = vr(y)
        >>> # compute multi-scale distance between persistence
        >>> # diagrams with sigma set to 1
        >>> msk = MultiScaleKernel(1.)
        >>> msk_value = msk(vr_x, vr_y)
        """
        X_ = wrap_if_not_iterable(X)
        Y_ = wrap_if_not_iterable(Y)

        k_sigma = 0.0

        for pers_info in zip(X_, Y_):
            # ensure that all points in the diagram are
            # above the diagonal
            D1 = self._check_upper(pers_info[0].diagram)
            D2 = self._check_upper(pers_info[1].diagram)

            # compute the pairwise distances between the
            # two diagrams
            nom = self._dist(D1, D2)
            # distance between diagram 1 and mirrored
            # diagram 2
            denom = self._dist(D1, self._mirror(D2))

            M = torch.exp(-nom) / (8 * self.sigma)
            M -= torch.exp(-denom) / (8 * self.sigma)

            # sum over all points
            k_sigma += M.sum() / (8. * self.sigma * torch.pi)

        return k_sigma
