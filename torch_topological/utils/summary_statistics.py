"""Summary statistics for persistence diagrams."""

import torch


def persistent_entropy(D, **kwargs):
    """Calculate persistent entropy of a persistence diagram.

    Parameters
    ----------
    D : `torch.tensor`
        Persistence diagram, assumed to be in shape `(n, 2)`, where each
        entry corresponds to a tuple of the form :math:`(x, y)`, with
        :math:`x` denoting the creation of a topological feature and
        :math:`y` denoting its destruction.

    Returns
    -------
    Persistent entropy of `D`.
    """
    persistence = torch.diff(D)
    persistence = persistence[torch.isfinite(persistence)].abs()

    P = persistence.sum()
    probabilities = persistence / P

    # Ensures that a probability of zero will just result in
    # a logarithm of zero as well. This is required whenever
    # one deals with entropy calculations.
    indices = probabilities > 0
    log_prob = torch.zeros_like(probabilities)
    log_prob[indices] = torch.log2(probabilities[indices])

    return torch.sum(-probabilities * log_prob)


def polynomial_function(D, p, q, **kwargs):
    r"""Parametrise polynomial function over persistence diagrams.

    This function follows an approach by Adcock et al. [Adcock16a]_ and
    parametrises a polynomial function over a persistence diagram.

    Parameters
    ----------
    D : `torch.tensor`
        Persistence diagram, assumed to be in shape `(n, 2)`, where each
        entry corresponds to a tuple of the form :math:`(x, y)`, with
        :math:`x` denoting the creation of a topological feature and
        :math:`y` denoting its destruction.

    p : int
        Exponent for persistence differences in the diagram.

    q : int
        Exponent for mean persistence in the diagram.

    Returns
    -------
    Sum of the form :math:`\sigma L^p * \mu^q`, with :math:`L` denoting
    an individual persistence value, and :math:`\mu` denoting its
    average persistence.

    References
    ----------
    .. [Adcock16a] A. Adcock et al., "The Ring of Algebraic Functions on
        Persistence Bar Codes", *Homology, Homotopy and Applications*,
        Volume 18, Issue 1, pp. 381--402, 2016.
    """
    lengths = torch.diff(D)
    means = torch.sum(D, dim=-1, keepdim=True) / 2

    # Filter out non-finite values; the same mask works here because the
    # mean is non-finite if and only if the persistence is.
    mask = torch.isfinite(lengths)
    lengths = lengths[mask]
    means = means[mask]

    return torch.sum(torch.mul(lengths.pow(p), means.pow(q)))


def total_persistence(D, p=2, **kwargs):
    """Calculate total persistence of a persistence diagram.

    This function will calculate the totla persistence of a persistence
    diagram. Infinite value will be ignored.

    Parameters
    ----------
    D : `torch.tensor`
        Persistence diagram, assumed to be in shape `(n, 2)`, where each
        entry corresponds to a tuple of the form :math:`(x, y)`, with
        :math:`x` denoting the creation of a topological feature and
        :math:`y` denoting its destruction.

    p : float
        Weight parameter for the total persistence calculation.

    Returns
    -------
    Total persistence of `D`.
    """
    persistence = torch.diff(D)
    persistence = persistence[torch.isfinite(persistence)]

    return persistence.abs().pow(p).sum()
