"""Summary statistics for persistence diagrams."""


import torch


def total_persistence(D, p=2):
    """Calculate total persistence of a persistence diagram.

    This function will calculate the totla persistence of a persistence
    diagram. Infinite value will be ignored.

    Parameters
    ----------
    D : `torch.tensor`
        Persistence diagram, assumed to be in shape `(n, 2)`, where each
        entry corresponds to a tuple of the form $(x, y)$, with $x$
        denoting the creation of a topological feature and $y$ denoting
        its destruction.

    p : float
        Weight parameter for the total persistence calculation.

    Returns
    -------
    Total persistence of `D`.
    """
    persistence = torch.diff(D)
    persistence = persistence[torch.isfinite(persistence)]

    return persistence.abs().pow(p).sum()


def persistent_entropy(D):
    """Calculate persistent entropy of a persistence diagram.

    Parameters
    ----------
    D : `torch.tensor`
        Persistence diagram, assumed to be in shape `(n, 2)`, where each
        entry corresponds to a tuple of the form $(x, y)$, with $x$
        denoting the creation of a topological feature and $y$ denoting
        its destruction.

    Returns
    -------
    Persistent entropy of `D`.
    """
    persistence = torch.diff(D)
    persistence = persistence[torch.isfinite(persistence)].abs()

    P = persistence.sum()
    probabilities = torch.as_tensor([
        p / P for p in persistence
    ])

    # Ensures that a probability of zero will just result in
    # a logarithm of zero as well. This is required whenever
    # one deals with entropy calculations.
    indices = probabilities > 0
    log_prob = torch.zeros_like(probabilities)
    log_prob[indices] = torch.log2(probabilities[indices])

    return torch.sum(-probabilities * log_prob)
