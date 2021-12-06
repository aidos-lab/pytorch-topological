"""Summary statistics for persistence diagrams."""


def total_persistence(D, p=2):
    """Calculate total persistence of a persistence diagram.

    Parameters
    ----------
    D : `np.array`
        Persistence diagram, assumed to be in the usual `giotto-ph`
        format: each entry is supposed to be a tuple of the form $(x, y,
        d)$, with $(x, y)$ being the usual creation--destruction pair,
        and $d$ denoting the dimension.

    p : float
        Weight parameter for the total persistence calculation.

    Returns
    -------
    Total persistence of `D`.
    """
    persistence = np.diff(D[:, 0:2])
    persistence = persistence[np.isfinite(persistence)]

    # TODO: Normalise?
    return np.sum(np.power(np.abs(persistence), p))
