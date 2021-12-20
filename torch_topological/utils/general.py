"""General utility functions.

This module contains generic utility functions that are used throughout
the code base. If a function is 'relatively small' and somewhat generic
it should be put here.
"""


def is_iterable(x):
    """Check whether variable is iterable.

    Parameters
    ----------
    x : any
        Input object.

    Returns
    -------
    bool
        `true` if `x` is iterable.
    """
    result = True

    # This is the most generic way; it also permits objects that only
    # implement the `__getitem__` interface.
    try:
        iter(x)
    except TypeError:
        result = False

    return result
