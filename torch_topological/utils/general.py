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


def wrap_if_not_iterable(x):
    """Wrap variable in case it cannot be iterated over.

    This function provides a convenience wrapper for variables that need
    to be iterated over. If the variable is already an `iterable`, there
    is nothing to be done and it will be returned as-is. Otherwise, will
    will 'wrap' the variable to be the single item of a list.

    The primary purpose of this function is to make it easier for users
    to interact with certain classes: essentially, one does not have to
    think any more about single inputs vs. `iterable` inputs.

    Parameters
    ----------
    x : any
        Input object.

    Returns
    -------
    list or type of x
        If `x` can be iterated over, `x` will be returned as-is. Else,
        will return `[x]`, i.e. a list containing `x`.

    Examples
    --------
    >>> wrap_if_not_iterable(1.0)
    [1.0]
    >>> wrap_if_not_iterable('Hello, World!')
    'Hello, World!'
    """
    if is_iterable(x):
        return x
    else:
        return [x]


def nesting_level(x):
    """Calculate nesting level of a list of objects.

    To convert between sparse and dense representations of topological
    features, we need to determine the nesting level of an input list.
    The nesting level is defined as the maximum number of times we can
    recurse into the object while still obtaining lists.

    Parameters
    ----------
    x : list
        Input list of objects.

    Returns
    -------
    int
        Nesting level of `x`. If `x` has no well-defined nesting level,
        for example because `x` is not a list of something, will return
        `0`.

    Notes
    -----
    This function is implemented recursively. It is therefore a bad idea
    to apply it to objects with an extremely high nesting level.

    Examples
    --------
    >>> nesting_level([1, 2, 3])
    1

    >>> nesting_level([[1, 2], [3, 4]])
    2
    """
    # This is really only supposed to work with lists. Anything fancier,
    # for example a `torch.tensor`, can already be used as a dense data
    # structure.
    if not isinstance(x, list):
        return 0

    # Empty lists have a nesting level of 1.
    if len(x) == 0:
        return 1
    else:
        return max(nesting_level(y) for y in x) + 1
