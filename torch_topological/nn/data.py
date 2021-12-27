"""Data structures for persistent homology calculations."""

from collections import namedtuple


class PersistenceInformation(namedtuple(
            'PersistenceInformation',
            [
                'pairing',
                'diagram',
                'dimension',
            ],
            # Ensures that there is always a dimension specified, albeit an
            # 'incorrect' one.
            defaults=[None]
        )
):
    """Persistence information data structure.

    This is a light-weight data structure for carrying information about
    the calculation of persistent homology. It consists of the following
    components:

    - A *persistence pairing*
    - A *persistence diagram*
    - An (optional) *dimension*

    Due to its lightweight nature, no validity checks are performed, but
    all calculation modules should return a sequence of instances of the
    `PersistenceInformation` class.
    """

    __slots__ = ()
