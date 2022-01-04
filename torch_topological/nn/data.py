"""Data structures for persistent homology calculations."""

from collections import namedtuple

from itertools import chain

from operator import itemgetter

from torch_topological.utils import is_iterable
from torch_topological.utils import nesting_level

import torch


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


def make_tensor(x):
    """Create dense tensor representation from sparse inputs."""
    level = nesting_level(x)

    # List of lists: the first axis is treated as the batch axis, while
    # the second axis is treated as the dimension of diagrams or pairs.
    if level == 2:
        B = len(x)

        # Collect information that we need to create the full tensor. An
        # entry of the resulting list contains the length of the diagram
        # and the dimension, making it possible to derive padding values
        # for all entries.
        M = list(map(
            lambda a: (len(a.diagram), a.dimension), chain.from_iterable(x)
        ))

        print(list(M))

        N = max(M, key=itemgetter(0))[0]
        D = max(M, key=itemgetter(1))[1]

        print(B, N, D)

        make_tensor_from_persistence_information(x[0][0])


def make_tensor_from_persistence_information(pers_info):
    """Convert (sequence) of persistence information entries to tensor."""
    if is_iterable(pers_info) or \
            isinstance(pers_info[0], PersistenceInformation):
        pers_info = [pers_info]

    pairs = torch.cat(
        [torch.as_tensor(x.diagram, dtype=torch.float) for x in pers_info]
    )

    dimensions = torch.cat(
        [
            torch.as_tensor([x.dimension] * len(x.diagram), dtype=torch.float)
            for x in pers_info
        ]
    )

    result = torch.column_stack((pairs, dimensions))
    return result
