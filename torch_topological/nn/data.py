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
    :class:`PersistenceInformation` class.
    """

    __slots__ = ()


def make_tensor(x):
    """Create dense tensor representation from sparse inputs."""
    level = nesting_level(x)

    # Internal utility function for calculating the length of the output
    # tensor. This is required to ensure that all inputs can be *merged*
    # into a single output tensor.
    def _calculate_length(x, level):

        # Simple base case; should never occur in practice but let's be
        # consistent here.
        if len(x) == 0:
            return 0

        # `chain.from_iterable()` only removes one layer of nesting, but
        # we may have more.
        elif level > 2:
            x = list(chain.from_iterable(x))

        # Collect information that we need to create the full tensor. An
        # entry of the resulting list contains the length of the diagram
        # and the dimension, making it possible to derive padding values
        # for all entries.
        M = list(map(
            lambda a: (len(a.diagram), a.dimension), chain.from_iterable(x)
        ))

        # Get maximum dimension
        dim = max(M, key=itemgetter(1))[1]

        # Get *sum* of maximum number of entries for each dimension.
        # This is calculated over all batches.
        N = sum([
            max([L for L in M if L[1] == d], key=itemgetter(0))[0]
            for d in range(dim + 1)
        ])

        return N

    N = _calculate_length(x, level)

    # List of lists: the first axis is treated as the batch axis, while
    # the second axis is treated as the dimension of diagrams or pairs.
    if level == 2:
        tensors = [
            make_tensor_from_persistence_information(pers_infos)
            for pers_infos in x
        ]

        # Pad all tensors to length N in the first dimension, then turn
        # them into a batch.
        result = torch.stack(
                list(
                    map(
                        lambda t: torch.nn.functional.pad(
                                t,
                                (0, 0, N - len(t), 0),
                                mode='constant',
                                value=torch.nan),
                        tensors
                    )
                )
        )

        return result


def make_tensor_from_persistence_information(
    pers_info,
    extract_generators=False
):
    """Convert (sequence) of persistence information entries to tensor.

    This function converts instance(s) of :class:`PersistenceInformation`
    objects into a single tensor. No padding will be performed. A client
    may specify what type of information to extract from the object. For
    instance, by default, the function will extract persistence diagrams
    but this behaviour can be changed by setting `extract_generators` to
    `true`.

    Parameters
    ----------
    pers_info : :class:`PersistenceInformation` or iterable thereof
        Input persistence information object(s). The function is able to
        handle both single objects and sequences. This has no bearing on
        the length of the returned tensor.

    extract_generators : bool
        If set, extracts generators instead of persistence diagram from
        `pers_info`.

    Returns
    -------
    torch.tensor
        Tensor of shape `(n, 3)`, where `n` is the sum of all features,
        over all dimensions in the input `pers_info`. Each triple shall
        be of the form `(creation, destruction, dim)` for a persistence
        diagram. If the client requested generators to be returned, the
        first two entries of the triple refer to *indices* with respect
        to the input data set. Depending on the algorithm employed, the
        meaning of these indices can change. Please refer to the module
        used to calculate persistent homology for more details.
    """
    # Looks a little bit cumbersome, but since `namedtuple` is iterable
    # as well, we need to ensure that we are actually dealing with more
    # than one instance here.
    if len(pers_info) > 1 and not \
            isinstance(pers_info[0], PersistenceInformation):
        pers_info = [pers_info]

    # TODO: This might not always work since the size of generators
    # changes in different dimensions.
    if extract_generators:
        pairs = torch.cat(
            [torch.as_tensor(x.pairing, dtype=torch.float) for x in pers_info],
        ).long()
    else:
        pairs = torch.cat(
            [torch.as_tensor(x.diagram, dtype=torch.float) for x in pers_info],
        ).float()

    dimensions = torch.cat(
        [
            torch.as_tensor([x.dimension] * len(x.diagram), dtype=torch.long)
            for x in pers_info
        ]
    )

    result = torch.column_stack((pairs, dimensions))
    return result
