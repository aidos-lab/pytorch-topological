import torch


from torch_topological.nn import PersistenceInformation
from torch_topological.nn import WassersteinDistance


def wrap(diagram):
    diagram = torch.as_tensor(diagram, dtype=torch.float)
    return [
        PersistenceInformation([], diagram)
    ]


class TestWassersteinDistance:

    def test_simple(self):
        X = [
            (3, 5),
            (1, 2),
            (3, 4)
        ]

        Y = [
            (0, 0),
            (3, 4),
            (5, 3),
            (7, 4)
        ]

        X = wrap(X)
        Y = wrap(Y)

        dist = WassersteinDistance()(X, Y)
        assert dist > 0.0
