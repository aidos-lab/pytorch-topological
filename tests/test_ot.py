import numpy as np

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

    def test_random(self):
        n_points = 10
        n_instances = 10

        for i in range(n_instances):
            X = np.random.default_rng().normal(size=(n_points, 2))
            Y = np.random.default_rng().normal(size=(n_points, 2))

            X = wrap(X)
            Y = wrap(Y)

            dist = WassersteinDistance()(X, Y)
            assert dist > 0.0

    def test_almost_zero(self):
        n_points = 100

        X = np.random.default_rng().uniform(-1e-16, 1e-11, size=(n_points, 2))
        Y = np.random.default_rng().uniform(-1e-16, 1e-11, size=(n_points, 2))

        X = wrap(X)
        Y = wrap(Y)

        dist = WassersteinDistance()(X, Y)
        assert dist > 0.0
