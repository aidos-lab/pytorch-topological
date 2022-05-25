from torchvision.datasets import MNIST

from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor

from torch_topological.datasets import SphereVsTorus

from torch_topological.nn.data import batch_iter
from torch_topological.nn.data import make_tensor
from torch_topological.nn.data import PersistenceInformation

from torch_topological.nn import AlphaComplex
from torch_topological.nn import CubicalComplex
from torch_topological.nn import VietorisRipsComplex

from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

import numpy as np

batch_size = 64


class TestAlphaComplexBatchHandling:
    data_set = SphereVsTorus(n_point_clouds=3 * batch_size)
    loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    ac = AlphaComplex()

    def test_processing(self):
        for (x, y) in self.loader:
            pers_info = self.ac(x)

            assert pers_info is not None
            assert len(pers_info) == batch_size

            pers_info_dense = make_tensor(pers_info)

            assert pers_info_dense is not None


class TestCubicalComplexBatchHandling:
    data_set = MNIST(
        './data/MNIST',
        train=False,
        transform=Compose(
            [
                ToTensor(),
                Normalize([0.5], [0.5])
            ]
        ),
    )

    loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        sampler=RandomSampler(
            data_set,
            replacement=True,
            num_samples=100
        )
    )

    cc = CubicalComplex()

    def test_processing(self):
        for (x, y) in self.loader:
            pers_info = self.cc(x)

            assert pers_info is not None
            assert len(pers_info) == batch_size

            pers_info_dense = make_tensor(pers_info)

            assert pers_info_dense is not None

    def test_batch_iter(self):
        for (x, y) in self.loader:
            pers_info = self.cc(x)

            assert pers_info is not None
            assert len(pers_info) == batch_size

            assert sum(1 for x in batch_iter(pers_info)) == batch_size

            for x in batch_iter(pers_info):
                for y in x:
                    assert isinstance(y, PersistenceInformation)

            for x in batch_iter(pers_info, dim=0):

                # Make sure that we have something to iterate over.
                assert sum(1 for y in x) != 0

                for y in x:
                    assert isinstance(y, PersistenceInformation)
                    assert y.dimension == 0


class TestVietorisRipsComplexBatchHandling:
    data_set = SphereVsTorus(n_point_clouds=3 * batch_size)
    loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    vr = VietorisRipsComplex(dim=1)

    def test_processing(self):
        for (x, y) in self.loader:
            pers_info = self.vr(x)

            assert pers_info is not None
            assert len(pers_info) == batch_size

            pers_info_dense = make_tensor(pers_info)

            assert pers_info_dense is not None

    def test_ragged_processing(self):
        rng = np.random.default_rng()

        data = [
            np.random.default_rng().uniform(size=(rng.integers(32, 64), 8))
            for _ in range(batch_size)
        ]

        pers_info = self.vr(data)

        assert pers_info is not None
        assert len(pers_info) == batch_size

    def test_batch_iter(self):
        for (x, y) in self.loader:
            pers_info = self.vr(x)

            assert pers_info is not None
            assert len(pers_info) == batch_size

            # This is just to confirm that we can properly iterate over
            # this batch. Here, `batch_iter` is a little bit like `NoP`,
            # but in general, more complicated nested structures may be
            # present.
            assert sum(1 for x in batch_iter(pers_info)) == batch_size

            for x in batch_iter(pers_info):
                for y in x:
                    assert isinstance(y, PersistenceInformation)

            for x in batch_iter(pers_info, dim=0):

                # Make sure that we have something to iterate over.
                assert sum(1 for y in x) != 0

                for y in x:
                    assert isinstance(y, PersistenceInformation)
                    assert y.dimension == 0
