from torchvision.datasets import MNIST

from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor

from torch_topological.datasets import SphereVsTorus

from torch_topological.nn.data import make_tensor

from torch_topological.nn import CubicalComplex
from torch_topological.nn import VietorisRipsComplex

from torch.utils.data import DataLoader

batch_size = 64


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
        shuffle=True,
        drop_last=True,
    )

    cc = CubicalComplex()

    def test_processing(self):
        for (x, y) in self.loader:
            pers_info = self.cc(x)

            assert pers_info is not None
            assert len(pers_info) == batch_size

            pers_info_dense = make_tensor(pers_info)

            assert pers_info_dense is not None
