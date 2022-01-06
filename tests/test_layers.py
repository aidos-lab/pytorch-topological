from torch_topological.datasets import SphereVsTorus

from torch_topological.nn.data import make_tensor

from torch_topological.nn import VietorisRipsComplex

from torch.utils.data import DataLoader

batch_size = 64


class TestStructureElementLayer:
    data_set = SphereVsTorus(n_point_clouds=batch_size)
    loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    vr = VietorisRipsComplex(dim=1)

    def test_processing(self):
        for (x, y) in self.loader:
            pers_info = make_tensor(self.vr(x))

            assert pers_info is not None
