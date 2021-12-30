from torch_topological.datasets import SphereVsTorus

from torch_topological.nn import VietorisRipsComplex

from torch.utils.data import DataLoader

batch_size = 64
data_set = SphereVsTorus()
loader = DataLoader(
    data_set,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

vr = VietorisRipsComplex(dim=1)


class TestBatchHandling:
    def test_processing(self):
        for (x, y) in loader:
            pers_info = vr(x)

            assert pers_info is not None
            assert len(pers_info) == batch_size
