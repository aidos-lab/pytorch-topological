from torch_topological.datasets import SphereVsTorus

from torch_topological.nn.data import make_tensor

from torch_topological.nn import AlphaComplex

from torch.utils.data import DataLoader

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
