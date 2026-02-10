import torch
from torch_topological.nn import PersistenceInformation, LowerStarPersistence
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

batch_size = 64


class TestLowerStarPersistenceBatchHandling:

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ls_list = LowerStarPersistence(list_of_lists=True)
    ls_tuple = LowerStarPersistence(list_of_lists=False)

    def test_batch_processing(self):
        for x in self.loader:
            filtration = torch.rand(x.num_nodes)
            pers_info_list = self.ls_list(x, filtration)
            pers_info_tuple = self.ls_tuple(x, filtration)

            assert pers_info_list is not None
            assert pers_info_tuple is not None
            assert len(pers_info_list) == x.num_graphs

            for pers_info in pers_info_list:
                for p in pers_info:
                    assert isinstance(p, PersistenceInformation)

            for p in pers_info_tuple:
                assert isinstance(p[0], PersistenceInformation)
                assert torch.is_tensor(p[1])

    def test_batch(self):
        for x in self.loader:
            filtration = torch.rand(x.num_nodes)
            pers_info_list = self.ls_list(x, filtration)
            pers_info_tuple = self.ls_tuple(x, filtration)

            for pers_info in pers_info_list:
                for p in pers_info:
                    assert p.pairing.size(0) == p.diagram.size(0)
                    assert p.pairing.size(1) == 2 * p.dimension + 3
                    assert p.diagram.size(1) == 2

            for p in pers_info_tuple:
                assert p[0].pairing.size(0) == p[0].diagram.size(0)
                assert p[0].pairing.size(1) == 2 * p[0].dimension + 3
                assert p[0].diagram.size(1) == 2
                assert p[0].pairing.size(0) == p[1].size(0)


class TestLowerStarPersistence:

    simplex_0 = 4
    simplex_1 = torch.tensor([[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 2, 3]])
    simplex_2 = torch.tensor([[0, 0, 1], [1, 2, 2], [3, 3, 3]])
    simplex = [simplex_0, simplex_1, simplex_2]
    filtration = torch.tensor([0., 1., 2., 3.])
    ls = LowerStarPersistence()

    def test_simplex(self):
        pers_info = self.ls(self.simplex, self.filtration)
        diagram_0 = torch.tensor([[0., torch.inf]])
        diagram_1 = torch.tensor([[2., 3.]])
        assert torch.all(torch.isclose(pers_info[0].diagram, diagram_0)).item()
        assert torch.all(torch.isclose(pers_info[1].diagram, diagram_1)).item()
