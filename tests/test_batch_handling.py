from torch_topological.datasets import SphereVsTorus

from torch_topological.nn import VietorisRipsComplex

from torch.utils.data import DataLoader

data_set = SphereVsTorus()
loader = DataLoader(data_set, batch_size=64, shuffle=True)

vr = VietorisRipsComplex(dim=1)

def test_batch_handling():
    for (x, y) in loader:
        pers_info = vr(x)
