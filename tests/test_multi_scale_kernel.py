from torch_topological.data import sample_from_unit_cube

from torch_topological.nn import MultiScaleKernel
from torch_topological.nn import VietorisRipsComplex


class TestMultiScaleKernel:
    vr = VietorisRipsComplex(dim=1)
    kernel = MultiScaleKernel(1.0)
    X = sample_from_unit_cube(100)
    Y = sample_from_unit_cube(100)

    def test_pseudo_metric(self):
        pers_info_X, pers_info_Y = self.vr([self.X, self.Y])
        k_XX = self.kernel(pers_info_X, pers_info_X)
        k_YY = self.kernel(pers_info_Y, pers_info_Y)
        k_XY = self.kernel(pers_info_X, pers_info_Y)

        assert k_XY > 0
        assert k_XX > 0
        assert k_YY > 0

        assert k_XX + k_YY - 2 * k_XY >= 0
