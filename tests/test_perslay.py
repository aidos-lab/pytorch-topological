import torch
from torch_topological.datasets import SphereVsTorus
from torch_topological.nn import VietorisRipsComplex
from torch.utils.data import DataLoader
from torch_topological.nn import (PersLay,
                                  PermutationEquivariant,
                                  Image,
                                  Landscape,
                                  BettiCurve,
                                  Entropy,
                                  Exponential,
                                  Rational,
                                  RationalHat)

batch_size = 64
num_barcodes = 30
image_size = (10, 20)
output_dim = 10


class TestPersLayPersInfoHandling:
    dataset = SphereVsTorus(n_point_clouds=3 * batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    output_dim = 10

    vr = VietorisRipsComplex()

    def test_processing(self):
        pl = PersLay(self.output_dim, "Landscape", "max")
        for x, _ in self.loader:
            pers_info = self.vr(x)
            output = pl(pers_info)

            assert pers_info is not None
            assert output is not None
            assert len(output) == batch_size


class TestPersLay:

    data_x = torch.rand((batch_size, num_barcodes, 1))
    data_y = data_x + torch.rand((batch_size, num_barcodes, 1))
    data = torch.cat((data_x, data_y), dim=-1)

    def test_permutaion_equivariant_layer(self):
        layer = PermutationEquivariant(output_dim)
        output = layer(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size,
                                            num_barcodes,
                                            output_dim])

    def test_image_layer(self):
        layer = Image(image_size)
        output = layer(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size,
                                            num_barcodes,
                                            image_size[1],
                                            image_size[0]])

    def test_landscape_layer(self):
        layer = Landscape(output_dim)
        output = layer(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size,
                                            num_barcodes,
                                            output_dim])

    def test_betti_curve_layer(self):
        layer = BettiCurve(output_dim)
        output = layer(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size,
                                            num_barcodes,
                                            output_dim])

    def test_entropy_layer(self):
        layer = Entropy(output_dim)
        output = layer(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size,
                                            num_barcodes,
                                            output_dim])

    def test_exponential_layer(self):
        layer = Exponential(output_dim)
        output = layer(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size,
                                            num_barcodes,
                                            output_dim])

    def test_rational_layer(self):
        layer = Rational(output_dim)
        output = layer(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size,
                                            num_barcodes,
                                            output_dim])

    def test_rational_hat_layer(self):
        layer = RationalHat(output_dim)
        output = layer(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size,
                                            num_barcodes,
                                            output_dim])

    def test_max_op(self):
        pl = PersLay(output_dim, "Landscape", "max")
        output = pl(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size, output_dim])

    def test_min_op(self):
        pl = PersLay(output_dim, "Landscape", "min")
        output = pl(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size, output_dim])

    def test_sum_op(self):
        pl = PersLay(output_dim, "Landscape", "sum")
        output = pl(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size, output_dim])

    def test_mean_op(self):
        pl = PersLay(output_dim, "Landscape", "mean")
        output = pl(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size, output_dim])

    def test_topk_op(self):
        pl = PersLay(output_dim, "Landscape", "topk", k=num_barcodes // 2)
        output = pl(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size, output_dim])

    def test_image_op(self):
        pl = PersLay(image_size, "Image", "max")
        output = pl(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size,
                                            image_size[1],
                                            image_size[0]])
