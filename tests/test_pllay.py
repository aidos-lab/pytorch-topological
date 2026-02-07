import torch
from torch_topological.nn import PLLay

batch_size = 64
num_barcodes = 30
output_dim = 10
t_min = 0.01
t_max = 4
m = 50
K_max = 5


class TestPLLay:

    data_x = torch.rand(batch_size, num_barcodes, 1)
    data_y = data_x + torch.rand(batch_size, num_barcodes, 1) * 2
    data = torch.cat((data_x, data_y), dim=-1)

    def test_pllay_affine(self):
        pllay = PLLay(output_dim, K_max, t_min, t_max, m, layer="Affine")
        output = pllay(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size, output_dim])

    def test_pllay_log(self):
        pllay = PLLay(output_dim, K_max, t_min, t_max, m, layer="Logarithmic")
        output = pllay(self.data)

        assert output is not None
        assert output.size() == torch.Size([batch_size, output_dim])
