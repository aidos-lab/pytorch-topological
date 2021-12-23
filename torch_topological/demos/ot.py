"""Check optimal transport integration."""


import torch

from torch_topological.nn import WassersteinDistance


PD1 = [
    (3, 5),
    (1, 2),
    (3, 4)
]

PD2 = [
    (0, 0),
    (3, 4),
    (5, 3),
    (7, 4)
]


PD1 = torch.as_tensor(PD1, dtype=torch.float)
PD2 = torch.as_tensor(PD2, dtype=torch.float)
print(PD1)
print(PD2)
print(WassersteinDistance()._make_distance_matrix(PD1, PD2))
print('wd = ', WassersteinDistance()(PD1, PD2))
