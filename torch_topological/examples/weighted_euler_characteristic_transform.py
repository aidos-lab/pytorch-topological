"""Demo for using the Weighted Euler Characteristic transform
in an optimiblzation routine.

This example demonstrates how the WECT can be used to optimize
a neural networks predictions to match the topological signature
of a target.
"""
from torch import nn
import torch
from torch_topological.nn import EulerDistance, WeightedEulerCurve
import torch.optim as optim


class NN(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim):
        super(NN, self).__init__()
        self.fc1 = torch.nn.Linear(inp_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x_):
        x = x_.clone()
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.nn.functional.sigmoid(x)
        out = int(self.out_dim ** (1 / 3))
        return x.reshape([out, out, out])


if __name__ == "__main__":
    torch.manual_seed(4)
    z = 3
    arr = torch.ones([z, z, z], requires_grad=False)
    model = NN(z * z * z, 100, z * z * z)
    arr2 = torch.rand([z, z, z], requires_grad=False)
    arr2[arr2 > 0.5] = 1
    arr2[arr2 <= 0.5] = 0
    ec = WeightedEulerCurve(prod=True)
    dist = EulerDistance()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    ans = 100
    while ans > 0.05:
        optimizer.zero_grad()
        ans = dist(ec(model(arr.flatten())), ec(arr2))
        ans.backward()
        optimizer.step()
        with torch.no_grad():
            print(
                "L2 distance:",
                dist(model(arr.flatten()), arr2),
                "   Euler Distance:",
                ans,
            )
