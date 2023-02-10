"""Debug script for benchmarking some calculations."""

import time
import torch
import sys

from torch_topological.nn import WassersteinDistance
from torch_topological.nn import VietorisRipsComplex


if __name__ == "__main__":
    X = torch.load(sys.argv[1])

    vr = VietorisRipsComplex(dim=0)
    W1 = WassersteinDistance()

    pre = time.perf_counter()

    X_pi = vr(X)
    Y_pi = vr(X)

    dists = torch.stack([W1(x_pi, y_pi) for x_pi, y_pi in zip(X_pi, Y_pi)])
    dist = dists.mean()

    cur = time.perf_counter()
    print(f"{cur - pre:.4f}s")
